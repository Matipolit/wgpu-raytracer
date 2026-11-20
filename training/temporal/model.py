"""Temporal neural denoiser: learned blend + à-trous kernels over demodulated radiance.

Per frame (see `step`):
  1. demodulate:  e = noisy / (albedo + EPS)
  2. reproject:   bilinear-warp last frame's output and guides with the motion
                  vectors; validity from distance / normal / hit-kind agreement
  3. network:     3x3 conv -> ReLU -> 3x3 conv -> ReLU -> 1x1 head giving a
                  temporal blend weight and softmax 3x3 kernels for à-trous
                  levels with dilations 1, 2, 4
  4. filter:      x = a*warped + (1-a)*e, then the three kernel levels
  5. output:      y (demodulated) is next frame's history; radiance = y*(albedo+EPS)

The WGSL inference in export.py implements exactly these operations.

Second-generation options (`configure`; all off is the model above; export.py generates
all of them except `hidden`):
  edge      explicit edge stops in the kernel logits: a neighbour loses weight with
            its relative depth difference, normal difference and albedo difference
            (learned scales per level), so kernels stop at edges by construction
  length    the history carries its length N; the blend defaults to the running mean
            N / (N + 1) (the head corrects it) and N is an input feature
  feedback  "blend": the unfiltered accumulation is the history (the kernels filter
            a copy for display), instead of re-filtering the filtered output
  hidden    a ReLU layer of this many units in the per-pixel head
  eps       the demodulation floor (0.01 lets dark, glossy surfaces reach a gain of 100)
`step(..., exposure)` scales radiance before the log features (exposure-normalized
inputs); None is the old behaviour.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.01
FEATURES = 15
# Architecture; `configure` changes these before building a Net (saved as config.json).
DILATIONS = (1, 2, 4)
CHANNELS = 16
HEAD = 1 + 9 * len(DILATIONS)
# Half resolution: features are 2x2 average-pooled for the two 3x3 convs; the
# 1x1 head runs per full-resolution pixel on [upsampled hidden, full-res features].
HALF = False
EDGE = False
LENGTH = False
FEEDBACK = "filtered"
HIDDEN = 0
MAX_LENGTH = 32.0


def configure(channels=16, dilations=(1, 2, 4), half=False, edge=False, length=False, feedback="filtered", hidden=0, eps=0.01, **_):
    global CHANNELS, DILATIONS, HEAD, HALF, EDGE, LENGTH, FEEDBACK, HIDDEN, EPS, FEATURES
    assert channels % 4 == 0 and feedback in ("filtered", "blend")
    CHANNELS, DILATIONS, HALF = channels, tuple(dilations), half
    EDGE, LENGTH, FEEDBACK, HIDDEN, EPS = edge, length, feedback, hidden, eps
    FEATURES = 15 + int(length)
    HEAD = 1 + 9 * len(DILATIONS)


def settings():
    """The configuration, as saved in config.json."""
    return dict(channels=CHANNELS, dilations=list(DILATIONS), half=HALF, edge=EDGE, length=LENGTH, feedback=FEEDBACK, hidden=HIDDEN, eps=EPS)

# Guide channel layout in captured frames (after the 3 noisy channels).
ALBEDO, DIST, NORMAL, ROUGH, MOTION, METAL, KIND = slice(0, 3), 3, slice(4, 7), 7, slice(8, 10), 10, 11


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(FEATURES, CHANNELS, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1, padding_mode="replicate")
        inputs = CHANNELS + (FEATURES if HALF else 0)
        last = nn.Conv2d(HIDDEN or inputs, HEAD, 1)
        self.head = nn.Sequential(nn.Conv2d(inputs, HIDDEN, 1), nn.ReLU(), last) if HIDDEN else last
        # Start close to "no history, uniform small blur" (with `length`: the running
        # mean of the history).
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if EDGE:
            # Per level: scales of the depth, normal and albedo edge stops (softplus).
            self.edge = nn.Parameter(torch.tensor([[3.0, 1.0, -1.0]] * len(DILATIONS)))

    def forward(self, x):
        if not HALF:
            h = F.relu(self.c1(x))
            h = F.relu(self.c2(h))
            return self.head(h)
        hgt, wid = x.shape[-2:]
        # Odd sizes: replicate the last row/column so every 2x2 block is complete.
        padded = F.pad(x, (0, wid % 2, 0, hgt % 2), mode="replicate")
        h = F.relu(self.c1(F.avg_pool2d(padded, 2)))
        h = F.relu(self.c2(h))
        up = F.interpolate(h, scale_factor=2, mode="nearest")[..., :hgt, :wid]
        return self.head(torch.cat((up, x), 1))


def warp(image, motion):
    """Bilinear sample `image` at pixel centre + motion (pixels); zero outside."""
    b, _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=image.dtype, device=image.device),
        torch.arange(w, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    px = xs + 0.5 + motion[:, 0]
    py = ys + 0.5 + motion[:, 1]
    grid = torch.stack((px / w * 2 - 1, py / h * 2 - 1), dim=-1)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    inside = ((px >= 0.5) & (px <= w - 0.5) & (py >= 0.5) & (py <= h - 0.5)).unsqueeze(1)
    return sampled, inside


def shift(x, dy, dx):
    """x sampled at (y+dy, x+dx) with edge replication."""
    p = max(abs(dy), abs(dx))
    padded = F.pad(x, (p, p, p, p), mode="replicate")
    h, w = x.shape[-2:]
    return padded[..., p + dy : p + dy + h, p + dx : p + dx + w]


def step(net, noisy, guides, history, prev_guides, first, exposure=None):
    """One frame. history/prev_guides: last frame's history (3 channels, 4 with its
    length) and (dist, normal, kind). Returns (history, guides to keep, radiance,
    validity of the reprojection)."""
    albedo = guides[:, ALBEDO]
    dist = guides[:, DIST : DIST + 1]
    normal = guides[:, NORMAL]
    rough = guides[:, ROUGH : ROUGH + 1]
    motion = guides[:, MOTION]
    kind = guides[:, KIND : KIND + 1]
    demod = noisy / (albedo + EPS)
    length = torch.zeros_like(dist)
    if first:
        warped = torch.zeros_like(demod)
        valid = torch.zeros_like(dist)
    else:
        stacked = torch.cat((history, prev_guides), 1)
        sampled, inside = warp(stacked, motion)
        warped = sampled[:, 0:3]
        g = history.shape[1]
        pdist, pnormal, pkind = sampled[:, g : g + 1], sampled[:, g + 1 : g + 4], sampled[:, g + 4 : g + 5]
        agree = (
            inside
            & ((pdist - dist).abs() <= 0.1 * dist + 1e-3)
            & ((pnormal * normal).sum(1, keepdim=True) >= 0.8)
            & ((pkind - kind).abs() < 0.5)
        )
        valid = agree.to(demod.dtype)
        if LENGTH:
            length = sampled[:, 3:4] * valid
    scale = 1.0 if exposure is None else exposure
    features = [
        torch.log1p(scale * demod.clamp(min=0)),
        torch.log1p(scale * warped.clamp(min=0)) * valid,
        valid,
        albedo.clamp(0, 4),
        normal,
        rough,
        torch.log1p(dist.clamp(min=0)),
    ]
    if LENGTH:
        features.append(length / MAX_LENGTH)
    out = net(torch.cat(features, 1))
    if LENGTH:
        # The running mean by default: the history's weight is N / (N + 1).
        mean = (length / (length + 1)).clamp(1e-3, 1 - 1e-3)
        alpha = torch.sigmoid(torch.log(mean / (1 - mean)) + out[:, 0:1]) * valid
    else:
        alpha = torch.sigmoid(out[:, 0:1]) * valid
    x = alpha * warped + (1 - alpha) * demod
    blend = x
    for level, d in enumerate(DILATIONS):
        logits = out[:, 1 + 9 * level : 10 + 9 * level]
        if EDGE:
            stops = F.softplus(net.edge[level])
            penalties = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    depth = (shift(dist, dy * d, dx * d) - dist).abs() / (dist + 1e-3)
                    facing = 1 - (shift(normal, dy * d, dx * d) * normal).sum(1, keepdim=True).clamp(0, 1)
                    colour = (shift(albedo, dy * d, dx * d) - albedo).abs().mean(1, keepdim=True)
                    penalties.append(stops[0] * depth + stops[1] * facing + stops[2] * colour)
            logits = logits - torch.cat(penalties, 1)
        weights = torch.softmax(logits, dim=1)
        acc = torch.zeros_like(x)
        k = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc = acc + weights[:, k : k + 1] * shift(x, dy * d, dx * d)
                k += 1
        x = acc
    radiance = x * (albedo + EPS)
    kept = blend if FEEDBACK == "blend" else x
    if LENGTH:
        kept = torch.cat((kept, (length + 1).clamp(max=MAX_LENGTH)), 1)
    return kept, torch.cat((dist, normal, kind), 1), radiance, valid


def tonemap(x):
    return torch.log1p(x.clamp(min=0))
