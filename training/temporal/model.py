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


def configure(channels=16, dilations=(1, 2, 4), half=False):
    global CHANNELS, DILATIONS, HEAD, HALF
    assert channels % 4 == 0
    CHANNELS, DILATIONS, HALF = channels, tuple(dilations), half
    HEAD = 1 + 9 * len(DILATIONS)

# Guide channel layout in captured frames (after the 3 noisy channels).
ALBEDO, DIST, NORMAL, ROUGH, MOTION, METAL, KIND = slice(0, 3), 3, slice(4, 7), 7, slice(8, 10), 10, 11


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(FEATURES, CHANNELS, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1, padding_mode="replicate")
        self.head = nn.Conv2d(CHANNELS + (FEATURES if HALF else 0), HEAD, 1)
        # Start close to "no history, uniform small blur".
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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


def step(net, noisy, guides, history, prev_guides, first):
    """One frame. history/prev_guides: last frame's output (3ch) and (dist, normal, kind)."""
    albedo = guides[:, ALBEDO]
    dist = guides[:, DIST : DIST + 1]
    normal = guides[:, NORMAL]
    rough = guides[:, ROUGH : ROUGH + 1]
    motion = guides[:, MOTION]
    kind = guides[:, KIND : KIND + 1]
    demod = noisy / (albedo + EPS)
    if first:
        warped = torch.zeros_like(demod)
        valid = torch.zeros_like(dist)
    else:
        stacked = torch.cat((history, prev_guides), 1)
        sampled, inside = warp(stacked, motion)
        warped = sampled[:, 0:3]
        pdist, pnormal, pkind = sampled[:, 3:4], sampled[:, 4:7], sampled[:, 7:8]
        agree = (
            inside
            & ((pdist - dist).abs() <= 0.1 * dist + 1e-3)
            & ((pnormal * normal).sum(1, keepdim=True) >= 0.8)
            & ((pkind - kind).abs() < 0.5)
        )
        valid = agree.to(demod.dtype)
    features = torch.cat(
        (
            torch.log1p(demod.clamp(min=0)),
            torch.log1p(warped.clamp(min=0)) * valid,
            valid,
            albedo.clamp(0, 4),
            normal,
            rough,
            torch.log1p(dist.clamp(min=0)),
        ),
        1,
    )
    out = net(features)
    alpha = torch.sigmoid(out[:, 0:1]) * valid
    x = alpha * warped + (1 - alpha) * demod
    for level, d in enumerate(DILATIONS):
        weights = torch.softmax(out[:, 1 + 9 * level : 10 + 9 * level], dim=1)
        acc = torch.zeros_like(x)
        k = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc = acc + weights[:, k : k + 1] * shift(x, dy * d, dx * d)
                k += 1
        x = acc
    radiance = x * (albedo + EPS)
    return x, torch.cat((dist, normal, kind), 1), radiance, valid


def tonemap(x):
    return torch.log1p(x.clamp(min=0))
