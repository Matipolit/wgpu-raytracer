"""Fast denoise + upscale network
(docs/superpowers/specs/2026-11-18-fast-denoise-upscale-network-design.md).

Stage A, render resolution: the temporal denoiser (training/temporal/model.py), which
returns demodulated radiance per render pixel.
Stage B, output resolution, one pass per frame (the learned form of
src/raytracer/detail.wgsl):
  1. gather stage A's output at the output pixel from this frame's jittered samples,
     weighted by distance and by agreement of depth and surface kind (joint bilateral);
  2. multiply by the output-resolution albedo: this frame's colour;
  3. reproject the output history (final colour, depth, history length N);
  4. clip the history to the 3x3 neighbourhood of this frame's colour (YCoCg mean +-
     gamma x deviation);
  5. blend with weight alpha, 1 / (N + 1) by default.
A per-pixel head (one hidden layer) turns a few statistics into corrections of gamma,
of alpha and of the history's confidence (N shrinks instead of resetting). With a zero
head the pass is the non-learned detail upscaler. Accumulating final colour averages
out jittered albedo and edges, which a per-frame remodulation cannot.
"""
import math
import pathlib
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "temporal"))
import model as A  # noqa: E402  (the temporal denoiser)

EPS = A.EPS
FEATURES = 9
# The head; `configure` changes these before building a FastNet (saved in config.json).
HIDDEN = 8
KERNEL = 1  # 3: the first layer sees the 3x3 neighbourhood
SHARPEN = False  # a fourth output: sharpen or soften this frame's colour
MAX_HISTORY = 64.0
LUMA = (0.2126, 0.7152, 0.0722)


def configure(hidden=8, kernel=1, sharpen=False):
    global HIDDEN, KERNEL, SHARPEN
    HIDDEN, KERNEL, SHARPEN = hidden, kernel, sharpen


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(FEATURES, HIDDEN, KERNEL, padding=KERNEL // 2, padding_mode="replicate")
        self.b = nn.Conv2d(HIDDEN, 3 + int(SHARPEN), 1)
        # Start as the non-learned pass.
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, f):
        return self.b(F.relu(self.a(f)))


class FastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = A.Net()
        self.head = Head()


def luminance(x):
    return LUMA[0] * x[:, 0:1] + LUMA[1] * x[:, 1:2] + LUMA[2] * x[:, 2:3]


def to_ycocg(c):
    r, g, b = c[:, 0:1], c[:, 1:2], c[:, 2:3]
    return torch.cat((0.25 * r + 0.5 * g + 0.25 * b, 0.5 * r - 0.5 * b, -0.25 * r + 0.5 * g - 0.25 * b), 1)


def from_ycocg(c):
    y, co, cg = c[:, 0:1], c[:, 1:2], c[:, 2:3]
    return torch.cat((y + co - cg, y + cg, y - co - cg), 1)


def gather(low, low_depth, low_kind, coords, origin, jitter, scale, depth, kind):
    """Stage A's samples near each output pixel, joint bilateral.

    low (B, C, h, w) with low_depth and low_kind (B, 1, h, w); coords (B, 2, H, W)
    absolute output-pixel centres; origin (B, 2) low crop origin in low pixels; jitter
    (B, 2); scale (B, 2) render / output per axis; depth, kind (B, 1, H, W).
    Returns the weighted mean, the weight sum and the nearest sample's distance.
    """
    b, c, h, w = low.shape
    H, W = coords.shape[-2:]  # noqa: N806
    sx, sy = scale[:, 0].view(b, 1, 1), scale[:, 1].view(b, 1, 1)
    jx, jy = jitter[:, 0].view(b, 1, 1), jitter[:, 1].view(b, 1, 1)
    # Output pixel centre in low-pixel units; low pixel a's sample lies at a + jitter.
    qx, qy = coords[:, 0] * sx, coords[:, 1] * sy
    bx, by = torch.floor(qx - jx), torch.floor(qy - jy)
    stacked = torch.cat((low, low_depth, low_kind), 1).reshape(b, c + 2, h * w)
    total = torch.zeros(b, c, H, W, device=low.device)
    weight = torch.zeros(b, 1, H, W, device=low.device)
    nearest = torch.full((b, 1, H, W), 1e9, device=low.device)
    for dy in (-1, 0, 1, 2):
        for dx in (-1, 0, 1, 2):
            ax, ay = bx + dx, by + dy
            d2 = ((ax + jx - qx) ** 2 + (ay + jy - qy) ** 2).unsqueeze(1)
            lx = (ax - origin[:, 0].view(b, 1, 1)).clamp(0, w - 1).long()
            ly = (ay - origin[:, 1].view(b, 1, 1)).clamp(0, h - 1).long()
            index = (ly * w + lx).view(b, 1, H * W).expand(b, c + 2, H * W)
            tap = torch.gather(stacked, 2, index).view(b, c + 2, H, W)
            relative = (tap[:, c : c + 1] - depth) / (0.05 * depth + 1e-3)
            same = torch.where((tap[:, c + 1 : c + 2] - kind).abs() < 0.5, 1.0, 1e-4)
            wgt = torch.exp(-2.0 * d2 - relative * relative) * same
            total = total + wgt * tap[:, :c]
            weight = weight + wgt
            nearest = torch.minimum(nearest, d2)
    return total / weight.clamp(min=1e-6), weight, nearest.sqrt()


def step_b(head, x_low, low_guides, guides, coords, origin, jitter, scale, state, exposure, emission=None):
    """One output frame. x_low: stage A's demodulated output (B, 3, h, w); low_guides
    (B, 12, h, w); guides (B, 12, H, W) at output resolution; state: None or (history
    rgb, depth, length); exposure (B, 1, 1, 1); emission: this frame's directly visible
    emission at output resolution, accumulated with the rest. Returns (image, state,
    valid)."""
    albedo = guides[:, A.ALBEDO]
    depth = guides[:, A.DIST : A.DIST + 1]
    kind = guides[:, A.KIND : A.KIND + 1]
    motion = guides[:, A.MOTION]
    value, weight, nearest = gather(
        x_low, low_guides[:, A.DIST : A.DIST + 1], low_guides[:, A.KIND : A.KIND + 1], coords, origin, jitter, scale, depth, kind
    )
    current = value * (albedo + A.EPS)
    if emission is not None:
        current = current + emission
    if state is None:
        history, valid, length = current, torch.zeros_like(depth), torch.zeros_like(depth)
    else:
        sampled, inside = A.warp(torch.cat(state, 1), motion)
        history = sampled[:, 0:3]
        valid = (inside & ((sampled[:, 3:4] - depth).abs() <= 0.1 * depth + 1e-3)).to(current.dtype)
        length = sampled[:, 4:5] * valid
        history = valid * history + (1 - valid) * current
    y = to_ycocg(current)
    pad = F.pad(y, (1, 1, 1, 1), mode="replicate")
    m1 = F.avg_pool2d(pad, 3, 1)
    sigma = (F.avg_pool2d(pad * pad, 3, 1) - m1 * m1).clamp(min=0).sqrt()
    log = lambda v: torch.log1p(exposure * v.clamp(min=0))
    lc, lh = log(luminance(current)), log(luminance(history))
    features = torch.cat(
        (
            lc,
            lh * valid,
            (lc - lh).abs() * valid,
            log(sigma[:, 0:1]),
            length / MAX_HISTORY,
            valid,
            weight.clamp(max=4) / 4,
            nearest,
            torch.log1p(motion.norm(dim=1, keepdim=True)),
        ),
        1,
    )
    out = head(features)
    if out.shape[1] > 3:
        # Away from (positive) or towards (negative) the 3x3 mean; 0 with a zero head.
        amount = -1 + 2.5 * torch.sigmoid(out[:, 3:4] + math.log(0.4 / 0.6))
        current = (current + amount * (current - from_ycocg(m1))).clamp(min=0)
    # Clip width in deviations, 0.5 to 4 (1.25 with a zero head).
    gamma = 0.5 + 3.5 * torch.sigmoid(out[:, 0:1] + math.log(0.75 / 2.75))
    clipped = from_ycocg(torch.maximum(torch.minimum(to_ycocg(history), m1 + gamma * sigma), m1 - gamma * sigma))
    # Confidence in the history shrinks its length (1 with a zero head: almost).
    length = length * torch.sigmoid(out[:, 2:3] + 6.0)
    base = (1 / (length + 1)).clamp(1e-3, 1 - 1e-3)
    alpha = torch.sigmoid(torch.log(base / (1 - base)) + out[:, 1:2])
    alpha = valid * alpha + (1 - valid)
    image = (alpha * current + (1 - alpha) * clipped).clamp(min=0)
    return image, (image, depth, (length + 1).clamp(max=MAX_HISTORY)), valid
