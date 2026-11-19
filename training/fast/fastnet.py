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
SHARPEN = False  # one more output: sharpen or soften this frame's colour
GATHER = False  # two more outputs: towards a narrower or a wider gather of the samples
CUBIC = False  # the history is resampled with Catmull-Rom instead of bilinear
# exp(-k d^2) of the gather: the fixed pass, the narrow and the wide alternative.
WIDTHS = (2.0, 8.0, 0.5)
MAX_HISTORY = 64.0
LUMA = (0.2126, 0.7152, 0.0722)


def configure(hidden=8, kernel=1, sharpen=False, gather=False, cubic=False):
    global HIDDEN, KERNEL, SHARPEN, GATHER, CUBIC, FEATURES
    HIDDEN, KERNEL, SHARPEN, GATHER, CUBIC = hidden, kernel, sharpen, gather, cubic
    FEATURES = 9 + 2 * int(gather)


def settings():
    """The configuration, as saved in config.json."""
    return dict(head_features=FEATURES, head_hidden=HIDDEN, head_kernel=KERNEL, head_sharpen=SHARPEN, head_gather=GATHER, cubic_history=CUBIC)


def configure_from(c):
    configure(c.get("head_hidden", 8), c.get("head_kernel", 1), c.get("head_sharpen", False), c.get("head_gather", False), c.get("cubic_history", False))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(FEATURES, HIDDEN, KERNEL, padding=KERNEL // 2, padding_mode="replicate")
        self.b = nn.Conv2d(HIDDEN, 3 + int(SHARPEN) + 2 * int(GATHER), 1)
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


def warp_cubic(image, motion):
    """Catmull-Rom sample of `image` at pixel centre + motion, as 9 bilinear taps (the
    form a shader uses); edge replication, not below zero."""
    b, _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=image.dtype, device=image.device), torch.arange(w, dtype=image.dtype, device=image.device), indexing="ij"
    )

    def taps(p):
        centre = torch.floor(p - 0.5) + 0.5
        f = p - centre
        w0 = f * (-0.5 + f * (1 - 0.5 * f))
        w1 = 1 + f * f * (-2.5 + 1.5 * f)
        w2 = f * (0.5 + f * (2 - 1.5 * f))
        w3 = f * f * (-0.5 + 0.5 * f)
        return (centre - 1, centre + w2 / (w1 + w2), centre + 2), (w0, w1 + w2, w3)

    (px, wx), (py, wy) = taps(xs + 0.5 + motion[:, 0]), taps(ys + 0.5 + motion[:, 1])
    total = torch.zeros_like(image)
    for i in range(3):
        for j in range(3):
            grid = torch.stack((px[i] / w * 2 - 1, py[j] / h * 2 - 1), dim=-1)
            total = total + (wx[i] * wy[j]).unsqueeze(1) * F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return total.clamp(min=0)


def gather(low, low_depth, low_kind, coords, origin, jitter, scale, depth, kind, widths=(2.0,)):
    """Stage A's samples near each output pixel, joint bilateral.

    low (B, C, h, w) with low_depth and low_kind (B, 1, h, w); coords (B, 2, H, W)
    absolute output-pixel centres; origin (B, 2) low crop origin in low pixels; jitter
    (B, 2); scale (B, 2) render / output per axis; depth, kind (B, 1, H, W).
    Returns the weighted means (one per width k of exp(-k d^2)), the first width's
    weight sum and the nearest sample's distance.
    """
    b, c, h, w = low.shape
    H, W = coords.shape[-2:]  # noqa: N806
    sx, sy = scale[:, 0].view(b, 1, 1), scale[:, 1].view(b, 1, 1)
    jx, jy = jitter[:, 0].view(b, 1, 1), jitter[:, 1].view(b, 1, 1)
    # Output pixel centre in low-pixel units; low pixel a's sample lies at a + jitter.
    qx, qy = coords[:, 0] * sx, coords[:, 1] * sy
    bx, by = torch.floor(qx - jx), torch.floor(qy - jy)
    stacked = torch.cat((low, low_depth, low_kind), 1).reshape(b, c + 2, h * w)
    totals = [torch.zeros(b, c, H, W, device=low.device) for _ in widths]
    weights = [torch.zeros(b, 1, H, W, device=low.device) for _ in widths]
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
            agree = torch.exp(-relative * relative) * same
            for k, width in enumerate(widths):
                wgt = torch.exp(-width * d2) * agree
                totals[k] = totals[k] + wgt * tap[:, :c]
                weights[k] = weights[k] + wgt
            nearest = torch.minimum(nearest, d2)
    return [t / w.clamp(min=1e-6) for t, w in zip(totals, weights)], weights[0], nearest.sqrt()


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
    values, weight, nearest = gather(
        x_low,
        low_guides[:, A.DIST : A.DIST + 1],
        low_guides[:, A.KIND : A.KIND + 1],
        coords,
        origin,
        jitter,
        scale,
        depth,
        kind,
        WIDTHS if GATHER else WIDTHS[:1],
    )
    current, *others = [v * (albedo + A.EPS) + (0 if emission is None else emission) for v in values]
    if state is None:
        history, valid, length = current, torch.zeros_like(depth), torch.zeros_like(depth)
    else:
        sampled, inside = A.warp(torch.cat(state, 1), motion)
        history = warp_cubic(state[0], motion) if CUBIC else sampled[:, 0:3]
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
            *[log(luminance(o)) - lc for o in others],
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
        extra = out[:, 3:]
        if GATHER:
            # Towards the narrow and the wide gather, -0.25 to 1 each; 0 with a zero head.
            mix = -0.25 + 1.25 * torch.sigmoid(extra[:, -2:] + math.log(0.2 / 0.8))
            current = (current + mix[:, 0:1] * (others[0] - current) + mix[:, 1:2] * (others[1] - current)).clamp(min=0)
        if SHARPEN:
            # Away from (positive) or towards (negative) the 3x3 mean; 0 with a zero head.
            amount = -1 + 2.5 * torch.sigmoid(extra[:, 0:1] + math.log(0.4 / 0.6))
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
