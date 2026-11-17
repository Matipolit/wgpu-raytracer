"""Joint denoise + upscale network (one model for any render scale).

Inputs per frame, all at output resolution unless noted:
  low      (B, 15, h, w)  render-resolution 1-spp radiance + its 12 guides (a crop)
  guides   (B, 12, H, W)  output-resolution guides: albedo 0:3, distance 3, normal 4:7,
                          roughness 7, motion 8:10 (output pixels), metallic 10, kind 11
  geometry where each low sample lies: absolute output-pixel coordinates of the crop,
           the low crop origin, this frame's jitter and the render/output scale.

Per output pixel the step gathers this frame's samples by their exact jittered positions
(4x4 low pixels around, Gaussian weights of half a low pixel), demodulated by the low
albedo; reprojects last frame's output with the output motion; and feeds the network the
gathered radiance, its confidence (weight sum, nearest-sample distance), the history and
its validity, the output guides and the scale. The network predicts a temporal blend
factor and three softmax filter kernels (dilations 1, 2, 4) applied in demodulated space;
the result is remodulated by the output albedo, so texture detail comes from the
full-resolution G-buffer. The structure mirrors training/temporal/model.py so the WGSL
export can follow the same pattern.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.01
ALBEDO, DIST, NORMAL, ROUGH, MOTION, METAL, KIND = slice(0, 3), 3, slice(4, 7), 7, slice(8, 10), 10, 11
# Low inputs: radiance 0:3, then the same 12 guides at render resolution.
LOW_RADIANCE, LOW_ALBEDO = slice(0, 3), slice(3, 6)
FEATURES = 18
CHANNELS = 16
DILATIONS = (1, 2, 4)
HEAD = 1 + 9 * len(DILATIONS)


def configure(channels, dilations):
    global CHANNELS, DILATIONS, HEAD
    CHANNELS, DILATIONS = channels, tuple(dilations)
    HEAD = 1 + 9 * len(DILATIONS)


class Net(nn.Module):
    """Two 3x3 convs at half output resolution, a 1x1 head at full resolution."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(FEATURES, CHANNELS, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1, padding_mode="replicate")
        self.head = nn.Conv2d(CHANNELS + FEATURES, HEAD, 1)

    def forward(self, f):
        h, w = f.shape[-2:]
        x = F.avg_pool2d(F.pad(f, (0, w % 2, 0, h % 2), mode="replicate"), 2)
        x = F.relu(self.c2(F.relu(self.c1(x))))
        x = F.interpolate(x, scale_factor=2, mode="nearest")[..., :h, :w]
        return self.head(torch.cat((x, f), 1))


def gather(low, coords, origin, jitter, scale):
    """This frame's samples near each output pixel.

    low (B, C, h, w); coords (B, 2, H, W) absolute output-pixel centres (x, y); origin (B, 2)
    low crop origin in low pixels; jitter (B, 2); scale (B, 2) render/output per axis.
    Returns the weighted mean (B, C, H, W), the weight sum and the nearest-sample
    distance (B, 1, H, W) in low pixels.
    """
    b, c, h, w = low.shape
    H, W = coords.shape[-2:]  # noqa: N806
    sx, sy = scale[:, 0].view(b, 1, 1), scale[:, 1].view(b, 1, 1)
    jx, jy = jitter[:, 0].view(b, 1, 1), jitter[:, 1].view(b, 1, 1)
    # Output pixel centre in low-pixel units; low pixel a's sample lies at a + jitter.
    qx, qy = coords[:, 0] * sx, coords[:, 1] * sy
    bx, by = torch.floor(qx - jx), torch.floor(qy - jy)
    flat = low.reshape(b, c, h * w)
    total = torch.zeros(b, c, H, W, device=low.device)
    weight = torch.zeros(b, 1, H, W, device=low.device)
    nearest = torch.full((b, 1, H, W), 1e9, device=low.device)
    for dy in (-1, 0, 1, 2):
        for dx in (-1, 0, 1, 2):
            ax, ay = bx + dx, by + dy
            d2 = (ax + jx - qx) ** 2 + (ay + jy - qy) ** 2
            lx = (ax - origin[:, 0].view(b, 1, 1)).clamp(0, w - 1).long()
            ly = (ay - origin[:, 1].view(b, 1, 1)).clamp(0, h - 1).long()
            index = (ly * w + lx).view(b, 1, H * W).expand(b, c, H * W)
            value = torch.gather(flat, 2, index).view(b, c, H, W)
            wgt = torch.exp(-2.0 * d2).unsqueeze(1)
            total = total + wgt * value
            weight = weight + wgt
            nearest = torch.minimum(nearest, d2.unsqueeze(1))
    return total / weight.clamp(min=1e-6), weight, nearest.sqrt()


def warp(x, motion):
    """Sample x at each pixel's previous position (motion in pixels); zeros outside."""
    b, _, h, w = x.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    px = xs + 0.5 + motion[:, 0]
    py = ys + 0.5 + motion[:, 1]
    grid = torch.stack((px / w * 2 - 1, py / h * 2 - 1), -1)
    inside = ((px >= 0.5) & (px <= w - 0.5) & (py >= 0.5) & (py <= h - 0.5)).unsqueeze(1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False), inside


def shift(x, dx, dy):
    """x shifted by (dx, dy) pixels with edge replication (tap of an a-trous kernel)."""
    h, w = x.shape[-2:]
    p = max(abs(dx), abs(dy))
    padded = F.pad(x, (p, p, p, p), mode="replicate")
    return padded[..., p + dy : p + dy + h, p + dx : p + dx + w]


def step(net, low, guides, coords, origin, jitter, scale, history, prev_guides, first):
    """One frame. history: last output (demodulated rgb); prev_guides: (dist, normal, kind)."""
    albedo = guides[:, ALBEDO]
    dist, normal, kind = guides[:, DIST : DIST + 1], guides[:, NORMAL], guides[:, KIND : KIND + 1]
    demod_low = low[:, LOW_RADIANCE] / (low[:, LOW_ALBEDO] + EPS)
    current, weight, nearest = gather(demod_low, coords, origin, jitter, scale)
    if first:
        warped = torch.zeros_like(current)
        valid = torch.zeros_like(dist)
    else:
        sampled, inside = warp(torch.cat((history, prev_guides), 1), guides[:, MOTION])
        warped = sampled[:, 0:3]
        agree = (
            inside
            & ((sampled[:, 3:4] - dist).abs() <= 0.1 * dist + 1e-3)
            & ((sampled[:, 4:7] * normal).sum(1, keepdim=True) >= 0.8)
            & ((sampled[:, 7:8] - kind).abs() < 0.5)
        )
        valid = agree.to(current.dtype)
    inverse_scale = (1.0 / scale.mean(1)).view(-1, 1, 1, 1).expand_as(dist)
    features = torch.cat(
        (
            torch.log1p(current.clamp(min=0)),
            weight.clamp(max=4.0),
            nearest.clamp(max=2.0),
            torch.log1p(warped.clamp(min=0)) * valid,
            valid,
            albedo.clamp(0, 4),
            normal,
            guides[:, ROUGH : ROUGH + 1],
            torch.log1p(dist.clamp(min=0)),
            inverse_scale,
        ),
        1,
    )
    out = net(features)
    alpha = torch.sigmoid(out[:, 0:1]) * valid
    x = alpha * warped + (1 - alpha) * current
    for level, d in enumerate(DILATIONS):
        weights = torch.softmax(out[:, 1 + 9 * level : 10 + 9 * level], dim=1)
        x = sum(weights[:, k : k + 1] * shift(x, (k % 3 - 1) * d, (k // 3 - 1) * d) for k in range(9))
    radiance = x * (albedo + EPS)
    return x, torch.cat((dist, normal, kind), 1), radiance, valid


def tonemap(x):
    return torch.log1p(x.clamp(min=0))


# --- non-learned baselines (evaluation) ---


def splat(low, coords, origin, jitter, scale):
    """Raw 1-spp radiance gathered to output resolution (no demodulation, no history)."""
    rad, _, _ = gather(low[:, LOW_RADIANCE], coords, origin, jitter, scale)
    return rad


def taau_step(low, guides, coords, origin, jitter, scale, history, prev_guides, first):
    """The in-renderer upscaler (src/raytracer/upscale.wgsl) on raw input: gathered
    radiance blended into reprojected history with weight mix(0.04, 0.15, confidence)."""
    current, _, nearest = gather(low[:, LOW_RADIANCE], coords, origin, jitter, scale)
    dist = guides[:, DIST : DIST + 1]
    if first:
        return current, dist, current
    sampled, inside = warp(torch.cat((history, prev_guides), 1), guides[:, MOTION])
    valid = (inside & ((sampled[:, 3:4] - dist).abs() <= 0.1 * dist + 1e-3)).to(current.dtype)
    confidence = torch.exp(-2.0 * nearest**2)
    alpha = 0.04 + (0.15 - 0.04) * confidence
    alpha = valid * alpha + (1 - valid)
    out = alpha * current + (1 - alpha) * sampled[:, 0:3]
    return out, dist, out


def psnr(mse):
    return 10 * math.log10(1 / max(mse, 1e-12))
