"""Final reference image from a `--reference-output` directory (src/reference.rs).

    python training/reference/finish.py DIR [--exposure K] [--bound 2] [--no-denoise] [--no-demodulate]

1. Firefly rejection, only for extreme outliers: per pixel, a group whose luminance is
   more than twice the other groups' mean plus 12 of their standard deviations is
   dropped (at most two). Milder outliers are left to step 2, which spreads their
   energy over similar pixels instead of removing it.
2. Bounded denoise: the two halves of the groups are independent images A and B.
   A is filtered with non-local-means weights measured on B and B with weights from A
   (Rousselle et al. 2012: weights from an independent buffer do not correlate with
   the noise they filter), distances normalized by the measured per-pixel variance.
   The filter runs on lighting only: the image divided by the pixel-integrated albedo
   (DIR/guides.f32), multiplied back afterwards, with a weight on the normal.
   The result may differ from the pixel's mean by at most `--bound` standard errors:
   where the image is converged, nothing changes.
Writes DIR/final.f32 (rgb f32 of the traced window), mean.png, final.png, spp.png and
prints the measured errors (exposure-normalized log space, as the training loss).
"""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("--exposure", type=float, help="display and error exposure (default: 0.18 / mean luminance)")
p.add_argument("--bound", type=float, default=2.0, help="largest change by the denoise, in standard errors")
p.add_argument("--radius", type=int, default=5, help="search window radius of the denoise")
p.add_argument("--strength", type=float, default=0.45, help="filter strength: the k of Rousselle et al.")
p.add_argument("--normal", type=float, default=0.02, help="normal weight: exp(-|n - n'|^2 / this)")
p.add_argument("--no-demodulate", action="store_true", help="filter the image itself, not image / albedo")
p.add_argument("--no-denoise", action="store_true")
p.add_argument("--groups", help="A:B: use only groups A..B-1 (two disjoint halves give two independent finished images)")
p.add_argument("--tag", default="", help="suffix of the written file names")
a = p.parse_args()
d = pathlib.Path(a.dir)
meta = json.loads((d / "meta.json").read_text())
w, h, g = meta["width"], meta["height"], meta["groups"]
spp = np.fromfile(d / "spp.f32", dtype=np.float32).reshape(h, w)
ys, xs = np.nonzero(spp)
window = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
groups = torch.from_numpy(np.fromfile(d / "groups.f32", dtype=np.float32).reshape(g, h, w, 3)[(slice(None), *window)].copy())
if a.groups:
    first, last = map(int, a.groups.split(":"))
    groups = groups[first:last]
    g = last - first
spp = spp[window]
LUMA = torch.tensor([0.2126, 0.7152, 0.0722])
mean = groups.mean(0)
k = a.exposure or 0.18 / float((mean @ LUMA).mean())


def save(image, name):
    x = (image * k).clamp(min=0)
    x = (x / (1 + x)) ** (1 / 2.2)  # Reinhard, gamma 2.2
    Image.fromarray((x.numpy() * 255 + 0.5).astype(np.uint8)).save(d / name)


def log(image):
    return torch.log1p(image.clamp(min=0) * k)


save(mean, f"mean{a.tag}.png")
report = {"exposure": k, "mean_spp": float(spp.mean()), "seconds": meta["seconds"]}
final = mean
if g > 1:
    # 1. Firefly rejection on group luminance.
    luma = groups @ LUMA  # g, h, w
    order = luma.argsort(0)
    ranked = luma.gather(0, order)
    keep = torch.ones_like(luma, dtype=torch.bool)
    floor = 0.01 / k
    dropped = torch.zeros(luma.shape[1:], dtype=torch.bool)
    still = torch.ones_like(dropped)
    for top in (g - 1, g - 2):
        rest = ranked[:top]
        limit = 2 * rest.mean(0) + 12 * rest.std(0) + floor
        still = still & (ranked[top] > limit)
        keep.scatter_(0, order[top : top + 1], ~still[None])
        dropped |= still
    count = keep.sum(0)[..., None]
    robust = (groups * keep[..., None]).sum(0) / count
    report["firefly_pixels"] = float(dropped.float().mean())
    report["firefly_energy"] = float(1 - robust.sum() / mean.sum())
    # Variance of the mean, per channel, from the kept groups (3x3 smoothed: 8 groups
    # estimate it roughly).
    variance = ((groups - robust) ** 2 * keep[..., None]).sum(0) / (count - 1).clamp(min=1) / count
    variance = F.avg_pool2d(variance.permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
    error = variance.sqrt()
    report["mean_error"] = float(((k * error / (1 + k * robust)) ** 2).mean().sqrt())
    final = robust
    if not a.no_denoise:
        # 2. Dual-buffer non-local means.
        half = g // 2
        halves = []
        for part in (slice(0, half), slice(half, g)):
            n = keep[part].sum(0)[..., None].clamp(min=1)
            halves.append(((groups[part] * keep[part][..., None]).sum(0) / n).permute(2, 0, 1)[None])
        var = (2 * variance).permute(2, 0, 1)[None]
        # Lighting only: divided by the pixel-integrated albedo, the image has no
        # texture left for the filter to smooth; normals keep it within a surface.
        albedo = normal = None
        if (d / "guides.f32").exists() and not a.no_demodulate:
            guides = torch.from_numpy(np.fromfile(d / "guides.f32", dtype=np.float32).reshape(h, w, 6)[window].copy())
            albedo = (guides[..., :3] + 0.01).permute(2, 0, 1)[None]
            normal = guides[..., 3:].permute(2, 0, 1)[None]
            halves = [x / albedo for x in halves]
            var = var / albedo**2
        r = a.radius
        pad = lambda t: F.pad(t, (r, r, r, r), mode="replicate")
        hh, ww = mean.shape[:2]
        outputs = []
        for guide, source in ((halves[1], halves[0]), (halves[0], halves[1])):
            pg, ps, pv = pad(guide), pad(source), pad(var)
            pn = pad(normal) if normal is not None else None
            total = torch.zeros_like(source)
            weight = torch.zeros(1, 1, hh, ww)
            for dy in range(2 * r + 1):
                for dx in range(2 * r + 1):
                    q = (slice(None), slice(None), slice(dy, dy + hh), slice(dx, dx + ww))
                    distance = ((guide - pg[q]) ** 2 - (var + torch.minimum(var, pv[q]))) / (1e-10 + a.strength**2 * (var + pv[q]))
                    distance = F.avg_pool2d(distance.mean(1, keepdim=True), 3, 1, 1, count_include_pad=False)
                    weights = torch.exp(-distance.clamp(min=0))
                    if pn is not None:
                        weights = weights * torch.exp(-((normal - pn[q]) ** 2).sum(1, keepdim=True) / a.normal)
                    total += weights * ps[q]
                    weight += weights
            outputs.append(total / weight)
        filtered = 0.5 * (outputs[0] + outputs[1])
        if albedo is not None:
            filtered = filtered * albedo
        filtered = filtered[0].permute(1, 2, 0)
        final = torch.maximum(torch.minimum(filtered, robust + a.bound * error), robust - a.bound * error).clamp(min=0)
        report["changed_by_denoise"] = float((log(final) - log(robust)).abs().mean())
        save(robust, f"robust{a.tag}.png")
report["brightness"] = float(final.sum() / mean.sum())
save(final, f"final{a.tag}.png")
s = np.log2(np.maximum(spp, 1))
s = (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
Image.fromarray((s * 255).astype(np.uint8)).save(d / f"spp{a.tag}.png")
final.numpy().astype(np.float32).tofile(d / f"final{a.tag}.f32")
report["window"] = [int(xs.min()), int(ys.min()), int(final.shape[1]), int(final.shape[0])]
(d / f"finish{a.tag}.json").write_text(json.dumps(report))
print(json.dumps(report))
