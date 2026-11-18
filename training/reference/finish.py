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
import torch.nn.functional as F  # noqa: F401
from PIL import Image

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finishing import finish  # noqa: E402

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
    guides = None
    if (d / "guides.f32").exists() and not a.no_demodulate:
        guides = torch.from_numpy(np.fromfile(d / "guides.f32", dtype=np.float32).reshape(h, w, 6)[window].copy())
    final, robust, more = finish(groups, guides, k, a.bound, a.radius, a.strength, a.normal, not a.no_denoise)
    report.update(more)
    if not a.no_denoise:
        save(robust, f"robust{a.tag}.png")
save(final, f"final{a.tag}.png")
s = np.log2(np.maximum(spp, 1))
s = (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
Image.fromarray((s * 255).astype(np.uint8)).save(d / f"spp{a.tag}.png")
final.numpy().astype(np.float32).tofile(d / f"final{a.tag}.f32")
report["window"] = [int(xs.min()), int(ys.min()), int(final.shape[1]), int(final.shape[0])]
(d / f"finish{a.tag}.json").write_text(json.dumps(report))
print(json.dumps(report))
