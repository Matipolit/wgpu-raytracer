"""Per-sample noise of `--reference-output` directories (src/reference.rs):

    python training/reference/noise.py DIR...

Per pixel, the relative standard deviation of one sample's luminance (from the
groups), and the samples that pixel needs for 1% relative error."""
import json, sys, pathlib
import numpy as np
for name in sys.argv[1:]:
    d = pathlib.Path(name); m = json.loads((d / "meta.json").read_text()); G = m["groups"]
    spp = np.fromfile(d / "spp.f32", dtype=np.float32).reshape(m["height"], m["width"])
    ys, xs = np.nonzero(spp); w = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    g = np.fromfile(d / "groups.f32", dtype=np.float32).reshape(G, m["height"], m["width"], 3)[(slice(None), *w)]
    lum = g @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    n = spp[w] / G
    mean = lum.mean(0); var1 = lum.var(0, ddof=1) * n          # variance of one sample
    ok = mean > 1e-6
    rel = np.sqrt(var1[ok]) / mean[ok]
    need = (rel / 0.01) ** 2
    # Image-level: total one-sample variance over total squared signal (energy weighted).
    print(f"{d.name:26s} mean {mean.mean():.4f}  rel.std/sample median {np.median(rel):5.2f} p90 {np.quantile(rel,0.9):6.2f}"
          f"  spp for 1%: median {np.median(need):7.0f} p90 {np.quantile(need,0.9):8.0f}  var/signal^2 {var1.sum()/ (mean**2).sum():7.2f}  {m['seconds']:.0f} s")
