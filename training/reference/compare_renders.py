"""Two linear rgb f32 renders of the same view (ours: a `--reference-output` directory
or a raw file; the other: a raw file, e.g. from cycles_render.py): brightness ratios per
8x8 block and a picture (ours, theirs, ratio map: blue = ours darker, red = brighter,
full scale +-50%).

    python training/reference/compare_renders.py WxH OURS THEIRS OUT.png
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image

size, ours, theirs, out = sys.argv[1:5]
w, h = map(int, size.split("x"))
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load(path):
    path = pathlib.Path(path)
    if path.is_dir():
        groups = json.loads((path / "meta.json").read_text())["groups"]
        return np.fromfile(path / "groups.f32", dtype=np.float32).reshape(groups, h, w, 3).mean(0)
    return np.fromfile(path, dtype=np.float32).reshape(h, w, 3)


a, b = load(ours), load(theirs)
la, lb = a @ LUMA, b @ LUMA
blocks = lambda x: x[: h // 8 * 8, : w // 8 * 8].reshape(h // 8, 8, w // 8, 8).mean((1, 3))
ba, bb = blocks(la), blocks(lb)
lit = bb > 0.02 * np.median(bb)
ratio = ba[lit] / bb[lit]
q = lambda p: float(np.quantile(ratio, p))
print(f"ours / theirs: mean {la.mean() / lb.mean():.3f}; per 8x8 block median {q(0.5):.3f}, 10th-90th percentile {q(0.1):.3f}-{q(0.9):.3f}, 1st-99th {q(0.01):.3f}-{q(0.99):.3f}")
k = 0.18 / np.exp(np.log(np.maximum(lb, 1e-6)).mean())
tone = lambda x: Image.fromarray(((lambda y: (y / (1 + y)) ** (1 / 2.2))(np.maximum(x, 0) * k) * 255).astype(np.uint8))
r = np.clip(np.log2(np.maximum(ba, 1e-9) / np.maximum(bb, 1e-9)) / np.log2(1.5), -1, 1)
r[~lit] = 0
rgb = np.stack([1 + np.minimum(r, 0), 1 - np.abs(r), 1 - np.maximum(r, 0)], -1)
sheet = Image.new("RGB", (w * 3, h))
sheet.paste(tone(a), (0, 0))
sheet.paste(tone(b), (w, 0))
sheet.paste(Image.fromarray((rgb * 255).astype(np.uint8)).resize((w // 8 * 8, h // 8 * 8), Image.NEAREST), (2 * w, 0))
sheet.save(out)
