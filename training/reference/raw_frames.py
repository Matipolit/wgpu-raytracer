"""Noise of raw real-time frames (`--render-output x.png --raw-frames FILE`, offline.rs):

    python training/reference/raw_frames.py WIDTHxHEIGHT BASE.f32 VARIANT.f32... [--montage OUT.png --crop X,Y,W,H]

Each file holds near-independent 1-spp frames of a still camera. Per file: the mean
image's brightness against the first file's (a shift means bias), the noise of one
frame around that file's mean image (relative, per pixel and per 8x8 block; RMS in
exposure-normalized log space), the share of very bright outlier samples and of
samples with no light at all (among pixels that receive some). `--reference DIR`
(a `--reference-output` render of the same view): brightness against it instead. The
montage shows the first frame of every file.
"""
import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("size")
p.add_argument("files", nargs="+")
p.add_argument("--montage")
p.add_argument("--crop", default="")
p.add_argument("--zoom", type=int, default=2)
p.add_argument("--reference", help="a --reference-output directory of the same view and size: brightness and block error against it")
a = p.parse_args()
w, h = map(int, a.size.split("x"))
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
k = None
panels = []
truth = None
if a.reference:
    import json

    d = pathlib.Path(a.reference)
    groups = json.loads((d / "meta.json").read_text())["groups"]
    truth = np.fromfile(d / "groups.f32", dtype=np.float32).reshape(groups, h, w, 3).mean(0) @ LUMA
    k, base = 0.18 / truth.mean(), truth.mean()


def blocks8(image):
    return image[..., : h // 8 * 8, : w // 8 * 8].reshape(*image.shape[:-2], h // 8, 8, w // 8, 8).mean((-3, -1))

for name in a.files:
    frames = np.fromfile(name, dtype=np.float32).reshape(-1, h, w, 3)
    lum = frames @ LUMA
    mean = lum.mean(0)
    if k is None:
        k, base = 0.18 / mean.mean(), mean.mean()
    ok = mean > 1e-6
    # Noise relative to the reference image when there is one (comparable between
    # settings that change the brightness, e.g. direct light only), else to the mean.
    level = truth if truth is not None else mean
    visible = level > 0.02 * level.mean()
    rel = lum.std(0, ddof=1)[visible] / level[visible]
    blocks = blocks8(lum)
    block_level = blocks8(level)
    block = (blocks.std(0, ddof=1) / np.maximum(block_level, 1e-9))[block_level > 0.02 * level.mean()]
    log = np.log1p(k * np.maximum(frames, 0))
    noise = np.sqrt(((log - np.log1p(k * np.maximum(frames.mean(0), 0))) ** 2).mean())
    bright = (lum > 8 * np.maximum(mean, 1e-6)).mean()
    black = (lum[:, ok] == 0).mean()
    against = ""
    if truth is not None:
        # The mean image's 8x8 blocks against the reference's (bias plus the mean's
        # leftover noise).
        tb, mb = blocks8(truth), blocks8(mean)
        lit = tb > 0.02 * tb.mean()
        off = np.abs(mb[lit] - tb[lit]) / tb[lit]
        against = f"  block error vs reference median {np.median(off):.3f} p90 {np.quantile(off, 0.9):.3f}"
    brightness = mean.mean() / base
    if truth is not None:
        # Without the brightest 1% of the reference's blocks (emitters, sun glints).
        tb, mb = blocks8(truth), blocks8(mean)
        usual = tb < np.quantile(tb, 0.99)
        brightness = mb[usual].sum() / tb[usual].sum()
    print(f"{pathlib.Path(name).stem:16s} {len(frames)} frames  brightness {brightness:.3f}  noise/pixel median {np.median(rel):.2f} p90 {np.quantile(rel, 0.9):.2f}"
          f"  noise/8x8 block median {np.median(block):.3f} p90 {np.quantile(block, 0.9):.3f}  log RMS {noise:.4f}  outliers {100 * bright:.3f}%  no light {100 * black:.1f}%{against}")
    x = np.maximum(frames[0], 0) * k
    panels.append((pathlib.Path(name).stem, Image.fromarray(((x / (1 + x)) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8))))
if a.montage:
    x0, y0, cw, ch = map(int, a.crop.split(",")) if a.crop else (0, 0, w, h)
    z = a.zoom
    sheet = Image.new("RGB", ((cw * z + 4) * len(panels) - 4, ch * z + 22), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(panels):
        sheet.paste(image.crop((x0, y0, x0 + cw, y0 + ch)).resize((cw * z, ch * z), Image.NEAREST), (i * (cw * z + 4), 22))
        draw.text((i * (cw * z + 4) + 4, 5), label, fill=(230, 230, 230))
    sheet.save(a.montage)
