"""Noise of raw real-time frames (`--render-output x.png --raw-frames FILE`, offline.rs):

    python training/reference/raw_frames.py WIDTHxHEIGHT BASE.f32 VARIANT.f32... [--montage OUT.png --crop X,Y,W,H]

Each file holds near-independent 1-spp frames of a still camera. Per file: the mean
image's brightness against the first file's (a shift means bias), the noise of one
frame around that file's mean image (relative, per pixel and per 8x8 block; RMS in
exposure-normalized log space) and the share of very bright outlier samples. The
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
a = p.parse_args()
w, h = map(int, a.size.split("x"))
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
k = None
panels = []
for name in a.files:
    frames = np.fromfile(name, dtype=np.float32).reshape(-1, h, w, 3)
    lum = frames @ LUMA
    mean = lum.mean(0)
    if k is None:
        k, base = 0.18 / mean.mean(), mean.mean()
    ok = mean > 1e-6
    rel = lum.std(0, ddof=1)[ok] / mean[ok]
    blocks = lum[:, : h // 8 * 8, : w // 8 * 8].reshape(len(lum), h // 8, 8, w // 8, 8).mean((2, 4))
    block = blocks.std(0, ddof=1) / np.maximum(blocks.mean(0), 1e-6)
    log = np.log1p(k * np.maximum(frames, 0))
    noise = np.sqrt(((log - np.log1p(k * np.maximum(frames.mean(0), 0))) ** 2).mean())
    bright = (lum > 8 * np.maximum(mean, 1e-6)).mean()
    print(f"{pathlib.Path(name).stem:16s} {len(frames)} frames  brightness {mean.mean() / base:.3f}  noise/pixel median {np.median(rel):.2f} p90 {np.quantile(rel, 0.9):.2f}"
          f"  noise/8x8 block median {np.median(block):.3f} p90 {np.quantile(block, 0.9):.3f}  log RMS {noise:.4f}  outliers {100 * bright:.3f}%")
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
