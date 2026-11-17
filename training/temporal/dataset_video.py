"""Training-data preview: per scene, sequences as video with five panels:
1 spp input | reference | albedo | normals | motion vectors.

    training/.venv/bin/python training/temporal/dataset_video.py DATA OUT_DIR [--fps 4] [--sequences 3]
"""
import argparse
import json
import pathlib
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("out")
p.add_argument("--fps", type=int, default=4)
p.add_argument("--sequences", type=int, default=3)
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def curve(v):
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return (v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F) - E / F


def tonemap(x, exposure):
    v = np.clip(curve(exposure * np.clip(x, 0, None)) / curve(11.2), 0, 1)
    return (np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)


def srgb(x):
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


labels = ["1 spp input", "2048 spp reference", "albedo", "normals", "motion vectors"]
for scene in sorted(d for d in pathlib.Path(a.data).iterdir() if (d / "meta.json").exists()):
    meta = json.loads((scene / "meta.json").read_text())
    h, w = meta["height"], meta["width"]
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for seq in sorted(scene.glob("seq_*"))[: a.sequences]:
            frames = [np.fromfile(f, "<f2").reshape(h, w, 18).astype(np.float32) for f in sorted(seq.glob("frame_*.f16"))]
            frames = [np.nan_to_num(f) for f in frames]
            # One exposure per sequence: reference log-average luminance to middle grey.
            lum = np.concatenate([f[..., 15:18] @ [0.2126, 0.7152, 0.0722] for f in frames])
            exposure = 0.18 / np.exp(np.mean(np.log(1e-4 + lum))) * 2
            for t, f in enumerate(frames):
                motion = f[..., 11:13]
                mv = np.stack([0.5 + motion[..., 0] / 40, 0.5 + motion[..., 1] / 40, np.full((h, w), 0.5)], -1)
                panels = [
                    tonemap(f[..., 0:3], exposure),
                    tonemap(f[..., 15:18], exposure),
                    srgb(np.where(f[..., 14:15] < 0.5, 0.3, f[..., 3:6])),  # sky: albedo holds radiance
                    srgb(f[..., 7:10] * 0.5 + 0.5),
                    (np.clip(mv, 0, 1) * 255).astype(np.uint8),
                ]
                img = Image.fromarray(np.concatenate(panels, 1))
                d = ImageDraw.Draw(img)
                for i, label in enumerate(labels):
                    d.text((i * w + 6, 6), label, fill=(255, 255, 0))
                d.text((6, h - 16), f"{scene.name} {seq.name} frame {t}", fill=(255, 255, 0))
                img.save(f"{tmp}/{n:05d}.png")
                n += 1
        target = out / f"{scene.name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(target)], check=True)
        print(f"wrote {target} ({n} frames)")
