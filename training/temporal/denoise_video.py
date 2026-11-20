"""Denoise a captured sequence with a trained model (PyTorch) and encode it as video.

    python training/temporal/denoise_video.py SEQ_DIR WEIGHTS.pt OUT.mp4 [--fps 60] [--panel denoised|split]

SEQ_DIR comes from `--capture ... --no-reference` (any frame count). Display:
smoothed log-average auto-exposure (like the renderer) and the renderer's filmic curve.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import model as M  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("seq")
p.add_argument("weights")
p.add_argument("out")
p.add_argument("--fps", type=int, default=60)
p.add_argument("--panel", default="denoised", choices=["denoised", "split"])
p.add_argument("--exposure-compensation", type=float, default=2.0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

config = pathlib.Path(a.weights).with_name("config.json")
if config.exists():
    c = json.loads(config.read_text())
    M.configure(**c)
else:
    c = {}
net = M.Net().to(a.device)
net.load_state_dict(torch.load(a.weights, map_location=a.device))
net.eval()

seq = pathlib.Path(a.seq)
meta = json.loads((seq.parent / "meta.json").read_text())
h, w = meta["height"], meta["width"]


def curve(v):
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return (v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F) - E / F


def display(x, exposure):
    v = np.clip(curve(exposure * np.clip(x, 0, None)) / curve(11.2), 0, 1)
    return (np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
files = sorted(f for f in seq.glob("frame_*.f16") if f.name.count(".") == 1)
history = prev = None
log_lum = None
# Exposure-normalized models: 0.18 / the mean luminance of the first frames' input.
k = None
if c.get("exposure"):
    first = np.stack([np.nan_to_num(np.fromfile(f, "<f2").reshape(h, w, -1)[..., 0:3].astype(np.float32)) for f in files[:8]])
    k = 0.18 / max(float((first @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean()), 1e-6)
with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
    for t, f in enumerate(files):
        x = np.nan_to_num(np.fromfile(f, "<f2").reshape(h, w, -1).astype(np.float32))
        frame = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(a.device)
        y, g, rad, _ = M.step(net, frame[:, 0:3], frame[:, 3:15], history, prev, t == 0, k)
        history, prev = y, g
        out = rad[0].permute(1, 2, 0).cpu().numpy()
        lum = np.log(1e-4 + out @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean()
        log_lum = lum if log_lum is None else log_lum + (lum - log_lum) * 0.05
        exposure = 0.18 / np.exp(log_lum) * a.exposure_compensation
        img = display(out, exposure)
        if a.panel == "split":
            img = np.concatenate([display(x[..., 0:3], exposure), img], 1)
        Image.fromarray(img).save(f"{tmp}/{t:05d}.png")
        if t % 50 == 0:
            print(f"frame {t}/{len(files)}", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", a.out], check=True)
print(f"wrote {a.out} ({len(files)} frames at {a.fps} fps)")
