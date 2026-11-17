"""Run the denoise + upscale network on a captured sequence and encode a video.

    python training/upscale/upscale_video.py SCENE_DIR SEQ WEIGHTS.pt OUT.mp4 [--scale-index 0] [--fps 60] [--panel split]

SCENE_DIR is a capture with --upscale-capture (references may be absent: --no-reference).
`split` shows the raw 1-spp render-resolution samples (gathered) next to the output.
Display: smoothed log-average auto-exposure and the renderer's filmic curve.
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
import data as D  # noqa: E402
import model as M  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("scene")
p.add_argument("seq", type=int)
p.add_argument("weights")
p.add_argument("out")
p.add_argument("--scale-index", type=int, default=0)
p.add_argument("--fps", type=int, default=60)
p.add_argument("--panel", default="network", choices=["network", "split"])
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
config = pathlib.Path(a.weights).with_name("config.json")
if config.exists():
    c = json.loads(config.read_text())
    M.configure(c["channels"], c["dilations"])
net = M.Net().to(a.device)
net.load_state_dict(torch.load(a.weights, map_location=a.device))
net.eval()
scene = pathlib.Path(a.scene)
seq = D.load(scene.parent)[scene.name][a.seq]


def curve(v):
    A, B, C, Dd, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return (v * (A * v + C * B) + Dd * E) / (v * (A * v + B) + Dd * F) - E / F


def display(x, exposure):
    v = np.clip(curve(exposure * np.clip(x, 0, None)) / curve(11.2), 0, 1)
    return (np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


log_lum = None
history = prev = None
coords, origin, scale = D.geometry(seq, a.scale_index, a.device)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
    count = len(seq.frames)
    for t in range(count):
        hi = torch.from_numpy(np.asarray(seq.frames[t])).to(a.device).float().permute(2, 0, 1).unsqueeze(0)
        lo = torch.from_numpy(np.asarray(seq.lows[a.scale_index]["frames"][t])).to(a.device).float().permute(2, 0, 1).unsqueeze(0)
        hi, lo = torch.nan_to_num(hi), torch.nan_to_num(lo)
        jitter = torch.tensor([seq.lows[a.scale_index]["jitter"][t]], device=a.device)
        history, prev, radiance, _ = M.step(net, lo, hi[:, 3:15], coords, origin, jitter, scale, history, prev, t == 0)
        out = radiance[0].permute(1, 2, 0).cpu().numpy()
        lum = np.log(1e-4 + out @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean()
        log_lum = lum if log_lum is None else log_lum + (lum - log_lum) * 0.05
        exposure = 0.18 / np.exp(log_lum) * 2.0
        img = display(out, exposure)
        if a.panel == "split":
            raw = M.splat(lo, coords, origin, jitter, scale)[0].permute(1, 2, 0).cpu().numpy()
            img = np.concatenate([display(raw, exposure), img], 1)
        Image.fromarray(img).save(f"{tmp}/{t:05d}.png")
        if t % 50 == 0:
            print(f"frame {t}/{count}", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
                   check=True)
print(f"wrote {a.out}")
