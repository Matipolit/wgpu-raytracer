"""Side-by-side preview of denoise + upscale pipelines on a captured sequence.

    python training/fast/preview_video.py SCENE_DIR OUT.mp4 --old-td RUN/best.pt
        --new-td RUN/best.pt --old-joint RUN/best.pt --fast RUN/best.pt
        [--scale-index 0] [--fps 30]

SCENE_DIR: a capture with --upscale-capture (references may be absent). A 2 x 2 grid:
  old temporal denoiser + TAAU | retrained temporal denoiser + TAAU
  old joint network (v3)       | new joint network (training/fast)
The denoisers run at render resolution and TAAU (training/upscale/model.py taau_step,
the renderer's upscaler) brings them to output resolution. One exposure for all panels
(smoothed log-average of the new joint network's image) and the renderer's filmic curve.
"""
import argparse
import importlib.util
import resource
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# One descriptor per memory-mapped frame: lift the soft limit to the hard one.
resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "upscale"))
import data as D  # noqa: E402
import fastnet as N  # noqa: E402

A = N.A
spec = importlib.util.spec_from_file_location("joint_v3", here.parent / "upscale" / "model.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)

p = argparse.ArgumentParser()
p.add_argument("scene")
p.add_argument("out")
p.add_argument("--old-td", required=True)
p.add_argument("--new-td", required=True)
p.add_argument("--old-joint", required=True)
p.add_argument("--fast", required=True)
p.add_argument("--scale-index", type=int, default=0)
p.add_argument("--fps", type=int, default=30)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()


def config(weights):
    return json.loads(pathlib.Path(weights).with_name("config.json").read_text())


def temporal(weights):
    c = config(weights)
    A.configure(c["channels"], c["dilations"], c.get("half", False))
    net = A.Net().to(a.device)
    net.load_state_dict(torch.load(weights, map_location=a.device))
    return net.eval(), (c["channels"], tuple(c["dilations"]), c.get("half", False))


old_td, old_config = temporal(a.old_td)
new_td, new_config = temporal(a.new_td)
fast_config = config(a.fast)
A.configure(fast_config["channels"], fast_config["dilations"], True)
fast = N.FastNet().to(a.device)
fast.load_state_dict(torch.load(a.fast, map_location=a.device))
fast.eval()
J.configure_from(config(a.old_joint))
joint = J.Net().to(a.device)
joint.load_state_dict(torch.load(a.old_joint, map_location=a.device))
joint.eval()

scene = pathlib.Path(a.scene)
seq = D.load_root(scene.parent)[scene.name][0]
coords, origin, scale = D.geometry(seq, a.scale_index, a.device)


def curve(v):
    return (v * (0.15 * v + 0.05) + 0.004) / (v * (0.15 * v + 0.5) + 0.06) - 0.02 / 0.3


def display(x, exposure):
    v = np.clip(curve(exposure * np.clip(x, 0, None)) / curve(11.2), 0, 1)
    return (np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


font = ImageFont.load_default(size=22)
labels = ["old denoiser + TAAU", "retrained denoiser + TAAU", "old joint network (v3)", "new joint network"]
# Per pipeline: denoiser history, TAAU history.
state = {k: dict(h=None, g=None, th=None, tg=None) for k in ("old", "new")}
jh = jg = fa = fg = fb = None
log_lum = None
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
    count = len(seq.frames)
    for t in range(count):
        hi = torch.nan_to_num(torch.from_numpy(np.asarray(seq.frames[t])).to(a.device).float().permute(2, 0, 1).unsqueeze(0))
        lo = torch.nan_to_num(torch.from_numpy(np.asarray(seq.lows[a.scale_index]["frames"][t])).to(a.device).float().permute(2, 0, 1).unsqueeze(0))
        guides = hi[:, 3:15]
        jitter = torch.tensor([seq.lows[a.scale_index]["jitter"][t]], device=a.device)
        images = []
        for key, net, c in (("old", old_td, old_config), ("new", new_td, new_config)):
            s = state[key]
            A.configure(*c)
            s["h"], s["g"], radiance, _ = A.step(net, lo[:, 0:3], lo[:, 3:15], s["h"], s["g"], t == 0)
            s["th"], s["tg"], image = J.taau_step(radiance, guides, coords, origin, jitter, scale, s["th"], s["tg"], t == 0)
            images.append(image)
        jh, jg, image, _ = J.step(joint, lo, guides, coords, origin, jitter, scale, jh, jg, t == 0)
        images.append(image)
        A.configure(fast_config["channels"], fast_config["dilations"], True)
        fa, fg, radiance, _ = A.step(fast.a, lo[:, 0:3], lo[:, 3:15], fa, fg, t == 0)
        if t == 0:
            seen = 0.18 / N.luminance(lo[:, 0:3]).mean().clamp(min=1e-6).view(1, 1, 1, 1)
        image, fb, _ = N.step_b(fast.head, radiance / (lo[:, 3:6] + A.EPS), lo[:, 3:15], guides, coords, origin, jitter, scale, fb, seen)
        images.append(image)
        frames = [x[0].permute(1, 2, 0).cpu().numpy() for x in images]
        lum = np.log(1e-4 + frames[3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean()
        log_lum = lum if log_lum is None else log_lum + (lum - log_lum) * 0.05
        exposure = 0.18 / np.exp(log_lum) * 2.0
        panels = []
        for label, frame in zip(labels, frames):
            panel = Image.fromarray(display(frame, exposure))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 14 + draw.textlength(label, font=font), 34), fill=(0, 0, 0))
            draw.text((7, 4), label, fill=(255, 255, 255), font=font)
            panels.append(panel)
        w, h = panels[0].size
        grid = Image.new("RGB", (2 * w, 2 * h))
        for i, panel in enumerate(panels):
            grid.paste(panel, ((i % 2) * w, (i // 2) * h))
        grid.save(f"{tmp}/{t:05d}.png")
        if t % 50 == 0:
            print(f"frame {t}/{count}", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
                   check=True)
print(f"wrote {a.out}")
