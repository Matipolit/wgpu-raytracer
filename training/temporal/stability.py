"""Temporal stability of a denoiser on long sequences without references (the preview
video captures): how much the output changes from frame to frame where the history is
valid, after the first frames.

    python stability.py --data RUN_DIR --weights RUN/best.pt [--frames 150] [--skip 30]

Per scene, in log(1 + exposure * x): mean |y_t - warp(y_t-1)| on valid pixels for the
1-spp input and the network (lower is calmer; blur lowers it too), and the mean gradient
of the network's output (lower is smoother), so a calm result can be told from a blurred
one. "boil" is the slow, blotchy part of the change: the same difference of the lighting
alone (radiance / albedo), averaged with its sign over 8x8 blocks before the absolute
value, so pixel noise and resampled texture cancel and what moves together stays.
"slow" is the same block measure against the lighting's running mean along the motion
(1/16 per frame) instead of the previous frame: it also counts blotches that drift over
many frames, which change too little per frame to show in "boil". "typical" is the
median block of "slow" over the frame: a few restless surfaces (a mirror, a moving
object's trail) can carry most of the mean.
The exposure is 0.18 / the mean luminance of the first frames' input.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import model as M  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--weights", required=True)
p.add_argument("--frames", type=int, default=150)
p.add_argument("--skip", type=int, default=30)
p.add_argument("--maps", help="directory for per-scene images: the last frame beside where the slow change is (white = most)")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
config = pathlib.Path(a.weights).with_name("config.json")
c = json.loads(config.read_text()) if config.exists() else {}
M.configure(**c)
net = M.Net().to(a.device)
net.load_state_dict(torch.load(a.weights, map_location=a.device))
net.eval()


def luminance(x):
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


totals = np.zeros(6)
scenes = sorted(d for d in pathlib.Path(a.data).iterdir() if (d / "meta.json").exists())
with torch.no_grad():
    for scene in scenes:
        meta = json.loads((scene / "meta.json").read_text())
        h, w = meta["height"], meta["width"]
        files = sorted(f for f in (scene / "seq_000").glob("frame_*.f16") if f.name.count(".") == 1)[: a.frames]

        def load(f):
            t = torch.from_numpy(np.fromfile(f, "<f2").reshape(h, w, -1).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).to(a.device)

        k = 0.18 / torch.stack([luminance(load(f)[:, 0:3]).mean() for f in files[:8]]).mean().clamp(min=1e-6).view(1, 1, 1, 1)
        history = prev = was_raw = was_out = was_light = mean_light = where = None
        sums, count = np.zeros(6), 0
        for t, f in enumerate(files):
            frame = load(f)
            noisy, guides = frame[:, 0:3], frame[:, 3:15]
            emission = frame[:, 18:21] if frame.shape[1] >= 21 else 0
            history, prev, radiance, valid = M.step(net, noisy, guides, history, prev, t == 0, k if c.get("exposure") else None)
            raw, out = torch.log1p(k * (noisy + emission).clamp(min=0)), torch.log1p(k * (radiance + emission).clamp(min=0))
            light = torch.log1p(k * (radiance / (guides[:, M.ALBEDO] + M.EPS)).clamp(min=0))
            if t >= a.skip:
                motion = guides[:, M.MOTION]
                n = valid.sum().clamp(min=1)
                change = lambda now, was: float(((now - M.warp(was, motion)[0]).abs().mean(1, keepdim=True) * valid).sum() / n)  # noqa: E731
                gradient = float((out[..., 1:, :] - out[..., :-1, :]).abs().mean() + (out[..., :, 1:] - out[..., :, :-1]).abs().mean())
                drift = (light - M.warp(was_light, motion)[0]).mean(1, keepdim=True) * valid
                boil = float((F.avg_pool2d(drift, 8) / F.avg_pool2d(valid, 8).clamp(min=0.25)).abs().mean())
                slow = (light - M.warp(mean_light, motion)[0]).mean(1, keepdim=True) * valid
                slow = (F.avg_pool2d(slow, 8) / F.avg_pool2d(valid, 8).clamp(min=0.25)).abs()
                where = slow if where is None else where + slow
                slow = float(slow.mean())
                sums += (change(raw, was_raw), change(out, was_out), gradient, boil, slow, 0)
                count += 1
            if t > 0:
                mean_light = torch.where(valid > 0, torch.lerp(M.warp(mean_light, guides[:, M.MOTION])[0], light, 1 / 16), light)
            else:
                mean_light = light
            was_raw, was_out, was_light = raw, out, light
        sums /= max(count, 1)
        if where is not None:
            sums[5] = float((where / count).median())
        if a.maps and where is not None:
            from PIL import Image
            pathlib.Path(a.maps).mkdir(parents=True, exist_ok=True)
            m = F.interpolate(where / count, scale_factor=8, mode="nearest")[0, 0].cpu().numpy()
            shown = (out[0].permute(1, 2, 0).cpu().numpy() / np.log1p(4.0)).clip(0, 1)
            m = np.repeat((m / 0.05).clip(0, 1)[..., None], 3, 2)
            m = np.pad(m, ((0, shown.shape[0] - m.shape[0]), (0, shown.shape[1] - m.shape[1]), (0, 0)))
            Image.fromarray((np.concatenate([shown, m], 1) * 255).astype(np.uint8)).save(pathlib.Path(a.maps) / f"{scene.name}.png")
        totals += sums / len(scenes)
        print(f"{scene.name:16} input {sums[0]:.4f}  network {sums[1]:.4f}  gradient {sums[2]:.4f}  boil {sums[3]:.5f}  slow {sums[4]:.5f}  typical {sums[5]:.5f}", flush=True)
print(f"{'mean':16} input {totals[0]:.4f}  network {totals[1]:.4f}  gradient {totals[2]:.4f}  boil {totals[3]:.5f}  slow {totals[4]:.5f}  typical {totals[5]:.5f}")
