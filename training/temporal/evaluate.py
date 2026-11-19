"""Evaluate the temporal denoiser on held-out sequences.

    training/.venv/bin/python training/temporal/evaluate.py --data DIR --weights RUN/best.pt --out RUN/eval

Held out: each scene's last two sequences and every sequence of --holdout-scene
(same split as train.py). Methods, all against the high-spp references:
  raw        the 1-spp input
  ema        non-learned baseline: reprojection with the same validity test and an
             exponential moving average (alpha 0.8), no spatial filtering
  network    the trained model
Metrics in tonemapped space log(1+x): mean L1, PSNR (peak 1), and temporal
instability: mean |(y_t - warp(y_t-1)) - (ref_t - warp(ref_t-1))| on valid pixels.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import model as M  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--weights", required=True)
p.add_argument("--out", required=True)
p.add_argument("--holdout-scene", default="fireplace-room")
p.add_argument("--all", action="store_true", help="every sequence is held out (separate evaluation set)")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
config = pathlib.Path(a.weights).with_name("config.json")
if config.exists():
    c = json.loads(config.read_text())
    M.configure(**c)
EXPOSURE = config.exists() and c.get("exposure", False)
net = M.Net().to(a.device)
net.load_state_dict(torch.load(a.weights, map_location=a.device))
net.eval()


def ema_step(noisy, guides, history, prev, first):
    albedo = guides[:, M.ALBEDO]
    demod = noisy / (albedo + M.EPS)
    dist, normal, kind = guides[:, M.DIST : M.DIST + 1], guides[:, M.NORMAL], guides[:, M.KIND : M.KIND + 1]
    if first:
        y = demod
        valid = torch.zeros_like(dist)
    else:
        sampled, inside = M.warp(torch.cat((history, prev), 1), guides[:, M.MOTION])
        ok = (inside & ((sampled[:, 3:4] - dist).abs() <= 0.1 * dist + 1e-3)
              & ((sampled[:, 4:7] * normal).sum(1, keepdim=True) >= 0.8) & ((sampled[:, 7:8] - kind).abs() < 0.5))
        valid = ok.float()
        alpha = 0.8 * valid
        y = alpha * sampled[:, 0:3] + (1 - alpha) * demod
    return y, torch.cat((dist, normal, kind), 1), y * (albedo + M.EPS), valid


def tm(x):
    return M.tonemap(x)


def image(x):
    v = (1 - torch.exp(-x.clamp(min=0).cpu() * 1.5)).clamp(0, 1) ** (1 / 2.2)
    return (v[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


results = {}
strips = []
scenes = sorted(d for d in pathlib.Path(a.data).iterdir() if (d / "meta.json").exists())
for scene_dir in scenes:
    meta = json.loads((scene_dir / "meta.json").read_text())
    h, w = meta["height"], meta["width"]
    seqs = sorted(scene_dir.glob("seq_*"))
    held = seqs if a.all or scene_dir.name == a.holdout_scene else seqs[-2:]
    stats = {m: dict(l1=0.0, mse=0.0, flicker=0.0, n=0, nf=0) for m in ("raw", "ema", "network")}
    for si, seq in enumerate(held):
        frames = [torch.from_numpy(np.fromfile(f, "<f2").reshape(h, w, -1).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
                  for f in sorted(seq.glob("frame_*.f16")) if f.name.count(".") == 1]
        frames = [torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).to(a.device) for f in frames]
        state = {"ema": (None, None), "network": (None, None)}
        prev_out = {}
        prev_ref = None
        row = []
        for t, f in enumerate(frames):
            noisy, guides, ref = f[:, 0:3], f[:, 3:15], f[:, 15:18]
            # Captures with `--capture-emission` keep directly visible emission apart
            # from the noisy radiance; every method gets it added back.
            emission = f[:, 18:21] if f.shape[1] >= 21 else torch.zeros_like(noisy)
            if t == 0:
                last = frames[-1][:, 15:18]
                seen = 0.18 / (0.2126 * last[:, 0] + 0.7152 * last[:, 1] + 0.0722 * last[:, 2]).mean().clamp(min=1e-6).view(1, 1, 1, 1) if EXPOSURE else None
            outs = {"raw": noisy + emission}
            with torch.no_grad():
                y, g, rad, valid = ema_step(noisy, guides, *state["ema"], t == 0)
                state["ema"] = (y, g)
                outs["ema"] = rad + emission
                y, g, rad, valid_net = M.step(net, noisy, guides, *state["network"], t == 0, seen)
                state["network"] = (y, g)
                outs["network"] = rad + emission
            tr = tm(ref)
            for m, o in outs.items():
                to = tm(o)
                s = stats[m]
                s["l1"] += (to - tr).abs().mean().item()
                s["mse"] += ((to.clamp(0, 1) - tr.clamp(0, 1)) ** 2).mean().item()
                s["n"] += 1
                if t > 0:
                    motion = guides[:, M.MOTION]
                    wo, _ = M.warp(prev_out[m], motion)
                    wr, _ = M.warp(prev_ref, motion)
                    s["flicker"] += (((to - wo) - (tr - wr)).abs() * valid_net).sum().item() / max(valid_net.sum().item(), 1)
                    s["nf"] += 1
                prev_out[m] = to
            prev_ref = tr
            if si == 0 and t in (0, 3, len(frames) - 1):
                row.append(np.concatenate([image(noisy), image(outs["ema"]), image(outs["network"]), image(ref)], 1))
        if si == 0:
            strips.append(np.concatenate(row, 0))
    results[scene_dir.name] = {
        m: dict(l1=s["l1"] / s["n"], psnr=10 * math.log10(1 / max(s["mse"] / s["n"], 1e-12)),
                flicker=s["flicker"] / max(s["nf"], 1)) for m, s in stats.items()
    }
    r = results[scene_dir.name]
    print(f"{scene_dir.name:24} " + "  ".join(f"{m}: L1 {v['l1']:.4f} PSNR {v['psnr']:.2f} flicker {v['flicker']:.4f}" for m, v in r.items()), flush=True)
(out / "metrics.json").write_text(json.dumps(results, indent=1))
from PIL import Image  # noqa: E402

for i, s in enumerate(strips):
    Image.fromarray(s).save(out / f"{scenes[i].name}.png")
print("columns: noisy 1 spp | EMA baseline | network | reference; rows: frames 0, 3, last")
