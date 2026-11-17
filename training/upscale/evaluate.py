"""Evaluate the denoise + upscale network per render scale against the references.

    python training/upscale/evaluate.py --data DIR --weights RUN/best.pt --out RUN/eval [--all]

Methods: splat (raw 1-spp samples gathered to output resolution), taau (the in-renderer
upscaler, src/raytracer/upscale.wgsl, on raw input) and the network. Metrics in log(1+x):
L1, PSNR (peak 1) and flicker (frame-to-frame change the reference does not have).
Held out: every sequence with --all (a separate evaluation set), otherwise each scene's
last two sequences and all of --holdout-scene.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import data as D  # noqa: E402
import model as M  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--weights", required=True)
p.add_argument("--out", required=True)
p.add_argument("--holdout-scene", default="fireplace-room")
p.add_argument("--all", action="store_true")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
config = pathlib.Path(a.weights).with_name("config.json")
if config.exists():
    c = json.loads(config.read_text())
    M.configure(c["channels"], c["dilations"])
net = M.Net().to(a.device)
net.load_state_dict(torch.load(a.weights, map_location=a.device))
net.eval()


def image(x):
    v = (1 - torch.exp(-x.clamp(min=0).cpu() * 1.5)).clamp(0, 1) ** (1 / 2.2)
    return (v[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


scenes = D.load(a.data)
results = {}
for name, seqs in scenes.items():
    held = seqs if a.all or name == a.holdout_scene else seqs[-2:]
    results[name] = {}
    for k in range(len(held[0].lows)):
        stats = {m: dict(l1=0.0, mse=0.0, flicker=0.0, n=0, nf=0) for m in ("splat", "taau", "network")}
        strip = []
        for si, seq in enumerate(held):
            frames, coords, origin, scale = D.tensors([seq], k, None, a.device)
            state = {"taau": (None, None), "network": (None, None)}
            prev_out, prev_ref = {}, None
            for t, (low, guides, ref, jitter) in enumerate(frames):
                with torch.no_grad():
                    outs = {"splat": M.splat(low, coords, origin, jitter, scale)}
                    h, g, outs["taau"] = M.taau_step(low, guides, coords, origin, jitter, scale, *state["taau"], t == 0)
                    state["taau"] = (h, g)
                    y, g, outs["network"], valid = M.step(net, low, guides, coords, origin, jitter, scale, *state["network"], t == 0)
                    state["network"] = (y, g)
                tr = M.tonemap(ref)
                for m, o in outs.items():
                    to = M.tonemap(o)
                    s = stats[m]
                    s["l1"] += (to - tr).abs().mean().item()
                    s["mse"] += ((to.clamp(0, 1) - tr.clamp(0, 1)) ** 2).mean().item()
                    s["n"] += 1
                    if t > 0:
                        motion = guides[:, M.MOTION]
                        wo, _ = M.warp(prev_out[m], motion)
                        wr, _ = M.warp(prev_ref, motion)
                        s["flicker"] += (((to - wo) - (tr - wr)).abs() * valid).sum().item() / max(valid.sum().item(), 1)
                        s["nf"] += 1
                    prev_out[m] = to
                prev_ref = tr
                if si == 0 and t in (0, len(frames) // 2, len(frames) - 1):
                    strip.append(np.concatenate([image(outs["splat"]), image(outs["taau"]), image(outs["network"]), image(ref)], 1))
        label = f"{seq.lows[k]['scale'][0]:.3f}"
        results[name][label] = {
            m: dict(l1=s["l1"] / s["n"], psnr=M.psnr(s["mse"] / s["n"]), flicker=s["flicker"] / max(s["nf"], 1))
            for m, s in stats.items()
        }
        from PIL import Image  # noqa: E402

        Image.fromarray(np.concatenate(strip, 0)).save(out / f"{name}-{label}.png")
        r = results[name][label]
        print(f"{name:24} scale {label}  " + "  ".join(f"{m}: PSNR {v['psnr']:.2f} flicker {v['flicker']:.4f}" for m, v in r.items()), flush=True)
(out / "metrics.json").write_text(json.dumps(results, indent=1))
labels = sorted({label for r in results.values() for label in r})
print("\nmean over scenes (PSNR / flicker):")
for label in labels:
    rows = [r[label] for r in results.values() if label in r]
    print(f"  scale {label}: " + "  ".join(
        f"{m} {sum(x[m]['psnr'] for x in rows) / len(rows):.2f} / {sum(x[m]['flicker'] for x in rows) / len(rows):.4f}"
        for m in ("splat", "taau", "network")))
print("strips: splat | taau | network | reference; rows: first, middle, last frame")
