"""How much history a training window loses at its edges.

A pixel whose reprojection falls outside the window starts a new history, although the
renderer (whole frame) would have kept it. For centred windows of several sizes this
compares the history length at the last frame, computed inside the window alone, with
the length the same pixels have when the whole captured window is available.

    python window_history.py RUN_DIR[,RUN_DIR...] [--sizes 64,96,128,192,256]
"""
import argparse
import json
import pathlib

import numpy as np
import torch

import model as M

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("--sizes", default="64,96,128,192,256")
a = p.parse_args()
sizes = [int(s) for s in a.sizes.split(",")]


def lengths(guides):
    """History length per pixel at the last frame; guides (T, 12, H, W)."""
    length = torch.zeros_like(guides[0, 0:1]).unsqueeze(0)
    prev = None
    for g in guides:
        g = g.unsqueeze(0)
        dist, normal, kind = g[:, M.DIST : M.DIST + 1], g[:, M.NORMAL], g[:, M.KIND : M.KIND + 1]
        if prev is not None:
            sampled, inside = M.warp(torch.cat((length, prev), 1), g[:, M.MOTION])
            agree = inside & ((sampled[:, 1:2] - dist).abs() <= 0.1 * dist + 1e-3) & ((sampled[:, 2:5] * normal).sum(1, keepdim=True) >= 0.8) & ((sampled[:, 5:6] - kind).abs() < 0.5)
            length = (sampled[:, 0:1] + 1) * agree
        prev = torch.cat((dist, normal, kind), 1)
    return length[0, 0]


rows = {s: [] for s in sizes}
for run in a.data.split(","):
    for scene in sorted(pathlib.Path(run).iterdir()):
        if not (scene / "meta.json").exists():
            continue
        meta = json.loads((scene / "meta.json").read_text())
        h, w = meta["height"], meta["width"]
        for seq in sorted(scene.glob("seq_*")):
            files = sorted(f for f in seq.glob("frame_*.f16") if f.name.count(".") == 1)
            if len(files) != meta["frames"]:
                continue
            frames = np.stack([np.memmap(f, dtype="<f2", mode="r", shape=(h, w, f.stat().st_size // (h * w * 2)))[..., 3:15] for f in files])
            guides = torch.nan_to_num(torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2))
            whole = lengths(guides)
            for s in sizes:
                if s >= min(h, w):
                    continue
                y, x = (h - s) // 2, (w - s) // 2
                alone = lengths(guides[:, :, y : y + s, x : x + s])
                kept = whole[y : y + s, x : x + s]
                rows[s].append((float(alone.mean()), float(kept.mean()), float((alone < 0.5 * kept).float().mean()), meta["frames"] - 1))
print("window  history alone / with the whole capture (frames)   lost   pixels with under half   sequences")
for s in sizes:
    if rows[s]:
        r = np.array(rows[s])
        print(f"{s:5d}   {r[:, 0].mean():6.2f} / {r[:, 1].mean():6.2f} of {r[:, 3].mean():.0f}   {100 * (1 - r[:, 0].mean() / r[:, 1].mean()):5.1f}%   {100 * r[:, 2].mean():5.1f}%   {len(r)}")
