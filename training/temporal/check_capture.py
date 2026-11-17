"""Sanity checks for captured sequences: motion vectors and guide ranges.

    training/.venv/bin/python training/temporal/check_capture.py DATA/scene [--image OUT.png]

For consecutive frames, the previous reference warped by the current motion
vectors must match the current reference much better than the unwarped one
(on pixels whose reprojection the denoiser would accept).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import model as M  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("scene")
p.add_argument("--image")
a = p.parse_args()
scene = pathlib.Path(a.scene)
meta = json.loads((scene / "meta.json").read_text())
h, w = meta["height"], meta["width"]
warped_err = static_err = 0.0
count = 0
for seq in sorted(scene.glob("seq_*"))[:6]:
    frames = [torch.from_numpy(np.fromfile(f, "<f2").reshape(h, w, 18).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
              for f in sorted(seq.glob("frame_*.f16"))]
    for prev, cur in zip(frames, frames[1:]):
        g = cur[:, 3:15]
        motion = g[:, M.MOTION]
        ref_prev, ref_cur = M.tonemap(prev[:, 15:18]), M.tonemap(cur[:, 15:18])
        warped, inside = M.warp(torch.cat((ref_prev, prev[:, 3 + M.DIST : 4 + M.DIST]), 1), motion)
        dist = g[:, M.DIST : M.DIST + 1]
        ok = inside & ((warped[:, 3:4] - dist).abs() <= 0.1 * dist + 1e-3)
        okf = ok.float()
        warped_err += ((warped[:, :3] - ref_cur).abs() * okf).sum().item() / max(okf.sum().item(), 1)
        static_err += ((ref_prev - ref_cur).abs() * okf).sum().item() / max(okf.sum().item(), 1)
        count += 1
print(f"{scene.name}: mean |ref_t - warp(ref_t-1)| {warped_err / count:.4f} vs unwarped {static_err / count:.4f} "
      f"(accepted pixels), motion |max| {motion.abs().max().item():.1f}px")
f = np.fromfile(sorted((sorted(scene.glob('seq_*'))[0]).glob('frame_*.f16'))[1], "<f2").reshape(h, w, 18).astype(np.float32)
names = meta["channels"]
for i, n in enumerate(names):
    c = f[..., i]
    print(f"  {n:10} min {np.nanmin(c):9.3f} max {np.nanmax(c):9.3f} mean {np.nanmean(c):8.3f}")
if a.image:
    from PIL import Image

    def tm(x):
        x = 1 - np.exp(-np.clip(x, 0, None))
        return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    mv = f[..., 11:13]
    mv_img = np.stack([np.clip(0.5 + mv[..., 0] / 20, 0, 1), np.clip(0.5 + mv[..., 1] / 20, 0, 1), np.full((h, w), 0.5)], -1)
    tiles = [tm(f[..., 0:3]), tm(f[..., 15:18]), (np.clip(f[..., 3:6], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8),
             ((f[..., 7:10] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8), (mv_img * 255).astype(np.uint8)]
    Image.fromarray(np.concatenate(tiles, 1)).save(a.image)
