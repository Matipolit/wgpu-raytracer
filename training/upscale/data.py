"""Denoise + upscale captures (src/capture.rs --upscale-capture, optionally --low-only).

Per scene directory: meta.json (output crops: 18 channels, reference in 15:18),
low_meta.json (or meta["low"]) with one entry per render scale, and per sequence
crop.json (output crop origin), low.json (low crop origin and per-frame jitter per scale),
frame_NN.f16 and frame_NN.lowI.f16.
"""
import json
import pathlib

import numpy as np
import torch


class Sequence:
    def __init__(self, scene, directory, meta, lows):
        self.scene = scene
        h, w = meta["height"], meta["width"]
        self.size = (h, w)
        frames = sorted(f for f in directory.glob("frame_*.f16") if f.name.count(".") == 1)
        self.frames = [np.memmap(f, dtype="<f2", mode="r", shape=(h, w, 18)) for f in frames]
        crop = json.loads((directory / "crop.json").read_text())
        self.origin = (crop["x"], crop["y"])
        low_info = json.loads((directory / "low.json").read_text())
        self.lows = []
        for i, (scale, info) in enumerate(zip(lows, low_info)):
            lh, lw = scale["height"], scale["width"]
            files = [f.with_name(f.name.replace(".f16", f".low{i}.f16")) for f in frames]
            self.lows.append(
                dict(
                    frames=[np.memmap(f, dtype="<f2", mode="r", shape=(lh, lw, 15)) for f in files],
                    origin=tuple(info["origin"]),
                    jitter=[tuple(j) for j in info["jitter"]],
                    scale=(scale["scale_x"], scale["scale_y"]),
                )
            )


def load(roots):
    """All complete sequences under one or more capture roots (a list, or a comma-separated
    string), grouped by root and scene: {"root/scene": [Sequence]}; Sequence.scene is the
    scene name (the hold-out key)."""
    if isinstance(roots, str):
        roots = [r for r in roots.split(",") if r]
    scenes = {}
    for root in roots:
        scenes.update({f"{pathlib.Path(root).name}/{k}": v for k, v in load_root(root).items()})
    return scenes


def load_root(root):
    scenes = {}
    for scene_dir in sorted(pathlib.Path(root).iterdir()):
        if not (scene_dir / "meta.json").exists():
            continue
        meta = json.loads((scene_dir / "meta.json").read_text())
        low_meta = scene_dir / "low_meta.json"
        lows = json.loads(low_meta.read_text()) if low_meta.exists() else meta.get("low", [])
        if not lows:
            continue
        seqs = []
        for directory in sorted(scene_dir.glob("seq_*")):
            if not (directory / "low.json").exists():
                continue
            seq = Sequence(scene_dir.name, directory, meta, lows)
            if len(seq.frames) == meta["frames"] and all(len(l["frames"]) == meta["frames"] for l in seq.lows):
                seqs.append(seq)
        if seqs:
            scenes[scene_dir.name] = seqs
    return scenes


def split(scenes, holdout_scene):
    """Train: all but each scene's last two sequences; validation: those; test: holdout scene."""
    train, val, test = [], [], []
    for seqs in scenes.values():
        if seqs[0].scene == holdout_scene:
            test += seqs
        else:
            train += seqs[:-2]
            val += seqs[-2:]
    return train, val, test


def tensors(seqs, scale_index, window, device, frames=None):
    """Batch of sequences at one scale. window = (oy, ox, crop) inside the stored output
    crop, or None for the whole crop. Returns (frames, coords, origin, scale): frames is a
    list of (low (B,15,h,w), guides (B,12,H,W), reference (B,3,H,W), jitter (B,2));
    coords (B,2,H,W) absolute output-pixel centres; origin (B,2) low crop origins;
    scale (B,2) render/output per axis."""
    t_count = frames or len(seqs[0].frames)
    h, w = seqs[0].size
    oy, ox, c = window if window else (0, 0, None)
    ch, cw = (c, c) if c else (h, w)
    out = []
    for t in range(t_count):
        hi = np.stack([np.asarray(s.frames[t][oy : oy + ch, ox : ox + cw]) for s in seqs])
        lo = np.stack([np.asarray(s.lows[scale_index]["frames"][t]) for s in seqs])
        hi = torch.nan_to_num(torch.from_numpy(hi).to(device).float().permute(0, 3, 1, 2), nan=0.0, posinf=0.0, neginf=0.0)
        lo = torch.nan_to_num(torch.from_numpy(lo).to(device).float().permute(0, 3, 1, 2), nan=0.0, posinf=0.0, neginf=0.0)
        jitter = torch.tensor([s.lows[scale_index]["jitter"][t] for s in seqs], device=device)
        out.append((lo, hi[:, 3:15], hi[:, 15:18], jitter))
    ys, xs = torch.meshgrid(torch.arange(ch, device=device), torch.arange(cw, device=device), indexing="ij")
    coords = torch.stack(
        [torch.stack((xs + s.origin[0] + ox + 0.5, ys + s.origin[1] + oy + 0.5)).float() for s in seqs]
    )
    origin = torch.tensor([s.lows[scale_index]["origin"] for s in seqs], device=device, dtype=torch.float32)
    scale = torch.tensor([s.lows[scale_index]["scale"] for s in seqs], device=device, dtype=torch.float32)
    return out, coords, origin, scale


def geometry(seq, scale_index, device):
    """(coords (1,2,H,W), origin (1,2), scale (1,2)) of a whole stored crop."""
    h, w = seq.size
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coords = torch.stack((xs + seq.origin[0] + 0.5, ys + seq.origin[1] + 0.5)).float().unsqueeze(0)
    low = seq.lows[scale_index]
    origin = torch.tensor([low["origin"]], device=device, dtype=torch.float32)
    scale = torch.tensor([low["scale"]], device=device, dtype=torch.float32)
    return coords, origin, scale
