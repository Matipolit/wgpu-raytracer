"""Finish the references of a capture made with `--reference-groups G` (capture.rs):
every frame has a sidecar frame_NN.ref.f16 (G group means, then pixel-integrated
albedo and normal); the finished reference (finishing.py: firefly rejection, bounded
denoise) replaces the frame file's three reference channels, and the sidecar goes.

    python training/reference/finish_capture.py CAPTURE_DIR [--keep] [--device cuda]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finishing import finish  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("--keep", action="store_true", help="keep the sidecars")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
done = 0
changed = []
for meta_path in sorted(pathlib.Path(a.dir).rglob("meta.json")):
    meta = json.loads(meta_path.read_text())
    h, w = meta["height"], meta["width"]
    for side in sorted(meta_path.parent.rglob("frame_*.ref.f16")):
        frame = side.with_name(side.name.replace(".ref.f16", ".f16"))
        data = np.fromfile(side, dtype=np.float16).astype(np.float32)
        groups = (data.size - h * w * 6) // (h * w * 3)
        means = torch.from_numpy(data[: groups * h * w * 3].reshape(groups, h, w, 3)).to(a.device)
        guides = torch.from_numpy(data[groups * h * w * 3 :].reshape(h, w, 6)).to(a.device)
        final, _, report = finish(means, guides)
        pixels = np.memmap(frame, dtype=np.float16, mode="r+", shape=(h, w, 18))
        pixels[..., 15:18] = final.clamp(max=65000).cpu().numpy().astype(np.float16)
        pixels.flush()
        changed.append(report.get("changed_by_denoise", 0.0))
        if not a.keep:
            side.unlink()
        done += 1
print(f"finished {done} references; mean change by the denoise {np.mean(changed) if changed else 0:.4f} log units")
