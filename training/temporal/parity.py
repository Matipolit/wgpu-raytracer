"""Check WGSL inference against PyTorch on a captured sequence.

    training/.venv/bin/python training/temporal/parity.py SEQ_DIR WEIGHTS.pt WIDTH HEIGHT
SEQ_DIR comes from `--capture ... --no-reference --temporal-model MODEL.wgsl`.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import model as M  # noqa: E402

seq, weights, w, h = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
import json  # noqa: E402

config = pathlib.Path(weights).with_name("config.json")
if config.exists():
    c = json.loads(config.read_text())
    M.configure(c["channels"], c["dilations"], c.get("half", False))
net = M.Net()
net.load_state_dict(torch.load(weights, map_location="cpu"))
history = prev = None
worst = 0.0
for t, f in enumerate(sorted(seq.glob("frame_*.f16"))):
    x = torch.from_numpy(np.fromfile(f, "<f2").reshape(h, w, 18).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    # The renderer works on f32 inputs; the capture stored f16. Compare against the
    # renderer's own inputs as closely as possible: use the f16 values for both.
    with torch.no_grad():
        y, g, radiance, _ = M.step(net, x[:, 0:3], x[:, 3:15], history, prev, t == 0)
    history, prev = y, g
    gpu = torch.from_numpy(np.fromfile(str(f).replace(".f16", ".den.f32"), "<f4").reshape(h, w, 3).transpose(2, 0, 1).copy()).unsqueeze(0)
    diff = (M.tonemap(gpu) - M.tonemap(radiance)).abs()
    worst = max(worst, diff.mean().item())
    print(f"frame {t}: mean |tonemapped diff| {diff.mean().item():.5f}, 99.9% {diff.flatten().quantile(0.999).item():.4f}")
print("PARITY", "OK" if worst < 0.01 else "MISMATCH", worst)
