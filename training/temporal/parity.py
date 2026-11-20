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
    M.configure(**c)
else:
    c = {}
net = M.Net()
net.load_state_dict(torch.load(weights, map_location="cpu"))
history = prev = gpu = None
worst = 0.0
for t, f in enumerate(sorted(f for f in seq.glob("frame_*.f16") if f.name.count(".") == 1)):
    x = torch.from_numpy(np.fromfile(f, "<f2").reshape(h, w, -1).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    # The renderer works on f32 inputs; the capture stored f16. Compare against the
    # renderer's own inputs as closely as possible: use the f16 values for both.
    exposure = None
    if c.get("exposure"):
        # As the renderer (export.py csTdExposure): 0.18 / the mean luminance of the
        # frame it displayed last, over a 64x64 grid of pixels; 1 on the first frame.
        exposure = 1.0
        if gpu is not None:
            ys = (torch.arange(min(h, 64)) * h // min(h, 64)).clamp(max=h - 1)
            xs = (torch.arange(min(w, 64)) * w // min(w, 64)).clamp(max=w - 1)
            grid = gpu[0][:, ys][:, :, xs]
            shown = float((0.2126 * grid[0] + 0.7152 * grid[1] + 0.0722 * grid[2]).mean())
            exposure = 0.18 / max(shown, 1e-6) if shown > 0 else 1.0
    # --capture-emission stores the directly visible emission apart and takes it out of
    # the input, which the renderer's denoiser input never contained: put it back.
    emission = x[:, 18:21] if x.shape[1] >= 21 else 0.0
    with torch.no_grad():
        y, g, radiance, _ = M.step(net, x[:, 0:3] + emission, x[:, 3:15], history, prev, t == 0, exposure)
    history, prev = y, g
    radiance = radiance + emission
    gpu = torch.from_numpy(np.fromfile(str(f).replace(".f16", ".den.f32"), "<f4").reshape(h, w, 3).transpose(2, 0, 1).copy()).unsqueeze(0)
    diff = (M.tonemap(gpu) - M.tonemap(radiance)).abs()
    worst = max(worst, diff.mean().item())
    print(f"frame {t}: mean |tonemapped diff| {diff.mean().item():.5f}, 99.9% {diff.flatten().quantile(0.999).item():.4f}")
print("PARITY", "OK" if worst < 0.01 else "MISMATCH", worst)
