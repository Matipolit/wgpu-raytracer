"""The baseline the fast network has to beat: a temporal denoiser at render resolution
followed by the renderer's TAAU upscaler, on the fast trainer's validation split and
metric (L1 and PSNR in exposure-normalized log space, second half of every sequence).

    python training/fast/baseline.py --data DIR[,DIR...] --td RUN/best.pt [--fast RUN/best.pt]

With --fast the fast network is evaluated the same way next to it.
"""
import argparse
import importlib.util
import json
import pathlib
import resource
import sys

import torch

resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "upscale"))
import data  # noqa: E402
import fastnet as N  # noqa: E402

A = N.A
spec = importlib.util.spec_from_file_location("joint_v3", here.parent / "upscale" / "model.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--td", required=True)
p.add_argument("--fast")
p.add_argument("--holdout-scene", default="fireplace-room")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()


def config(weights):
    return json.loads(pathlib.Path(weights).with_name("config.json").read_text())


val, test = [], []
for seqs in data.load(a.data).values():
    (test if seqs[0].scene == a.holdout_scene else val).extend(seqs if seqs[0].scene == a.holdout_scene else seqs[-1:])


@torch.no_grad()
def evaluate(seqs, weights, fast):
    c = config(weights)
    A.configure(**c)
    net = (N.FastNet() if fast else A.Net()).to(a.device)
    net.load_state_dict(torch.load(weights, map_location=a.device))
    net.eval()
    stage_a = net.a if fast else net
    losses, errors = [], []
    for s in seqs:
        for scale_index in range(len(s.lows)):
            batch, coords, origin, scale = data.tensors([s], scale_index, None, a.device, emission=True)
            exposure = 0.18 / N.luminance(batch[-1][2]).mean((1, 2, 3), keepdim=True).clamp(min=1e-6)
            seen = exposure if c.get("exposure", fast) else None
            history = prev = state = taau = taau_guides = None
            for t, (low, guides, reference, jitter, low_emission, emission) in enumerate(batch):
                history, prev, radiance, _ = A.step(stage_a, low[:, 0:3], low[:, 3:15], history, prev, t == 0, seen)
                if fast:
                    x = radiance / (low[:, 3:6] + A.EPS)
                    image, state, _ = N.step_b(net.head, x, low[:, 3:15], guides, coords, origin, jitter, scale, state, exposure, emission)
                else:
                    taau, taau_guides, image = J.taau_step(radiance + low_emission, guides, coords, origin, jitter, scale, taau, taau_guides, t == 0)
                if float(reference.abs().sum()) == 0.0:
                    continue
                difference = torch.log1p(exposure * image.clamp(min=0)) - torch.log1p(exposure * reference.clamp(min=0))
                if t > 0:
                    losses.append(float(difference.abs().mean()))
                if t >= len(batch) // 2:
                    errors.append(float((difference**2).mean()))
    mse = sum(errors) / max(len(errors), 1)
    return sum(losses) / max(len(losses), 1), -10 * torch.log10(torch.tensor(max(mse, 1e-12))).item()


for name, seqs in (("validation", val), (f"hold-out ({a.holdout_scene})", test)):
    if not seqs:
        continue
    l1, psnr = evaluate(seqs, a.td, False)
    print(f"{name}: denoiser + TAAU   L1 {l1:.4f}  PSNR {psnr:.2f} dB")
    if a.fast:
        l1, psnr = evaluate(seqs, a.fast, True)
        print(f"{name}: fast network      L1 {l1:.4f}  PSNR {psnr:.2f} dB")
