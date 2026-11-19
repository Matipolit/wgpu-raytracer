"""Train the fast denoise + upscale network (fastnet.py) end to end.

    python training/fast/train.py --data DIR[,DIR...] --out RUN [--hours 3]
        [--channels 16] [--batch 4] [--crop 256]

Data: denoise + upscale captures (training/upscale/data.py), any number of directories
(e.g. one per output resolution). Every scene directory's last sequence is validation,
--holdout-scene is never trained on. Loss: L1 between output and reference in
exposure-normalized log space, plus half of the same on frame-to-frame changes
(motion compensated), over 16-frame sequences with gradients through both histories;
the first frame does not count and the first four count half.
"""
import argparse
import json
import pathlib
import random
import sys
import time

import torch

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "upscale"))
import data  # noqa: E402
import fastnet as N  # noqa: E402

A = N.A
p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--hours", type=float, default=3.0)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--crop", type=int, default=256)
p.add_argument("--channels", type=int, default=24)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--temporal-weight", type=float, default=0.5)
p.add_argument("--holdout-scene", default="fireplace-room")
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--init-a", help="stage A weights from a temporal denoiser run (best.pt, same configuration)")
# Stage A's second-generation options (training/temporal/model.py).
p.add_argument("--edge", action="store_true")
p.add_argument("--length", action="store_true")
p.add_argument("--feedback", default="filtered", choices=["filtered", "blend"])
p.add_argument("--hidden", type=int, default=0)
p.add_argument("--eps", type=float, default=0.01)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--head-hidden", type=int, default=8)
p.add_argument("--head-kernel", type=int, default=1, choices=[1, 3])
p.add_argument("--head-sharpen", action="store_true")
p.add_argument("--head-gather", action="store_true")
p.add_argument("--cubic-history", action="store_true")
p.add_argument("--freeze-a", action="store_true", help="train only the output-resolution head")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
import resource  # noqa: E402

resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
torch.manual_seed(a.seed)
random.seed(a.seed)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
A.configure(a.channels, (1, 2, 4), True, a.edge, a.length, a.feedback, a.hidden, a.eps)
N.configure(a.head_hidden, a.head_kernel, a.head_sharpen, a.head_gather, a.cubic_history)
config = dict(A.settings(), **N.settings())
(out / "config.json").write_text(json.dumps(config))

train, val, test = [], [], []
for seqs in data.load(a.data).values():
    if seqs[0].scene == a.holdout_scene:
        test += seqs
    else:
        train += seqs[:-1]
        val += seqs[-1:]
print(f"train {len(train)} sequences, validation {len(val)}, test ({a.holdout_scene}) {len(test)}, device {a.device}", flush=True)
# Batches hold sequences of one capture directory and scene size (equal tensor shapes).
groups = {}
for s in train:
    groups.setdefault((str(s.frames[0].filename.parent.parent.parent), s.lows[0]["frames"][0].shape, s.lows[1]["frames"][0].shape if len(s.lows) > 1 else 0), []).append(s)
groups = list(groups.values())

net = N.FastNet().to(a.device)
if a.init_a:
    net.a.load_state_dict(torch.load(a.init_a, map_location=a.device))
if a.freeze_a:
    net.a.requires_grad_(False)
optimizer = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=a.lr)


def run(seqs, scale_index, window, frames=None, zero_head=False):
    """The sequence through both stages. Returns (loss, per-frame squared log error of
    the second half, for PSNR)."""
    batch, coords, origin, scale = data.tensors(seqs, scale_index, window, a.device, frames, emission=True)
    exposure = None
    history = prev = state = prev_out = prev_ref = None
    referenced = False
    total, weights, errors = 0.0, 0.0, []
    head = (lambda f: torch.zeros(f.shape[0], 3, *f.shape[2:], device=f.device)) if zero_head else net.head
    for t, (low, guides, reference, jitter, _, emission) in enumerate(batch):
        noisy, low_guides = low[:, 0:3], low[:, 3:15]
        if exposure is None:
            # As the renderer's auto exposure: from the image's mean luminance (the
            # reference stands in for the displayed image). The network's features see
            # it off by up to 2x either way in training.
            exposure = 0.18 / N.luminance(batch[-1][2]).mean((1, 2, 3), keepdim=True).clamp(min=1e-6)
            seen = exposure * (2 ** (2 * torch.rand_like(exposure) - 1) if net.training else 1)
        history, prev, radiance, _ = A.step(net.a, noisy, low_guides, history, prev, t == 0, seen)
        x = radiance / (low_guides[:, A.ALBEDO] + A.EPS)
        image, state, valid = N.step_b(head, x, low_guides, guides, coords, origin, jitter, scale, state, seen, emission)
        log_out, log_ref = torch.log1p(exposure * image), torch.log1p(exposure * reference.clamp(min=0))
        loss = (log_out - log_ref).abs().mean()
        missing = float(reference.abs().sum()) == 0.0
        if t > 0 and not missing and referenced:
            motion = guides[:, A.MOTION]
            was_out, _ = A.warp(prev_out, motion)
            was_ref, _ = A.warp(prev_ref, motion)
            loss = loss + a.temporal_weight * (((log_out - was_out) - (log_ref - was_ref)).abs() * valid).mean()
        prev_out, prev_ref, referenced = log_out, log_ref, not missing
        if missing:
            # Warm-up frames carry no loss: gradients start at the first reference.
            history, prev, state = history.detach(), prev.detach(), tuple(x.detach() for x in state)
        # Frames without a reference (their reference is all zero: captures with
        # --reference-frames) only feed the histories.
        weight = 0.0 if t == 0 or missing else (0.5 if t < 4 else 1.0)
        total, weights = total + weight * loss, weights + weight
        if t >= len(batch) // 2 and not missing:
            errors.append(float(((log_out - log_ref) ** 2).mean().detach()))
    return total / max(weights, 1e-6), errors


@torch.no_grad()
def evaluate(seqs, zero_head=False):
    net.eval()
    losses, errors = [], []
    for s in seqs:
        for scale_index in range(len(s.lows)):
            loss, e = run([s], scale_index, None, zero_head=zero_head)
            losses.append(float(loss))
            errors += e
    mse = sum(errors) / max(len(errors), 1)
    net.train()
    return sum(losses) / max(len(losses), 1), -10 * torch.log10(torch.tensor(max(mse, 1e-12))).item()


start, step, best, running = time.time(), 0, float("inf"), 0.0
budget = a.hours * 3600
while time.time() - start < budget and (a.max_steps == 0 or step < a.max_steps):
    group = random.choice(groups)
    seqs = random.choices(group, k=a.batch)
    size = seqs[0].size
    crop = min(a.crop, size[0], size[1])
    window = (random.randint(0, size[0] - crop), random.randint(0, size[1] - crop), crop)
    for g in optimizer.param_groups:
        g["lr"] = a.lr * (1 - 0.9 * min(1.0, step / a.max_steps if a.max_steps else (time.time() - start) / budget))
    loss, _ = run(seqs, random.randrange(len(seqs[0].lows)), window)
    if not torch.isfinite(loss):
        print(f"step {step}: non-finite loss, skipped", flush=True)
        optimizer.zero_grad()
        continue
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    step += 1
    running = 0.98 * running + 0.02 * float(loss) if step > 1 else float(loss)
    if step % a.eval_every == 0:
        value, psnr = evaluate(val)
        print(f"step {step} t={time.time() - start:.0f}s loss {running:.4f} val {value:.4f} psnr {psnr:.2f} dB", flush=True)
        if value < best:
            best = value
            torch.save(net.state_dict(), out / "best.pt")
if not (out / "best.pt").exists():
    torch.save(net.state_dict(), out / "best.pt")
net.load_state_dict(torch.load(out / "best.pt", map_location=a.device))
summary = dict(steps=step)
summary["val"], summary["val_psnr"] = evaluate(val)
summary["val_zero_head"], summary["val_zero_head_psnr"] = evaluate(val, zero_head=True)
if test:
    summary["test"], summary["test_psnr"] = evaluate(test)
(out / "summary.json").write_text(json.dumps(summary))
print(summary, flush=True)
