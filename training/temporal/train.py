"""Train the temporal denoiser on captured sequences (see src/capture.rs).

    training/.venv/bin/python training/temporal/train.py --data DIR --out RUN [--hours 3]

DIR holds one directory per scene, each with seq_NNN/frame_MM.f16 and meta.json.
Holdout: every scene's last two sequences, plus all of --holdout-scene, are never
trained on; validation reports the denoiser against the raw 1-spp input on them.
"""
import argparse
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import model as M

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--hours", type=float, default=3.0)
p.add_argument("--threads", type=int, default=10)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--crop", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--holdout-scene", default="fireplace-room")
p.add_argument("--val-sequences", type=int, default=2, help="last sequences of every scene directory held out for validation")
p.add_argument("--temporal-weight", type=float, default=0.5)
p.add_argument("--eval-every", type=int, default=400)
p.add_argument("--channels", type=int, default=16)
p.add_argument("--dilations", default="1,2,4")
p.add_argument("--half", action="store_true", help="3x3 convs at half resolution")
p.add_argument("--init", help="start from these weights (same architecture)")
# Second-generation options (model.py).
p.add_argument("--edge", action="store_true", help="explicit edge stops in the kernels")
p.add_argument("--length", action="store_true", help="history length input, running-mean default blend")
p.add_argument("--feedback", default="filtered", choices=["filtered", "blend"], help="what becomes the history")
p.add_argument("--hidden", type=int, default=0, help="hidden units of the per-pixel head")
p.add_argument("--eps", type=float, default=0.01, help="demodulation floor")
p.add_argument("--exposure", action="store_true", help="exposure-normalized loss and input features")
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
torch.set_num_threads(a.threads)
# One descriptor per memory-mapped frame: lift the soft limit to the hard one.
import resource  # noqa: E402

_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
torch.manual_seed(a.seed)
random.seed(a.seed)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
M.configure(a.channels, [int(d) for d in a.dilations.split(",")], a.half, a.edge, a.length, a.feedback, a.hidden, a.eps)
(out / "config.json").write_text(json.dumps(dict(M.settings(), exposure=a.exposure)))


REFERENCES = 10**9


def load(scene_dir):
    """Sequences as lists of memory-mapped frames (H, W, 18) f16: parallel training
    runs share one copy in the page cache and read only the crops they use."""
    meta = json.loads((scene_dir / "meta.json").read_text())
    h, w = meta["height"], meta["width"]
    # Frames before the last `reference_frames` have no reference (they only warm the
    # history up); all sequences of a run must agree.
    global REFERENCES
    REFERENCES = min(REFERENCES, meta.get("reference_frames", meta["frames"]))
    seqs = []
    for seq in sorted(scene_dir.glob("seq_*")):
        # Frame files only: not the render-scale inputs (.lowN) or reference sidecars.
        files = sorted(f for f in seq.glob("frame_*.f16") if f.name.count(".") == 1)
        if len(files) == meta["frames"]:
            # 18 channels, or 21 with the directly visible emission (`--capture-emission`).
            seqs.append([np.memmap(f, dtype="<f2", mode="r", shape=(h, w, f.stat().st_size // (h * w * 2))) for f in files])
    return seqs


train, val, test = [], [], []
# --data takes several capture directories (e.g. one per output resolution).
for scene_dir in sorted(d for data in a.data.split(",") for d in pathlib.Path(data).iterdir()):
    if not (scene_dir / "meta.json").exists():
        continue
    seqs = load(scene_dir)
    if scene_dir.name == a.holdout_scene:
        test += [(scene_dir.name, s) for s in seqs]
    else:
        train += [(scene_dir.name, s) for s in seqs[: -a.val_sequences]]
        val += [(scene_dir.name, s) for s in seqs[-a.val_sequences :]]
print(f"train {len(train)} seqs, val {len(val)}, test ({a.holdout_scene}) {len(test)}, device {a.device}", flush=True)


def tensors(seq, window=None):
    """(noisy, guides, ref, emission) as (T, C, H, W) on the device; window = (y, x,
    size). Emission is zero for captures that keep it inside the noisy radiance."""
    if window:
        y, x, c = window
        frames = [np.ascontiguousarray(f[y : y + c, x : x + c]) for f in seq]
    else:
        frames = [np.asarray(f) for f in seq]
    t = torch.from_numpy(np.stack(frames)).to(a.device).float().permute(0, 3, 1, 2)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t[:, 0:3], t[:, 3:15], t[:, 15:18], (t[:, 18:21] if t.shape[1] >= 21 else torch.zeros_like(t[:, 0:3]))


def exposure_of(ref):
    """0.18 / mean luminance of the last reference frame (every sequence has that one),
    per sequence (B, 1, 1, 1)."""
    lum = 0.2126 * ref[-1][:, 0] + 0.7152 * ref[-1][:, 1] + 0.0722 * ref[-1][:, 2]
    return (0.18 / lum.mean((1, 2)).clamp(min=1e-6)).view(-1, 1, 1, 1)


def run_sequence(net, noisy, guides, ref, emission, grad=True):
    """noisy/guides/ref/emission: (T, B, C, H, W). Returns loss and per-frame outputs
    (radiance with the emission added back)."""
    history = prev = None
    loss = 0.0
    prev_out = prev_ref = None
    outs = []
    k = exposure_of(ref) if a.exposure else None
    # The features see the exposure off by up to 2x either way while training.
    seen = None if k is None else (k * 2 ** (2 * torch.rand_like(k) - 1) if grad else k)
    for t in range(noisy.shape[0]):
        y, g, radiance, valid = M.step(net, noisy[t], guides[t], history, prev, t == 0, seen)
        radiance = radiance + emission[t]
        tm, tr = M.tonemap(radiance if k is None else k * radiance), M.tonemap(ref[t] if k is None else k * ref[t])
        frame_loss = (tm - tr).abs().mean()
        if t > max(0, noisy.shape[0] - REFERENCES) and a.temporal_weight > 0:
            motion = guides[t][:, M.MOTION]
            wo, _ = M.warp(prev_out, motion)
            wr, _ = M.warp(prev_ref, motion)
            frame_loss = frame_loss + a.temporal_weight * (((tm - wo) - (tr - wr)).abs() * valid).mean()
        # Frames without a reference only feed the history; the cold first frame
        # counts half.
        loss = loss + frame_loss * (0.0 if t < noisy.shape[0] - REFERENCES else 0.5 if t == 0 else 1.0)
        history, prev = y, g
        if t < noisy.shape[0] - REFERENCES:
            # Warm-up frames carry no loss: gradients start at the first reference.
            history, prev = y.detach(), g.detach()
        prev_out, prev_ref = tm, tr
        outs.append(radiance.detach())
    return loss / min(noisy.shape[0], REFERENCES), outs


def batch():
    noisy, guides, ref, emission = [], [], [], []
    for _ in range(a.batch):
        _, seq = random.choice(train)
        h, w = seq[0].shape[:2]
        y, x = random.randrange(h - a.crop + 1), random.randrange(w - a.crop + 1)
        n, g, r, e = tensors(seq, (y, x, a.crop))
        g = g.clone()
        # Exposure augmentation: scale radiance (and sky "albedo", which is radiance).
        # Not with --exposure: the loss and the features are normalized instead.
        s = 1.0 if a.exposure else math.exp(random.uniform(-1.0, 1.0))
        sky = (g[:, M.KIND : M.KIND + 1] < 0.5).float()
        g[:, M.ALBEDO] = g[:, M.ALBEDO] * (1 + sky * (s - 1))
        noisy.append(n * s), guides.append(g), ref.append(r * s), emission.append(e * s)
    return torch.stack(noisy, 1), torch.stack(guides, 1), torch.stack(ref, 1), torch.stack(emission, 1)


@torch.no_grad()
def evaluate(net, sequences):
    """Mean tonemapped L1 of denoised and raw 1-spp frames against references, in the
    space the loss uses (exposure-normalized with --exposure)."""
    ours = raw = 0.0
    frames = 0
    for _, seq in sequences:
        n, g, r, e = tensors(seq)
        _, outs = run_sequence(net, n.unsqueeze(1), g.unsqueeze(1), r.unsqueeze(1), e.unsqueeze(1), grad=False)
        k = exposure_of(r.unsqueeze(1)) if a.exposure else 1.0
        for t, o in enumerate(outs):
            if t < len(outs) - REFERENCES:
                continue
            ours += (M.tonemap(k * o) - M.tonemap(k * r[t : t + 1])).abs().mean().item()
            raw += (M.tonemap(k * (n[t : t + 1] + e[t : t + 1])) - M.tonemap(k * r[t : t + 1])).abs().mean().item()
            frames += 1
    return ours / max(frames, 1), raw / max(frames, 1)


net = M.Net().to(a.device)
if a.init:
    net.load_state_dict(torch.load(a.init, map_location=a.device))
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
start = time.time()
budget = a.hours * 3600
step = 0
best = float("inf")
log = open(out / "train.log", "a")
checkpoint = out / "checkpoint.pt"
if checkpoint.exists():
    state = torch.load(checkpoint, map_location=a.device)
    net.load_state_dict(state["net"]), opt.load_state_dict(state["opt"])
    step, best = state["step"], state["best"]
    start -= state["elapsed"]
    print(f"resumed at step {step}", flush=True)
running = 0.0
while time.time() - start < budget and (a.max_steps == 0 or step < a.max_steps):
    # The schedule follows the steps when their number is fixed, else the time budget.
    progress = step / a.max_steps if a.max_steps else (time.time() - start) / budget
    for group in opt.param_groups:
        group["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    n, g, r, e = batch()
    loss, _ = run_sequence(net, n, g, r, e)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    running = 0.98 * running + 0.02 * loss.item() if step else loss.item()
    step += 1
    if step % a.eval_every == 0:
        val_ours, val_raw = evaluate(net, val)
        line = f"step {step} t={time.time()-start:.0f}s loss {running:.4f} val {val_ours:.4f} (raw {val_raw:.4f})"
        print(line, flush=True)
        log.write(line + "\n"), log.flush()
        if val_ours < best:
            best = val_ours
            torch.save(net.state_dict(), out / "best.pt")
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, best=best, elapsed=time.time() - start), checkpoint)
if not (out / "best.pt").exists() or evaluate(net, val)[0] < best:
    torch.save(net.state_dict(), out / "best.pt")
net.load_state_dict(torch.load(out / "best.pt", map_location=a.device))
val_ours, val_raw = evaluate(net, val)
test_ours, test_raw = evaluate(net, test)
summary = dict(steps=step, val=val_ours, val_raw=val_raw, test=test_ours, test_raw=test_raw, holdout_scene=a.holdout_scene)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(summary, flush=True)
