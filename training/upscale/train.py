"""Train the joint denoise + upscale network (model.py) on multi-scale captures.

    python training/upscale/train.py --data DIR --out RUN [--hours 4] [--channels 16]

Each batch uses one render scale (chosen at random), random output crops and the whole
sequence (recurrent). Validation reports the mean tonemapped L1 per scale on each scene's
last two sequences; --holdout-scene is never trained on (test).
"""
import argparse
import json
import math
import pathlib
import random
import resource
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import data as D  # noqa: E402
import model as M  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--hours", type=float, default=4.0)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--batch", type=int, default=6)
p.add_argument("--crop", type=int, default=160)
p.add_argument("--frames", type=int, default=0, help="frames per training sequence (0 = all)")
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--holdout-scene", default="fireplace-room")
p.add_argument("--temporal-weight", type=float, default=0.5)
p.add_argument("--eval-every", type=int, default=300)
p.add_argument("--tier", choices=list(M.TIERS), help="architecture preset (overrides --channels/--layers/--full-res)")
p.add_argument("--channels", type=int, default=16)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--full-res", action="store_true", help="3x3 convs at output resolution (default: half)")
p.add_argument("--dilations", default="1,2,4")
p.add_argument("--init", help="start from these weights (same architecture)")
p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke tests)")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--amp", action="store_true", help="bf16 mixed precision (tensor cores on CUDA)")
a = p.parse_args()
torch.set_num_threads(a.threads)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
torch.manual_seed(0)
random.seed(0)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tier = M.TIERS[a.tier] if a.tier else dict(channels=a.channels, layers=a.layers, half=not a.full_res)
M.configure(tier["channels"], [int(d) for d in a.dilations.split(",")], tier["layers"], tier["half"])
(out / "config.json").write_text(json.dumps(dict(tier=a.tier, channels=M.CHANNELS, layers=M.LAYERS, half=M.HALF, dilations=list(M.DILATIONS))))

scenes = D.load(a.data)
train, val, test = D.split(scenes, a.holdout_scene)
scales = len(train[0].lows)
print(f"train {len(train)} seqs, val {len(val)}, test ({a.holdout_scene}) {len(test)}, {scales} scales "
      f"{[s['scale'] for s in train[0].lows]}, device {a.device}", flush=True)


def run(net, frames, coords, origin, scale, grad=True):
    """Loss over a sequence; also returns the per-frame outputs."""
    history = prev = None
    loss = 0.0
    prev_out = prev_ref = None
    outs = []
    for t, (low, guides, ref, jitter) in enumerate(frames):
        y, g, radiance, valid = M.step(net, low, guides, coords, origin, jitter, scale, history, prev, t == 0)
        tm, tr = M.tonemap(radiance), M.tonemap(ref)
        frame_loss = (tm - tr).abs().mean()
        if t > 0 and a.temporal_weight > 0:
            motion = guides[:, M.MOTION]
            wo, _ = M.warp(prev_out, motion)
            wr, _ = M.warp(prev_ref, motion)
            frame_loss = frame_loss + a.temporal_weight * (((tm - wo) - (tr - wr)).abs() * valid).mean()
        loss = loss + frame_loss * (0.5 if t == 0 else 1.0)
        history, prev = y, g
        prev_out, prev_ref = tm, tr
        outs.append(radiance.detach())
    return loss / len(frames), outs


# Batches stack sequences of one stored size (datasets may use different crops).
groups = {}
for seq in train:
    key = (seq.size, tuple(l["frames"][0].shape for l in seq.lows))
    groups.setdefault(key, []).append(seq)
groups = list(groups.values())


def batch():
    k = random.randrange(scales)
    group = random.choices(groups, weights=[len(g) for g in groups])[0]
    seqs = random.sample(group, min(a.batch, len(group)))
    h, w = seqs[0].size
    c = min(a.crop, h, w)
    window = (random.randrange(h - c + 1), random.randrange(w - c + 1), c)
    frames, coords, origin, scale = D.tensors(seqs, k, window, a.device, a.frames or None)
    # Exposure augmentation: radiance, references and sky "albedo" (which holds radiance).
    s = torch.exp(torch.empty(len(seqs), 1, 1, 1, device=a.device).uniform_(-1.0, 1.0))
    augmented = []
    for low, guides, ref, jitter in frames:
        low, guides = low.clone(), guides.clone()
        low_sky = (low[:, 14:15] < 0.5).float()
        sky = (guides[:, M.KIND : M.KIND + 1] < 0.5).float()
        low[:, 0:3] *= s
        low[:, 3:6] *= 1 + low_sky * (s - 1)
        guides[:, M.ALBEDO] *= 1 + sky * (s - 1)
        augmented.append((low, guides, ref * s, jitter))
    return augmented, coords, origin, scale


@torch.no_grad()
def evaluate(net, sequences):
    """Mean tonemapped L1 per scale of the network and of the raw splat."""
    result = []
    for k in range(scales):
        ours = raw = 0.0
        count = 0
        for seq in sequences:
            frames, coords, origin, scale = D.tensors([seq], k, None, a.device)
            _, outs = run(net, frames, coords, origin, scale, grad=False)
            for (low, _, ref, jitter), o in zip(frames, outs):
                tr = M.tonemap(ref)
                ours += (M.tonemap(o) - tr).abs().mean().item()
                raw += (M.tonemap(M.splat(low, coords, origin, jitter, scale)) - tr).abs().mean().item()
                count += 1
        result.append((ours / max(count, 1), raw / max(count, 1)))
    return result


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
    net.load_state_dict(state["net"])
    opt.load_state_dict(state["opt"])
    step, best = state["step"], state["best"]
    start -= state["elapsed"]
    print(f"resumed at step {step}", flush=True)
running = 0.0
while time.time() - start < budget and (not a.max_steps or step < a.max_steps):
    progress = (time.time() - start) / budget
    for group in opt.param_groups:
        group["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    frames, coords, origin, scale = batch()
    # bf16 autocast runs the convolutions on tensor cores; weights and the optimizer
    # stay f32 (no loss scaling needed for bf16).
    with torch.autocast(device_type=a.device.split(":")[0], dtype=torch.bfloat16, enabled=a.amp):
        loss, _ = run(net, frames, coords, origin, scale)
    loss = loss.float()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    running = 0.98 * running + 0.02 * loss.item() if step else loss.item()
    step += 1
    if step % a.eval_every == 0:
        result = evaluate(net, val)
        mean = sum(r[0] for r in result) / len(result)
        line = f"step {step} t={time.time() - start:.0f}s loss {running:.4f} val {mean:.4f} per scale " + " ".join(
            f"{ours:.4f}(raw {raw:.4f})" for ours, raw in result)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if mean < best:
            best = mean
            torch.save(net.state_dict(), out / "best.pt")
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, best=best, elapsed=time.time() - start), checkpoint)
if not (out / "best.pt").exists():
    torch.save(net.state_dict(), out / "best.pt")
net.load_state_dict(torch.load(out / "best.pt", map_location=a.device))
summary = dict(steps=step, val=evaluate(net, val), test=evaluate(net, test) if test else None, holdout_scene=a.holdout_scene,
               scales=[s["scale"] for s in train[0].lows])
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(summary, flush=True)
