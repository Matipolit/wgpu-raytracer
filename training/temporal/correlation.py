"""How much 1-spp frames repeat their error from frame to frame (still camera).

    weekend-raytracer-wgpu SCENE --size 480x270 --restir --bounces 2 --no-denoise \
        --no-auto-tune --center-jitter --render-output /tmp/x.png --samples 312 \
        --raw-frames FRAMES.raw --raw-warmup 56 --raw-every 1
    python correlation.py FRAMES.raw 480 270 [LABEL] [REFERENCE.raw]

--center-jitter and --no-auto-tune matter: with a shared sub-pixel jitter (upscaling, or
the visibility raster the tuner may pick) texture detail alternates with the jitter's
period and reads as correlation. REFERENCE.raw: `--raw-output` of a many-sample render
with the same camera and flags, without --restir.

Per pixel and per 8x8 block, in log(1 + exposure * luminance): the noise ("sd"), its
correlation with the frame 1, 4, 16 and 32 frames later ("acf"; 0 for independent
frames), and what is left of it in the mean of 16 and of 32 consecutive frames
("m16", "m32"; sd / 4 and sd / 5.7 for independent frames). A temporal filter can only
remove what these means remove. With a reference: overall brightness of the mean frame
against it, and the difference of 16x16 block means ("blockerr").
"""
import os
import sys

import numpy as np

path, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else path
Y = np.array([0.2126, 0.7152, 0.0722], np.float32)
lum = np.fromfile(path, np.float32).reshape(-1, h, w, 3) @ Y
k = 0.18 / max(lum.mean(), 1e-8)
T = lum.shape[0]


def stats(v):
    v = np.log1p(k * v)
    d = v - v.mean(0)
    var = (d * d).mean() * T / (T - 1)
    acf = [float((d[lag:] * d[:-lag]).mean() / var + 1 / T) for lag in (1, 4, 16, 32)]
    left = []
    for n in (16, 32):
        m = d[: T // n * n].reshape(T // n, n, *d.shape[1:]).mean(1)
        left.append(float(np.sqrt((m * m).mean() * (T // n) / (T // n - 1))))
    return np.sqrt(var), acf, left


def blocks(v, size=8):
    return v[..., : h // size * size, : w // size * size].reshape(*v.shape[:-2], h // size, size, w // size, size).mean((-3, -1))


out = f"{label:26s}"
for name, v in (("px", lum), ("8x8", blocks(lum))):
    sd, acf, left = stats(v)
    out += f" | {name} sd {sd:.4f} acf1,4,16,32 " + " ".join(f"{c:+.2f}" for c in acf) + f" m16 {left[0]:.4f} m32 {left[1]:.4f}"
if len(sys.argv) > 5 and os.path.exists(sys.argv[5]):
    ref = np.fromfile(sys.argv[5], np.float32).reshape(h, w, 3) @ Y
    mean = lum.mean(0)
    kr = 0.18 / ref.mean()
    error = np.log1p(kr * blocks(mean, 16)) - np.log1p(kr * blocks(ref, 16))
    out += f" | bright {mean.mean() / ref.mean():.3f} blockerr {np.sqrt((error ** 2).mean()):.4f}"
print(out)
