"""Side-by-side video (noisy 1 spp | renderer's temporal denoiser output) from a
`--capture ... --no-reference --temporal-model MODEL.wgsl` run.

    training/.venv/bin/python training/temporal/make_video.py SEQ_DIR WIDTH HEIGHT OUT.mp4 [--fps 15]
"""
import argparse
import pathlib
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("seq")
p.add_argument("width", type=int)
p.add_argument("height", type=int)
p.add_argument("out")
p.add_argument("--fps", type=int, default=15)
a = p.parse_args()
w, h = a.width, a.height
frames = sorted(pathlib.Path(a.seq).glob("frame_*.f16"))
# One exposure for the whole clip, from the denoised log-average luminance.
lum = []
for f in frames:
    den = np.fromfile(str(f).replace(".f16", ".den.f32"), "<f4").reshape(h, w, 3)
    lum.append(np.exp(np.mean(np.log(1e-4 + den @ [0.2126, 0.7152, 0.0722]))))
exposure = 0.18 / float(np.median(lum)) * 2


def tonemap(x):
    def curve(v):
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return (v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F) - E / F
    v = np.clip(curve(exposure * np.clip(x, 0, None)) / curve(11.2), 0, 1)
    return (np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)


with tempfile.TemporaryDirectory() as tmp:
    for i, f in enumerate(frames):
        noisy = np.fromfile(f, "<f2").reshape(h, w, 18)[..., 0:3].astype(np.float32)
        den = np.fromfile(str(f).replace(".f16", ".den.f32"), "<f4").reshape(h, w, 3)
        img = Image.fromarray(np.concatenate([tonemap(noisy), tonemap(den)], 1))
        d = ImageDraw.Draw(img)
        d.text((8, 8), "1 spp input", fill=(255, 255, 0))
        d.text((w + 8, 8), "temporal neural denoiser", fill=(255, 255, 0))
        img.save(f"{tmp}/{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], check=True)
print(f"wrote {a.out} ({len(frames)} frames)")
