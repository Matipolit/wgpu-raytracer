"""Capture temporal-denoiser training sequences for every benchmark scene.

    python3 training/temporal/capture.py OUT [--sequences 24] [--frames 8] [--spp 160] [--size 320x200] [--crop N]

High-quality set used for the final model:
    python3 training/temporal/capture.py training/runs/temporal-hq --sequences 20 --frames 8 \
        --spp 2048 --size 1280x720 --crop 192

Runs `weekend-raytracer-wgpu --capture` per scene (see src/capture.rs) with the
fastest tracer settings. Completed scenes (meta.json present) are skipped, so an
interrupted capture resumes by rerunning the command.
"""
import argparse
import json
import os
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
# Typical feature size per view, in scene units: camera speed and sphere sizes.
SCALES = {
    "sponza-atrium": 1.0,
    "sponza-gallery": 1.0,
    "bistro-street-textured": 1.2,
    "hairball": 0.8,
    "fireplace-room": 0.4,
    "living-room": 0.4,
    "rungholt-aerial": 6.0,
    "chess": 0.04,
}
p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--sequences", type=int, default=24)
p.add_argument("--frames", type=int, default=8)
p.add_argument("--spp", type=int, default=160)
p.add_argument("--size", default="320x200")
p.add_argument("--crop", type=int, default=0, help="trace and store only an N x N window per sequence")
p.add_argument("--scenes", default=",".join(SCALES))
p.add_argument("--extra", nargs="*", default=[], help="extra renderer flags, e.g. --hw-rt")
p.add_argument("--motion-range", help="LO,HI per-sequence motion multiplier (log-uniform)")
p.add_argument("--render-scale", help="S1,S2,...: also capture 1-spp inputs at these render scales (denoise + upscale training)")
p.add_argument("--low-only", action="store_true", help="add the --render-scale inputs to an existing capture (same arguments and seeds)")
p.add_argument("--seed-base", type=int, default=1000, help="use a different base for evaluation sets")
a = p.parse_args()
scenes = json.loads((ROOT / "performance/scenes/scenes.json").read_text())
env = dict(os.environ, RT_QUAD_SLM="32", RT_QUAD_LEAF_COUNT="1")
for index, name in enumerate(a.scenes.split(",")):
    out = pathlib.Path(a.out) / name
    if (out / ("low_meta.json" if a.low_only else "meta.json")).exists():
        print(f"{name}: done, skipping", flush=True)
        continue
    start = time.time()
    subprocess.run(
        [str(ROOT / "target/release/weekend-raytracer-wgpu"), "--model", scenes[name]["model"],
         "--camera", scenes[name]["camera"], "--capture", str(out), "--scale", str(SCALES[name]),
         "--sequences", str(a.sequences), "--frames", str(a.frames), "--reference-spp", str(a.spp),
         "--size", a.size, "--seed", str(a.seed_base + index), *(["--crop", str(a.crop)] if a.crop else []),
         *(["--motion-range", a.motion_range] if a.motion_range else []),
         *(["--upscale-capture", a.render_scale] if a.render_scale else []),
         *(["--low-only"] if a.low_only else []),
         "--sah-stop", "--skip-final-scatter", "--quad-bvh", "--sbvh", *a.extra],
        cwd=ROOT, env=env, check=True,
    )
    print(f"{name}: {time.time() - start:.0f}s", flush=True)
