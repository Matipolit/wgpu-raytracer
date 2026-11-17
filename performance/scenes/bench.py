"""Interleaved multi-scene GPU timing and raw-output comparison.

    python3 performance/scenes/bench.py --out DIR \
        --config base=BIN:--sah-stop,--skip-final-scatter \
        --config cand=BIN2:--sah-stop,--skip-final-scatter [--reps 3] [--samples 32]

The first config is the reference. Each repetition visits every scene and runs
every config on it, rotating config order, so clock/thermal drift is shared.
Pixels whose tone-mapped value changes by more than 2/255 are reported as
a percentage ("=" means bit-identical raw radiance). A run's time is the median warmed GPU frame time (trace + filter + display);
the reported value per scene/config is the median over repetitions. Raw float
radiance of every config is compared with the reference at the same seed.
"""
import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys

import numpy as np


def display(raw):
    """Tone-mapped sRGB in [0, 1], matching the renderer at exposure 2."""
    def curve(x):
        a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f
    linear = np.clip(curve(2.0 * raw) / curve(11.2), 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCENES = json.loads((ROOT / "performance/scenes/scenes.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--config", action="append", required=True, help="name=binary:flag,flag")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--size", default="640x400")
    p.add_argument("--scenes", default=",".join(SCENES))
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = []
    for c in a.config:
        name, rest = c.split("=", 1)
        binary, _, flags = rest.partition(":")
        configs.append((name, binary, [f for f in flags.split(",") if f]))
    scenes = a.scenes.split(",")
    times = {(s, c[0]): [] for s in scenes for c in configs}
    for rep in range(a.reps):
        for si, scene in enumerate(scenes):
            order = configs[(rep + si) % len(configs):] + configs[:(rep + si) % len(configs)]
            for name, binary, flags in order:
                e = SCENES[scene]
                prefix = out / f"{scene}.{name}"
                cmd = [binary, "--model", str(ROOT / e["model"]), "--camera", e["camera"],
                       "--samples", str(a.samples), "--size", a.size, "--no-denoise",
                       "--exposure", "2", "--render-output", f"{prefix}.png",
                       "--timing-output", f"{prefix}.json", *flags]
                if rep == 0:
                    cmd += ["--raw-output", f"{prefix}.f32"]
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if r.returncode:
                    sys.exit(f"{scene} {name} failed:\n{r.stderr[-2000:]}")
                t = json.loads(pathlib.Path(f"{prefix}.json").read_text())
                times[(scene, name)].append(t["gpu_median_ms"])
                print(f"rep {rep} {scene:16} {name:12} {t['gpu_median_ms']:8.2f} ms", flush=True)
    ref = configs[0][0]
    rows = []
    for scene in scenes:
        base_raw = np.fromfile(out / f"{scene}.{ref}.f32", dtype="<f4")
        for name, _, _ in configs:
            raw = np.fromfile(out / f"{scene}.{name}.f32", dtype="<f4")
            diff = np.abs(raw.astype(np.float64) - base_raw)
            shown = np.abs(display(raw.astype(np.float64)) - display(base_raw.astype(np.float64))).reshape(-1, 3).max(1)
            rows.append(dict(
                scene=scene, config=name,
                ms=statistics.median(times[(scene, name)]),
                runs=times[(scene, name)],
                speedup=statistics.median(times[(scene, ref)]) / statistics.median(times[(scene, name)]),
                identical=bool(np.array_equal(raw, base_raw)),
                differing_fraction=float(np.mean(diff > 0)),
                relative_rmse=float(np.sqrt(np.mean(diff ** 2)) / max(np.mean(np.abs(base_raw)), 1e-12)),
                nan=int(np.isnan(raw).sum()),
                visible_fraction=float(np.mean(shown > 2 / 255)),
                display_mean_abs=float(np.mean(shown)),
            ))
    summary = {}
    for name, _, flags in configs:
        sp = [r["speedup"] for r in rows if r["config"] == name]
        summary[name] = dict(flags=flags, geomean_speedup=math.exp(sum(map(math.log, sp)) / len(sp)),
                             total_ms=sum(r["ms"] for r in rows if r["config"] == name))
    (out / "results.json").write_text(json.dumps(dict(args=vars(a), rows=rows, summary=summary), indent=1))
    print(f"\n{'scene':16} " + " ".join(f"{c[0]:>22}" for c in configs))
    for scene in scenes:
        cells = []
        for name, _, _ in configs:
            r = next(r for r in rows if r["scene"] == scene and r["config"] == name)
            mark = "=" if r["identical"] else f"{100 * r['visible_fraction']:.2f}%"
            cells.append(f"{r['ms']:7.2f} {r['speedup']:5.3f}x {mark:>7}")
        print(f"{scene:16} " + " ".join(f"{c:>22}" for c in cells))
    print("geomean          " + " ".join(f"{summary[c[0]]['geomean_speedup']:>21.3f}x" for c in configs))


if __name__ == "__main__":
    main()
