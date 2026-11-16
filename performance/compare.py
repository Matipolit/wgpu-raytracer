#!/usr/bin/env python3
"""Alternate two release executables and compare their offscreen benchmarks."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("performance/results"))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {"before": [], "after": []}
    for round_index in range(args.rounds):
        order = ["before", "after"] if round_index % 2 == 0 else ["after", "before"]
        for label in order:
            executable = getattr(args, label).resolve()
            result = subprocess.run(
                [str(executable), "--benchmark"],
                text=True, capture_output=True, check=True, timeout=120,
                cwd=Path(__file__).resolve().parents[1],
            )
            data = json.loads(result.stdout)
            runs[label].append(data)
            path = args.output / f"{label}-{round_index + 1}.json"
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"Completed {label} round {round_index + 1}", flush=True)
    before = runs["before"][0]
    for group in runs.values():
        for run in group:
            for field in ("adapter", "backend", "driver", "warmup_frames", "measured_frames"):
                if run[field] != before[field]:
                    raise ValueError(f"Benchmark configuration mismatch: {field}")
            if [(c["width"], c["height"], c["state"]) for c in run["cases"]] != [
                (c["width"], c["height"], c["state"]) for c in before["cases"]
            ]:
                raise ValueError("Benchmark scenarios do not match")
    summary = {
        "adapter": before["adapter"], "backend": before["backend"],
        "driver": before["driver"], "rounds": args.rounds,
        "warmup_frames": before["warmup_frames"],
        "measured_frames": before["measured_frames"],
        "method": "median of per-run medians; per-frame GPU completion; no vsync",
        "before_ui": before["ui"], "after_ui": runs["after"][0]["ui"],
        "before_renderer": before.get("renderer", "fragment"),
        "after_renderer": runs["after"][0].get("renderer", "fragment"), "cases": [],
    }
    for index, case in enumerate(before["cases"]):
        entry = {key: case[key] for key in ("width", "height", "state")}
        for metric in ("gpu_ms", "cpu_ms", "ui_cpu_ms", "wall_ms"):
            old = statistics.median(run["cases"][index][metric]["median"] for run in runs["before"])
            new = statistics.median(run["cases"][index][metric]["median"] for run in runs["after"])
            entry[metric] = {"before": old, "after": new, "change_percent": 100 * (new / old - 1)}
        summary["cases"].append(entry)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
