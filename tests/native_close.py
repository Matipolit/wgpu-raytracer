#!/usr/bin/env python3
"""Requires a graphical session; tests normal window-close teardown in a subprocess."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, default=Path("target/release/weekend-raytracer-wgpu"))
args = parser.parse_args()
for attempt in range(3):
    result = subprocess.run(
        [str(args.binary.resolve()), "--smoke-test-close"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode != 0:
        raise SystemExit(f"Close attempt {attempt + 1} failed ({result.returncode}):\n{result.stderr}")
print("Native window-close teardown passed three consecutive runs.")
