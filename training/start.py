#!/usr/bin/env python3
"""Start an overnight job detached from the invoking terminal/agent."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
runs = root / 'training/runs'
runs.mkdir(parents=True, exist_ok=True)
with (runs / 'overnight.log').open('a') as log:
    process = subprocess.Popen(
        [str(root / 'training/run.sh'), *sys.argv[1:]], cwd=root,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
(runs / 'overnight.pid').write_text(f'{process.pid}\n')
print(f'Started PID {process.pid}; log: {runs / "overnight.log"}')
