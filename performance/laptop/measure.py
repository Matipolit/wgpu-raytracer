#!/usr/bin/env python3
"""Sequential fixed-work offline benchmarks; GPU timing excludes readback and setup."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--binary', type=Path, default=ROOT/'target/release/weekend-raytracer-wgpu')
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--samples', type=int, default=24)
ap.add_argument('--sizes', nargs='+', default=['640x400'])
ap.add_argument('--repeats', type=int, default=3)
ap.add_argument('--cases', nargs='+', default=['sponza','sponza-reverse','spheres'])
ap.add_argument('--workgroups', nargs='+', default=['8x8'])
args = ap.parse_args()
cases = [('sponza', ['--sponza']), ('sponza-reverse', ['--sponza','--camera','7,1.6,0,-1,0.08,0.08,65']), ('spheres', [])]
results=[]
for group in args.workgroups:
 for size in args.sizes:
  for name, scene in cases:
   if name not in args.cases: continue
   times=[]
   for repeat in range(args.repeats):
    command=[str(args.binary.resolve()),*scene,'--no-russian-roulette','--bounces','8','--samples',str(args.samples),'--size',size,'--render-output','/tmp/laptop-benchmark.png']
    r=subprocess.run(command,cwd=ROOT,env={**os.environ,'RAYTRACER_WORKGROUP':group},capture_output=True,text=True,check=True)
    match=re.search(r'median GPU Some\(([^)]+)\)',r.stderr)
    if not match: raise RuntimeError(r.stderr)
    times.append(float(match[1])); print(group,size,name,repeat,times[-1],flush=True)
   results.append(dict(case=name,size=size,workgroup=group,samples=args.samples,median_ms=statistics.median(times),runs_ms=times))
   args.output.write_text(json.dumps(results,indent=2)+'\n')
