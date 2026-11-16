#!/usr/bin/env python3
"""Interleave baseline and candidate runs to reduce thermal/order bias."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser()
ap.add_argument('baseline',type=Path)
ap.add_argument('candidate',type=Path)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--cases',nargs='+',default=['sponza','sponza-reverse','spheres'])
ap.add_argument('--size',default='640x400')
ap.add_argument('--samples',type=int,default=24)
ap.add_argument('--repeats',type=int,default=2)
a=ap.parse_args()
records=[]
for name,scene in [('sponza',['--sponza']),('sponza-reverse',['--sponza','--camera','7,1.6,0,-1,0.08,0.08,65']),('spheres',[])]:
 if name not in a.cases: continue
 runs={'baseline':[],'candidate':[]}
 for repeat in range(a.repeats):
  order=[('baseline',a.baseline),('candidate',a.candidate)]
  if repeat%2: order.reverse()
  for label,binary in order:
   stem=Path('/tmp')/f'laptop-{name}-{label}'
   result=subprocess.run([str(binary.resolve()),*scene,'--no-russian-roulette','--bounces','8','--samples',str(a.samples),'--size',a.size,'--render-output',str(stem.with_suffix('.png')),'--raw-output',str(stem.with_suffix('.f32'))],cwd=ROOT,env={**os.environ,"RAYTRACER_WORKGROUP":"8x8"},capture_output=True,text=True,check=True)
   ms=float(re.search(r'median GPU Some\(([^)]+)\)',result.stderr)[1])
   runs[label].append(ms);print(name,label,repeat,ms,flush=True)
 before=statistics.median(runs['baseline']);after=statistics.median(runs['candidate'])
 equal=(Path('/tmp')/f'laptop-{name}-baseline.f32').read_bytes()==(Path('/tmp')/f'laptop-{name}-candidate.f32').read_bytes()
 records.append(dict(case=name,size=a.size,samples=a.samples,baseline_ms=before,candidate_ms=after,speedup=before/after,raw_identical=equal,runs=runs))
 a.output.write_text(json.dumps(records,indent=2)+'\n')
 print(name,'speedup',before/after,'raw identical',equal,flush=True)
 if not equal: raise SystemExit('Raw image mismatch; inspect before accepting')
