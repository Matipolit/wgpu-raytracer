#!/usr/bin/env python3
"""Compare RR-only against RR + sky next-event estimation at matched GPU budgets."""
from pathlib import Path
import argparse
import json
import math
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'performance/sky/results'
BIN=ROOT/'target/release/weekend-raytracer-wgpu'

def capture(e,label,spp,seed,importance):
 stem=OUT/label
 command=[str(BIN),'--bounces','8','--sponza','--no-denoise','--size',f'{e["width"]}x{e["height"]}','--camera',e['camera'],'--exposure',str(e['exposure']),'--samples',str(spp),'--sample-offset',str(seed),'--render-output',str(stem.with_suffix('.png')),'--raw-output',str(stem.with_suffix('.f32')),'--timing-output',str(stem.with_suffix('.json'))]
 command+=['--russian-roulette']
 if importance: command+=['--sky-importance']
 print('Rendering',label,spp,'spp',flush=True)
 subprocess.run(command,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 timing=json.loads(stem.with_suffix('.json').read_text())
 return dict(label=label,spp=spp,seed=seed,sky_importance=importance,**timing)

def display(x,exposure):
 def tone(z):return ((z*(.15*z+.1*.5)+.2*.02)/(z*(.15*z+.5)+.2*.3))-.02/.3
 x=np.clip(tone(np.maximum(x,0)*exposure)/tone(11.2),0,1)
 return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)

def metrics(r,e,reference):
 raw=np.fromfile(OUT/(r['label']+'.f32'),dtype='<f4')
 assert np.isfinite(raw).all()
 mse=float(np.mean((display(raw,e['exposure'])-display(reference,e['exposure']))**2,dtype=np.float64))
 r['display_mse']=mse;r['display_psnr_db']=-10*math.log10(max(mse,1e-16))
 r['compressed_radiance_mse']=float(np.mean((raw/(1+raw)-reference/(1+reference))**2,dtype=np.float64))
 return r


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=128);ap.add_argument('--views',type=int,nargs='+',default=[1,19]);a=ap.parse_args()
 OUT.mkdir(exist_ok=True)
 manifest=json.loads((ROOT/'training/runs/overnight/manifest.json').read_text())
 reference_manifest=ROOT/'performance/references/2048-rr/manifest.json'
 references={r['view']:r for r in json.loads(reference_manifest.read_text())} if reference_manifest.exists() else {}
 
 results=[]
 for idx in a.views:
  e=manifest[idx].copy();ref=references.get(idx)
  if ref: e['high']=ref['raw']
  reference=np.fromfile(e['high'],dtype='<f4')
  pilot=capture(e,f'{idx:03d}_pilot',24,90000,True)
  runs=[]
  for seed in [10000,20000]:
   base=metrics(capture(e,f'{idx:03d}_{seed}_baseline',a.samples,seed,False),e,reference)
   matched_spp=max(2,1+round(base['gpu_total_ms']/(pilot['gpu_total_ms']/pilot['warmed_samples'])))
   sky=metrics(capture(e,f'{idx:03d}_{seed}_sky',matched_spp,seed,True),e,reference)
   # Correct pilot prediction if clocks or divergence changed the budget >5%.
   if abs(sky['gpu_total_ms']/base['gpu_total_ms']-1)>.05:
    matched_spp=max(2,1+round(base['gpu_total_ms']/(sky['gpu_total_ms']/sky['warmed_samples'])))
    sky=metrics(capture(e,f'{idx:03d}_{seed}_sky',matched_spp,seed,True),e,reference)
   runs.append(dict(baseline=base,sky=sky,budget_ratio=sky['gpu_total_ms']/base['gpu_total_ms'],psnr_gain_db=sky['display_psnr_db']-base['display_psnr_db'],per_sample_speedup=(base['gpu_total_ms']/base['warmed_samples'])/(sky['gpu_total_ms']/sky['warmed_samples'])))
   print(json.dumps(runs[-1]),flush=True)
  results.append(dict(view=idx,reference=e['high'],reference_png=str(Path(e['high']).with_suffix('.png')),width=e['width'],height=e['height'],reference_spp=ref['samples'] if ref else 512,reference_bounces=8,reference_russian_roulette=bool(ref),runs=runs))
  (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=="__main__": main()
