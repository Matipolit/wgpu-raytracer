#!/usr/bin/env python3
"""Compare NDF and VNDF at three bounces with independent 2048-spp references."""
from pathlib import Path
import argparse
import json
import math
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'performance/vndf/results'
BIN=ROOT/'target/release/weekend-raytracer-wgpu'

def capture(e,label,spp,seed,visible):
 stem=OUT/label
 command=[str(BIN),'--bounces','3','--sponza','--no-denoise','--size',f'{e["width"]}x{e["height"]}','--camera',e['camera'],'--exposure',str(e['exposure']),'--samples',str(spp),'--sample-offset',str(seed),'--render-output',str(stem.with_suffix('.png')),'--raw-output',str(stem.with_suffix('.f32')),'--timing-output',str(stem.with_suffix('.json'))]
 command+=['--russian-roulette']
 if visible: command+=['--ggx-vndf']
 print('Rendering',label,spp,'spp',flush=True)
 subprocess.run(command,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 timing=json.loads(stem.with_suffix('.json').read_text())
 return dict(label=label,spp=spp,seed=seed,vndf=visible,**timing)

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

 results=[]
 for idx in a.views:
  e=manifest[idx].copy()
  label=f'{idx:03d}_reference'
  raw=OUT/(label+'.f32');png=OUT/(label+'.png');timing=OUT/(label+'.json')
  if not (raw.exists() and raw.stat().st_size==e['width']*e['height']*12 and png.exists() and timing.exists()):
   capture(e,label,2048,2000000,False)
  reference=np.fromfile(raw,dtype='<f4')
  e['high']=str(raw)
  pilot=capture(e,f'{idx:03d}_pilot',24,90000,True)
  runs=[]
  for seed in [10000,20000,30000]:
   base=metrics(capture(e,f'{idx:03d}_{seed}_baseline',a.samples,seed,False),e,reference)
   matched_spp=max(2,1+round(base['gpu_total_ms']/(pilot['gpu_total_ms']/pilot['warmed_samples'])))
   visible=metrics(capture(e,f'{idx:03d}_{seed}_vndf',matched_spp,seed,True),e,reference)
   # Correct pilot prediction if clocks or divergence changed the budget >5%.
   if abs(visible['gpu_total_ms']/base['gpu_total_ms']-1)>.05:
    matched_spp=max(2,1+round(base['gpu_total_ms']/(visible['gpu_total_ms']/visible['warmed_samples'])))
    visible=metrics(capture(e,f'{idx:03d}_{seed}_vndf',matched_spp,seed,True),e,reference)
   runs.append(dict(baseline=base,vndf=visible,budget_ratio=visible['gpu_total_ms']/base['gpu_total_ms'],psnr_gain_db=visible['display_psnr_db']-base['display_psnr_db'],per_sample_speedup=(base['gpu_total_ms']/base['warmed_samples'])/(visible['gpu_total_ms']/visible['warmed_samples'])))
   print(json.dumps(runs[-1]),flush=True)
  results.append(dict(view=idx,reference=e['high'],reference_png=str(Path(e['high']).with_suffix('.png')),width=e['width'],height=e['height'],reference_spp=2048,reference_bounces=3,reference_russian_roulette=True,runs=runs))
  (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=="__main__": main()
