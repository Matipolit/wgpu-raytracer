#!/usr/bin/env python3
"""Standalone reproducible local-optimization sweep. No agent or API calls."""
from pathlib import Path
import argparse, bisect, hashlib, json, math, platform, re, statistics, subprocess, time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'performance/local-sweep/results';OUT.mkdir(exist_ok=True)
BIN=ROOT/'target/release/weekend-raytracer-wgpu'
SHA=hashlib.sha256(BIN.read_bytes()).hexdigest()
MANIFEST=json.loads((ROOT/'training/runs/overnight/manifest.json').read_text())
VIEWS=[1,19]
VARIANTS=[
 ('bvh4',['--bvh4'],'geometry'),('bvh8',['--bvh8'],'geometry'),('quantized_bvh8',['--quantized-bvh'],'geometry'),
 ('leaf8',['--bvh-leaf','8'],'geometry'),('leaf16',['--bvh-leaf','16'],'geometry'),('sah32',['--sah32'],'geometry'),('sah_stop',['--sah-stop'],'geometry'),
 ('packed_shading',['--packed-shading'],'precision'),('stochastic_color',['--stochastic-color'],'sampling'),('stochastic_all',['--stochastic-all'],'approximation'),
 ('fp16_state',['--fp16-state'],'precision'),('packed_directions',['--packed-directions'],'precision'),
 ('skip_final_shading',['--skip-final-scatter'],'exact'),('classify_opacity',['--classify-opacity'],'exact'),
 ('opacity_micromap',['--opacity-micromap','--scene-cache'],'geometry'),
 ('alpha_shrink',['--approximate-alpha'],'approximation'),('alpha_cap4',['--alpha-cap4'],'approximation'),
 ('blue_noise_jitter',['--blue-noise-jitter'],'sampling'),('bounded_vndf',['--bounded-vndf'],'sampling'),('ray_cones',['--ray-cones','--scene-cache'],'filtering')]

def save(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def load(name,default):
 p=OUT/name
 return json.loads(p.read_text()) if p.exists() else default

def display(x,exposure):
 def tone(z):return (z*(.15*z+.05)+.004)/(z*(.15*z+.5)+.06)-.02/.3
 x=np.clip(tone(np.maximum(x,0)*exposure)/tone(11.2),0,1)
 return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)
REFERENCES={v:np.fromfile(ROOT/f'performance/vndf/results/{v:03d}_reference.f32',dtype='<f4') for v in VIEWS}
for v in VIEWS:assert len(REFERENCES[v])==MANIFEST[v]['width']*MANIFEST[v]['height']*3

def capture(label,view,spp,seed,flags=(),size='640x400',denoise=False,filter_report=False,binary=BIN):
 e=MANIFEST[view];prefix=OUT/label;cached=prefix.with_suffix('.record.json')
 binary=Path(binary);digest=SHA if binary==BIN else hashlib.sha256(binary.read_bytes()).hexdigest()
 key=dict(binary_sha256=digest,view=view,spp=spp,seed=seed,flags=list(flags),size=size,denoise=denoise,filter_report=filter_report)
 if cached.exists():
  r=json.loads(cached.read_text())
  if r.get('key')==key and prefix.with_suffix('.f32').exists() and prefix.with_suffix('.png').exists():return r
 command=[str(binary),'--sponza','--bounces','3','--russian-roulette','--samples',str(spp),'--sample-offset',str(seed),'--size',size,'--camera',e['camera'],'--exposure',str(e['exposure']),'--render-output',str(prefix.with_suffix('.png')),'--raw-output',str(prefix.with_suffix('.f32')),'--timing-output',str(prefix.with_suffix('.json')),*flags]
 if not denoise:command+=['--no-denoise']
 if filter_report:command+=['--filter-report',str(prefix.with_suffix('.filter.json'))]
 print('RENDER',label,spp,'spp',list(flags),flush=True)
 started=time.time();p=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
 prefix.with_suffix('.log').write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError(f'{label} failed: {p.stderr[-3000:]}')
 timing=json.loads(prefix.with_suffix('.json').read_text());raw=np.fromfile(prefix.with_suffix('.f32'),dtype='<f4')
 if not np.isfinite(raw).all():raise RuntimeError(f'nonfinite radiance: {label}')
 r=dict(key=key,label=label,view=view,spp=spp,seed=seed,flags=list(flags),started=started,midpoint=(time.time()+started)/2,process_wall_s=time.time()-started,**timing)
 r['ms_per_sample']=r['gpu_total_ms']/r['warmed_samples']
 if size=='640x400':
  ref=REFERENCES[view];mse=float(np.mean((display(raw,e['exposure'])-display(ref,e['exposure']))**2,dtype=np.float64))
  r.update(display_mse=mse,psnr_db=-10*math.log10(max(mse,1e-20)),radiance_mse=float(np.mean((raw/(1+raw)-ref/(1+ref))**2,dtype=np.float64)))
  if denoise:
   rgb=np.asarray(Image.open(prefix.with_suffix('.png')).convert('RGB'),dtype=np.float32).reshape(-1)/255
   mse=float(np.mean((rgb-display(ref,e['exposure']))**2,dtype=np.float64));r.update(filtered_display_mse=mse)
 if filter_report:r['filter_p95_ms']=json.loads(prefix.with_suffix('.filter.json').read_text())['filter_p95_ms']
 save(cached.name,r);return r

def equal_raw(a,b):return (OUT/(a['label']+'.f32')).read_bytes()==(OUT/(b['label']+'.f32')).read_bytes()

def screen():
 records=[];controls=[]
 for block in range((len(VARIANTS)+3)//4+1):
  for v in VIEWS:controls.append(capture(f's_control_{block}_{v}',v,32,10000))
  for index in range(block*4,min(block*4+4,len(VARIANTS))):
   name,flags,kind=VARIANTS[index]
   for v in VIEWS if index%2==0 else VIEWS[::-1]:
    r=capture(f's_{name}_{v}',v,32,10000,flags);r['variant']=name;r['kind']=kind
    baseline=next(c for c in controls if c['view']==v);r['raw_equal']=equal_raw(r,baseline)
    records.append(r)
   save('screen-progress.json',dict(records=records,controls=controls))
 for r in records:
  available=sorted([c for c in controls if c['view']==r['view']],key=lambda x:x['midpoint']);times=[c['midpoint'] for c in available]
  i=bisect.bisect_left(times,r['midpoint']);lo=available[max(0,i-1)];hi=available[min(i,len(available)-1)]
  mix=0 if hi['midpoint']==lo['midpoint'] else (r['midpoint']-lo['midpoint'])/(hi['midpoint']-lo['midpoint'])
  cost=lo['ms_per_sample']*(1-mix)+hi['ms_per_sample']*mix
  r['baseline_ms']=cost;r['speedup']=cost/r['ms_per_sample']
 save('screen.json',dict(records=records,controls=controls,variants=VARIANTS,binary_sha256=SHA))
 original=Path('/tmp/raytracer-local-sweep-baseline')
 if original.exists():
  r=capture('original_control',1,32,10000,binary=original);r['current_raw_equal']=equal_raw(r,controls[0]);save('original-control.json',r)
 return records

def matched(label,v,seed,flags,base,prediction):
 spp=max(2,1+round(base['gpu_total_ms']/prediction))
 for attempt in range(3):
  r=capture(f'{label}_try{attempt}',v,spp,seed,flags)
  if abs(r['gpu_total_ms']/base['gpu_total_ms']-1)<=.03:break
  spp=max(2,1+round(base['gpu_total_ms']/r['ms_per_sample']))
 return dict(baseline=base,candidate=r,budget_ratio=r['gpu_total_ms']/base['gpu_total_ms'],mse_ratio=r['display_mse']/base['display_mse'],radiance_ratio=r['radiance_mse']/base['radiance_mse'],psnr_gain_db=r['psnr_db']-base['psnr_db'])

def quality(screen_records):
 results=[]
 for v in VIEWS:
  for seed in [20000,30000]:
   base=capture(f'q_base_{v}_{seed}',v,64,seed)
   for name,flags,kind in VARIANTS:
    screens=[r for r in screen_records if r['variant']==name]
    if kind in ['exact','geometry'] and all(r['raw_equal'] for r in screens):continue
    prediction=next(r['ms_per_sample'] for r in screens if r['view']==v)
    trial=matched(f'q_{name}_{v}_{seed}',v,seed,flags,base,prediction);trial['variant']=name;results.append(trial)
    print('QUALITY',name,v,seed,'error ratio',trial['mse_ratio'],'budget',trial['budget_ratio'],flush=True)
    save('quality.json',results)
 return results

def combinations(screen_records,quality_records):
 safe=[]
 for name,flags,kind in VARIANTS:
  rows=[r for r in screen_records if r['variant']==name]
  if kind in ['exact','geometry'] and all(r['raw_equal'] and r['speedup']>1.02 for r in rows):safe.append((statistics.geometric_mean(r['speedup'] for r in rows),name,flags))
 safe.sort(reverse=True)
 exact_flags=[f for _,name,flags in safe[:2] for f in flags]
 candidates=[]
 if exact_flags:candidates.append(('best_exact_combo',exact_flags))
 eligible=[]
 for name,flags,kind in VARIANTS:
  if kind=='approximation' or name=='ray_cones':continue
  rows=[r for r in quality_records if r['variant']==name]
  if rows and statistics.mean(r['mse_ratio'] for r in rows)<.97 and all(r['mse_ratio']<1.02 for r in rows):eligible.append((statistics.mean(r['mse_ratio'] for r in rows),name,flags))
 if eligible:
  _,name,flags=min(eligible);candidates.append(('quality_combo',exact_flags+flags))
 if not candidates:candidates.append(('memory_combo',['--packed-shading','--quantized-bvh']))
 results=[]
 for name,flags in candidates:
  for v in VIEWS:
   for seed in [40000,50000]:
    base=capture(f'c_base_{v}_{seed}',v,64,seed)
    pilot=capture(f'c_pilot_{name}_{v}_{seed}',v,24,90000+seed,flags)
    row=matched(f'c_{name}_{v}_{seed}',v,seed,flags,base,pilot['ms_per_sample']);row['variant']=name;results.append(row);save('combinations.json',results)
 return results

def ancillary():
 startups=[]
 capture('cache_prime',1,8,70000,['--scene-cache'],size='96x64')
 for i in range(3):
  order=[False,True] if i%2==0 else [True,False]
  for enabled in order:
   r=capture(f'startup_{i}_{enabled}',1,8,70000,['--scene-cache'] if enabled else [],size='96x64');r['cache']=enabled;startups.append(r);save('startup.json',startups)
 filters=[]
 for v in VIEWS:
  for sparse in [False,True]:
   r=capture(f'filter_{v}_{sparse}',v,64,60000,['--sparse-denoise'] if sparse else [],denoise=True,filter_report=True);r['sparse']=sparse;filters.append(r);save('filters.json',filters)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','screen','quality','combinations','ancillary'],default='all');args=ap.parse_args()
 save('environment.json',dict(platform=platform.platform(),binary_sha256=SHA,capabilities=subprocess.check_output([str(BIN),'--gpu-info'],text=True),reference_spp=2048,reference_bounces=3))
 screens=screen() if args.stage in ['all','screen'] else load('screen.json',{})['records']
 q=quality(screens) if args.stage in ['all','quality'] else load('quality.json',[])
 if args.stage in ['all','combinations']:combinations(screens,q)
 if args.stage in ['all','ancillary']:ancillary()
 print('SWEEP COMPLETE',flush=True)
