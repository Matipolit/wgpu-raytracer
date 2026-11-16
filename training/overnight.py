#!/usr/bin/env python3
"""Resumable standalone capture / CPU training / WGSL export / GPU acceptance job."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def run(args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)

def export(model, path):
    w, b, v, c = [p.detach().cpu().tolist() for p in model.parameters()]
    def num(x):
        return f'{x:.9e}'
    lines = ['// Generated 127-parameter 27 -> 4 -> 3 residual MLP. RGB/(1+RGB).',
             '@compute @workgroup_size(8,8)',
             'fn csNeural(@builtin(global_invocation_id) id: vec3<u32>) {',
             'if id.x >= frameData.x || id.y >= frameData.y { return; }',
             'var x: array<f32,27>;',
             'var k = 0u;',
             'for (var dy = -1; dy <= 1; dy += 1) { for (var dx = -1; dx <= 1; dx += 1) {',
             'let p = clamp(vec2<i32>(id.xy)+vec2(dx,dy), vec2(0), vec2<i32>(frameData.xy)-vec2(1));',
             'let rgb = max(averagePixel(u32(p.x),u32(p.y)), vec3(0.));',
             'let z = rgb/(vec3(1.)+rgb);',
             'x[k]=z.r; x[k+1u]=z.g; x[k+2u]=z.b; k+=3u; } }']
    for i in range(4):
        lines.append(f'let h{i} = max(0., {num(b[i])}'+''.join(f'+({num(a)})*x[{j}]' for j,a in enumerate(w[i]))+');')
    for i in range(3):
        lines.append(f'let o{i} = clamp(x[{12+i}]+{num(c[i])}'+''.join(f'+({num(a)})*h{j}' for j,a in enumerate(v[i]))+', 0., 0.9999);')
    lines += ['var color = vec3(o0,o1,o2); color = color/(vec3(1.)-color);',
              'if samplingParams.denoisingEnabled == 0u { color = averagePixel(id.x,id.y); }',
              'filteredImage[frameData.x*id.y+id.x]=array<f32,3>(color.r,color.g,color.b);', '}']
    path.write_text('\n'.join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT/'training/runs/overnight')
    ap.add_argument('--pairs', type=int, default=36)
    ap.add_argument('--size', default='640x400')
    ap.add_argument('--benchmark-size', default='1280x720')
    ap.add_argument('--hours', type=float, default=6)
    ap.add_argument('--steps', type=int, default=1000000)
    ap.add_argument('--threads', type=int, default=max(1, min(4, (os.cpu_count() or 2)//2)))
    args = ap.parse_args()
    assert args.pairs >= 30
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/'job.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(pairs=args.pairs,size=args.size,version=1)
    config_path=out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit('Capture configuration changed; use a new --output directory')
    atomic_json(config_path,config)
    width,height = map(int,args.size.split('x'))
    assert width > 0 and height > 0 and args.hours > 0 and args.steps > 0
    binary = ROOT/'target/release/weekend-raytracer-wgpu'
    run(['cargo','build','--release','--locked'])
    manifest = []
    # Unique positions along both aisles and courtyard; varying height, aim, FOV and exposure.
    for i in range(args.pairs):
        rng = random.Random(1729+i)
        side = [-1,0,1][i%3]
        x = -9+18*(i//3)/max(1,(args.pairs-1)//3)
        eye = [x, 1.2+0.35*(i%4), side*2.6]
        target = [rng.uniform(-5,5),rng.uniform(2,6),rng.uniform(-1.5,1.5)]
        direction = [target[j]-eye[j] for j in range(3)]
        camera = ','.join(map(str,eye+direction+[50+5*(i%5)]))
        exposure = [1.,2.,3.][(i//3)%3]
        entry = dict(id=i,camera=camera,exposure=exposure,split='validation' if i%6==0 else 'train',width=width,height=height,low_spp=1,high_spp=512)
        for quality,spp in [('low',1),('high',512)]:
            stem = out/f'{i:03d}_{quality}'
            raw = stem.with_suffix('.f32'); png = stem.with_suffix('.png')
            entry[quality] = str(raw)
            if raw.exists() and raw.stat().st_size==width*height*12 and png.exists():
                continue
            temporary = raw.with_suffix('.partial')
            run([binary,'--bounces','8','--no-russian-roulette','--sponza','--no-denoise','--camera',camera,'--exposure',exposure,
                 '--samples',spp,'--size',args.size,'--render-output',png,'--raw-output',temporary])
            temporary.replace(raw)
        manifest.append(entry)
        atomic_json(out/'manifest.json',manifest)
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(27,4),torch.nn.ReLU(),torch.nn.Linear(4,3))
    torch.nn.init.zeros_(model[2].weight); torch.nn.init.zeros_(model[2].bias)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    datasets = {'train':[], 'validation':[]}
    for entry in manifest:
        tensors=[]
        for key in ['low','high']:
            a=np.fromfile(entry[key],dtype='<f4').reshape(height,width,3).copy()
            assert np.isfinite(a).all(), entry[key]
            t=torch.from_numpy(a).clamp_min_(0); tensors.append(t/(1+t))
        low,high=tensors
        padded=F.pad(low.permute(2,0,1).unsqueeze(0),(1,1,1,1),mode='replicate')[0].permute(1,2,0)
        datasets[entry['split']].append((padded,high))
    def batch(items,n,generator=None):
        low,high=items[random.randrange(len(items))]
        y=torch.randint(height,(n,),generator=generator); x=torch.randint(width,(n,),generator=generator)
        patch=torch.stack([low[y+dy,x+dx] for dy in range(3) for dx in range(3)],1).reshape(n,27)
        return patch,high[y,x]
    # Fixed held-out pixels from ALL validation cameras, never used for updates.
    random.seed(42)
    validation=[]
    for item in datasets['validation']:
        validation.append(batch([item],8192))
    vx=torch.cat([a for a,b in validation]); vy=torch.cat([b for a,b in validation])
    baseline=F.mse_loss(vx[:,12:15],vy).item()
    best=float('inf'); first_step=0
    checkpoint=out/'checkpoint.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,weights_only=True)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        best=ck['best']; first_step=ck['step']+1
    start=time.monotonic()
    for step in range(first_step,args.steps):
        model.train(); x,y=batch(datasets['train'],4096)
        predicted=(x[:,12:15]+model(x)).clamp(0,0.9999)
        loss=F.l1_loss(predicted,y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step%500==0 or step==args.steps-1 or time.monotonic()-start>args.hours*3600:
            model.eval()
            with torch.no_grad():
                mse=F.mse_loss((vx[:,12:15]+model(vx)).clamp(0,0.9999),vy).item()
            if mse<best:
                best=mse; export(model,out/'model.wgsl')
                torch.save(model.state_dict(),out/'best.pt')
            tmp=checkpoint.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,step=step),tmp); tmp.replace(checkpoint)
            atomic_json(out/'training.json',dict(step=step,baseline_mse=baseline,best_mse=best,psnr_gain_db=10*math.log10(baseline/max(best,1e-12)),parameters=127,domain='radiance/(1+radiance)',pairs=args.pairs))
            print(f'step {step}: validation MSE {mse:.6f}, best {best:.6f}, noisy {baseline:.6f}',flush=True)
            if time.monotonic()-start>args.hours*3600: break
    report=out/'timing.json'
    run([binary,'--bounces','8','--no-russian-roulette','--sponza','--samples','1','--size',args.benchmark_size,'--neural-model',out/'model.wgsl','--filter-report',report,'--render-output',out/'neural.png'])
    timing=json.loads(report.read_text())
    accepted=best<baseline and timing['filter_p95_ms']<=5
    atomic_json(out/'acceptance.json',dict(accepted=accepted,quality_improved=best<baseline,within_5ms=timing['filter_p95_ms']<=5,**timing))
    print('FINISHED; acceptance:',accepted,flush=True)
    if not accepted: raise SystemExit(2)

if __name__=='__main__': main()
