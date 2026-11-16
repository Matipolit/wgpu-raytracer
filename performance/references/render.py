#!/usr/bin/env python3
"""Render consistent high-sample references without changing the training dataset."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'performance/references/2048-rr'
OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'training/runs/overnight/manifest.json').read_text())
references=[]
for idx in [1,19]:
 e=manifest[idx];stem=OUT/f'{idx:03d}'
 raw=stem.with_suffix('.f32');png=stem.with_suffix('.png');timing=stem.with_suffix('.json')
 if not (raw.exists() and raw.stat().st_size==e['width']*e['height']*12 and png.exists() and timing.exists()):
  temporary=stem.with_suffix('.partial.f32')
  command=[str(ROOT/'target/release/weekend-raytracer-wgpu'),'--sponza','--bounces','8','--russian-roulette','--no-denoise','--samples','2048','--sample-offset','1000000','--size',f'{e["width"]}x{e["height"]}','--camera',e['camera'],'--exposure',str(e['exposure']),'--render-output',str(png),'--raw-output',str(temporary),'--timing-output',str(timing)]
  print('Rendering reference',idx,'2048 spp / 8 bounces / RR',flush=True)
  subprocess.run(command,cwd=ROOT,check=True)
  temporary.replace(raw)
 references.append(dict(view=idx,width=e['width'],height=e['height'],camera=e['camera'],exposure=e['exposure'],samples=2048,bounces=8,russian_roulette=True,sky_importance=False,denoising=False,sample_offset=1000000,raw=str(raw),png=str(png),timing=str(timing)))
 temp=OUT/'manifest.tmp';temp.write_text(json.dumps(references,indent=2)+'\n');temp.replace(OUT/'manifest.json')
print('Both references complete.',flush=True)
