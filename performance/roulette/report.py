#!/usr/bin/env python3
"""Build a comparison from actual renderer captures and timing reports."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from experiment import OUT, ROOT, capture, metrics

ap=argparse.ArgumentParser();ap.add_argument('--same-spp',action='store_true');args=ap.parse_args()
rows=json.loads((OUT/'summary.json').read_text())
if args.same_spp:
 manifest=json.loads((ROOT/'training/runs/overnight/manifest.json').read_text())
 for row in rows:
  e=manifest[row['view']];r=row['runs'][0];reference=np.fromfile(e['high'],dtype='<f4')
  if not r.get('same_spp_roulette'):
   r['same_spp_roulette']=metrics(capture(e,f'{row["view"]:03d}_10000_rr_same_spp',r['fixed']['spp'],10000,True),e,reference)
 (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
html='''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;color:var(--foreground);font-family:var(--font-sans)}p,label{font-size:13px}p{color:var(--muted-foreground);line-height:1.5}.metric{margin:15px 0}.bars{padding-right:90px}.bar{height:18px;background:var(--chart-2);margin:5px 0;border-radius:3px}.rr{background:var(--chart-1)}.bar span{position:relative;left:100%;padding-left:8px;white-space:nowrap;font-size:12px}select,button{padding:7px;background:var(--secondary);color:var(--foreground);border:1px solid var(--border);border-radius:5px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}img{width:100%;display:block}img[hidden]{display:none}.scene[hidden]{display:none}.detail{margin-top:12px} @media(max-width:500px){.grid{grid-template-columns:1fr}}
</style></head><body><p>640×400 · 8-bounce maximum · no denoising · two views × two seeds<br>Display error against independent-seed 512-spp references, at approximately equal warmed GPU time. Lower is better.</p>'''
for row in rows:
 fixed=statistics.mean(r['fixed']['display_mse'] for r in row['runs']);rr=statistics.mean(r['roulette']['display_mse'] for r in row['runs']);speed=statistics.mean(r['per_sample_speedup'] for r in row['runs'])
 html+=f'<div class="metric"><label>View {row["view"]} · <b>{speed:.2f}× faster samples · {100*(1-rr/fixed):.1f}% lower error</b></label><div class="bars"><div class="bar" style="width:100%"><span>Fixed: 100%</span></div><div class="bar rr" style="width:{100*rr/fixed:.2f}%"><span>RR: {100*rr/fixed:.1f}%</span></div></div></div>'
html+='''<label for="view">View </label><select id="view" onchange="document.querySelectorAll('.scene').forEach(e=>e.hidden=e.id!==this.value)">'''
for row in rows:html+=f'<option value="s{row["view"]}">Courtyard {row["view"]}</option>'
html+='</select><p>First seed shown. Select an image to enlarge it.</p>'
for i,row in enumerate(rows):
 r=row['runs'][0];html+=f'<div class="scene" id="s{row["view"]}"'+(' hidden' if i else '')+'>'
 html+=f'<label>Warmed GPU time: fixed {r["fixed"]["gpu_total_ms"]/1000:.2f} s · RR {r["roulette"]["gpu_total_ms"]/1000:.2f} s</label><div class="grid">'
 for label,path in [(f'Fixed · {r["fixed"]["spp"]} spp',OUT/(r['fixed']['label']+'.png')),(f'Roulette · {r["roulette"]["spp"]} spp',OUT/(r['roulette']['label']+'.png')),('Reference · 512 spp',Path(row['reference_png']))]:
  html+=f'<div><button onclick="document.getElementById(\'large\').src=this.nextElementSibling.src;document.getElementById(\'large\').hidden=false;document.getElementById(\'caption\').textContent=this.textContent">{label}</button><img alt="{label}" src="{path}"></div>'
 same=r.get('same_spp_roulette')
 html+='</div>'
 if same:
  gain=same['display_psnr_db']-r['fixed']['display_psnr_db'];html+=f'<p>At the same {same["spp"]} spp: RR took {same["gpu_total_ms"]/1000:.2f} s; PSNR change {gain:+.2f} dB versus fixed depth.</p>'
 html+='</div>'
html+='''<div class="detail"><label id="caption"></label><img id="large" hidden alt="Enlarged comparison"></div><p>Bars average two seeds per view. RR GPU budgets differ from fixed by up to MAXDIFF%. Reference images still contain Monte Carlo noise; this is a small scene-specific experiment, not proof of a universal gain.</p></body></html>'''
maxdiff=max(abs(r['budget_ratio']-1)*100 for row in rows for r in row['runs'])
html=html.replace('MAXDIFF',f'{maxdiff:.1f}')
reference_spp=rows[0].get('reference_spp',512)
html=html.replace('512-spp',f'{reference_spp}-spp').replace('Reference · 512 spp',f'Reference · {reference_spp} spp')
(OUT/'roulette-results.html').write_text(html)
print(json.dumps(rows,indent=2))
