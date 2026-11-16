#!/usr/bin/env python3
"""Build a measured NDF/VNDF image and error comparison."""
from pathlib import Path
import json
import re
import statistics

BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
rows=json.loads((OUT/'summary.json').read_text())
validation=[]
pattern=r'VNDF nv=([\d.]+) rough=([\d.]+) metal=([\d.]+): energy ([\d.]+)/([\d.]+), variance ([\d.]+)/([\d.]+), rejected ([\d.]+)/([\d.]+)'
for match in re.finditer(pattern,(BASE/'gpu-validation.log').read_text()):
 values=list(map(float,match.groups()))
 validation.append(dict(zip(['nv','roughness','metallic','ndf_mean','vndf_mean','ndf_variance','vndf_variance','ndf_rejected','vndf_rejected'],values)))
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
html='''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font-family:var(--font-sans);color:var(--foreground)}p,label{font-size:13px}p{color:var(--muted-foreground);line-height:1.5}.metric{margin:16px 0}.bars{padding-right:150px}.bar{height:18px;background:var(--chart-2);margin:5px 0;border-radius:3px}.candidate{background:var(--chart-1)}.bar span{position:relative;left:100%;padding-left:8px;white-space:nowrap;font-size:12px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:12px 0}img{display:block;width:100%}img[hidden],.scene[hidden]{display:none}button,select{padding:7px;background:var(--secondary);color:var(--foreground);border:1px solid var(--border);border-radius:5px}@media(max-width:500px){.grid{grid-template-columns:1fr}}</style></head><body><p>GGX NDF vs visible-normal sampling · 640×400 · 3 bounces · no denoising or sky sampling<br>Display error at similar warmed GPU time · lower is better · mean of three seeds per view.</p>'''
for row in rows:
 base=statistics.mean(r['baseline']['display_mse'] for r in row['runs']);visible=statistics.mean(r['vndf']['display_mse'] for r in row['runs']);ratio=visible/base
 cost=statistics.mean(1/r['per_sample_speedup'] for r in row['runs'])
 cost_label=f'{100*(cost-1):+.1f}% time/sample'
 html+=f'<div class="metric"><label>View {row["view"]}: <b>{abs(100*(1-ratio)):.1f}% '+('lower' if ratio<1 else 'higher')+f' error</b> · {cost_label}</label><div class="bars">'
 for label,value,css in [('NDF',1,''),('VNDF',ratio,'candidate')]:
  html+=f'<div class="bar {css}" style="width:{100*value/max(1,ratio):.2f}%"><span>{label}: {100*value:.1f}%</span></div>'
 html+='</div></div>'
html+='<label for="view">View </label><select id="view" onchange="document.querySelectorAll(\'.scene\').forEach(e=>e.hidden=e.id!==this.value)">'
for row in rows:html+=f'<option value="s{row["view"]}">Courtyard {row["view"]}</option>'
html+='</select><p>Select an image to enlarge it. First seed shown.</p>'
for idx,row in enumerate(rows):
 r=row['runs'][0];html+=f'<div class="scene" id="s{row["view"]}"'+(' hidden' if idx else '')+'>'
 html+=f'<label>GPU time: NDF {r["baseline"]["gpu_total_ms"]/1000:.2f} s · VNDF {r["vndf"]["gpu_total_ms"]/1000:.2f} s</label><div class="grid">'
 for label,path in [(f'NDF · {r["baseline"]["spp"]} spp',OUT/(r['baseline']['label']+'.png')),(f'VNDF · {r["vndf"]["spp"]} spp',OUT/(r['vndf']['label']+'.png')),('Reference · 2048 spp',Path(row['reference_png']))]:
  html+=f'<div><button onclick="document.getElementById(\'large\').src=this.nextElementSibling.src;document.getElementById(\'large\').hidden=false;document.getElementById(\'caption\').textContent=this.textContent">{label}</button><img alt="{label}" src="{path}"></div>'
 html+='</div></div>'
html+='<div><label id="caption"></label><img id="large" hidden alt="Enlarged rendering comparison"></div>'
case=next(v for v in validation if v['nv']==0.05 and v['roughness']==0.5 and v['metallic']==1)
ratio=case['vndf_variance']/case['ndf_variance']
html+=f'<p>Controlled grazing-metal test · cos(view)=0.05, roughness=0.5, metallic=1, uniform environment. Sample variance is {100*(1-ratio):.1f}% lower. This is not the scene-wide speedup.</p><div class="bars"><div class="bar" style="width:100%"><span>NDF variance: 100%</span></div><div class="bar candidate" style="width:{100*ratio:.2f}%"><span>VNDF: {100*ratio:.1f}%</span></div></div>'
max_budget=max(abs(r['budget_ratio']-1)*100 for row in rows for r in row['runs'])
html+=f'<p>GPU budgets differ by up to {max_budget:.1f}%. Reference renders also use three bounces and independent seeds; they retain some Monte Carlo noise. Roulette is enabled in both variants and has no effect at this bounce limit. VNDF remains opt-in.</p></body></html>'
(OUT/'vndf-results.html').write_text(html)
