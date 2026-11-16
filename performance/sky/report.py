#!/usr/bin/env python3
"""Show measured equal-budget sky-sampling results, including regressions."""
from pathlib import Path
import json
import statistics

OUT=Path(__file__).resolve().parent/'results'
rows=json.loads((OUT/'summary.json').read_text())
html='''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font-family:var(--font-sans);color:var(--foreground)}p,label{font-size:13px}p{color:var(--muted-foreground);line-height:1.5}.metric{margin:16px 0}.bars{padding-right:140px}.bar{height:18px;background:var(--chart-2);margin:5px 0;border-radius:3px}.sky{background:var(--chart-1)}.bar span{position:relative;left:100%;padding-left:8px;white-space:nowrap;font-size:12px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:12px 0}img{display:block;width:100%}img[hidden],.scene[hidden]{display:none}button,select{padding:7px;background:var(--secondary);color:var(--foreground);border:1px solid var(--border);border-radius:5px}@media(max-width:500px){.grid{grid-template-columns:1fr}}</style></head><body><p>Sky next-event estimation + MIS · 640×400 · roulette in both variants · no denoising<br>Display error at similar GPU budgets · lower is better · average of two seeds per view.</p>'''
for row in rows:
 base=statistics.mean(r['baseline']['display_mse'] for r in row['runs']);sky=statistics.mean(r['sky']['display_mse'] for r in row['runs']);ratio=sky/base
 html+=f'<div class="metric"><label>View {row["view"]}: <b>{abs(100*(1-ratio)):.1f}% '+('lower' if ratio<1 else 'higher')+' error</b></label><div class="bars">'
 scale=max(1,ratio)
 for label,value,style in [('RR only',1,''),('RR + sky',ratio,'sky')]:
  html+=f'<div class="bar {style}" style="width:{100*value/scale:.2f}%"><span>{label}: {100*value:.1f}%</span></div>'
 html+='</div></div>'
html+='<label for="view">View </label><select id="view" onchange="document.querySelectorAll(\'.scene\').forEach(e=>e.hidden=e.id!==this.value)">'
for row in rows:html+=f'<option value="s{row["view"]}">Courtyard {row["view"]}</option>'
html+='</select><p>Select an image to enlarge it. First seed shown.</p>'
for idx,row in enumerate(rows):
 r=row['runs'][0];html+=f'<div class="scene" id="s{row["view"]}"'+(' hidden' if idx else '')+'>'
 html+=f'<label>GPU time: RR only {r["baseline"]["gpu_total_ms"]/1000:.2f} s · RR + sky {r["sky"]["gpu_total_ms"]/1000:.2f} s</label><div class="grid">'
 for label,path in [(f'RR only · {r["baseline"]["spp"]} spp',OUT/(r['baseline']['label']+'.png')),(f'RR + sky · {r["sky"]["spp"]} spp',OUT/(r['sky']['label']+'.png')),('Reference · 512 spp',Path(row['reference_png']))]:
  html+=f'<div><button onclick="document.getElementById(\'large\').src=this.nextElementSibling.src;document.getElementById(\'large\').hidden=false;document.getElementById(\'caption\').textContent=this.textContent">{label}</button><img alt="{label}" src="{path}"></div>'
 html+='</div></div>'
max_budget=max(abs(r['budget_ratio']-1)*100 for row in rows for r in row['runs'])
html+=f'<div><label id="caption"></label><img id="large" hidden alt="Enlarged rendering comparison"></div><p>Sky sampling cost 42–46% more per sample. GPU budgets differ by up to {max_budget:.1f}%. References use different seeds and still contain noise. Compressed-radiance MSE worsened in both views. This is a two-view experiment; sky sampling remains opt-in.</p></body></html>'
reference_spp=rows[0].get('reference_spp',512)
html=html.replace('512-spp',f'{reference_spp}-spp').replace('Reference · 512 spp',f'Reference · {reference_spp} spp')
(OUT/'sky-results.html').write_text(html)
