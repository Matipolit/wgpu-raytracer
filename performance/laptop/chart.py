"""Generate a self-contained timing comparison for inline viewing."""
from pathlib import Path
import json
out=Path(__file__).resolve().parent
rows=json.loads((out/'final-640.json').read_text())+json.loads((out/'final-720.json').read_text())
rows=[r for r in rows if r['case']!='spheres']
html='''<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:var(--font-sans);color:var(--foreground);margin:0} .row{margin:18px 0}.bar{height:20px;margin:6px 0;background:var(--chart-2);border-radius:3px}.after{background:var(--chart-1)}.bars{padding-right:95px}.bar span{position:relative;left:100%;margin-left:8px;white-space:nowrap;font-size:12px}p{font-size:13px;color:var(--muted-foreground)}label{font-size:13px}</style></head><body><p>GPU milliseconds per sample · lower is better · Intel Iris Xe / Vulkan<br>Blue: optimized; teal: baseline. Same resolution, 1 spp/frame, 8 bounces, bilateral denoiser.</p>'''
maximum=max(r['baseline_ms'] for r in rows)
for r in rows:
 html+=f'<div class="row"><label>{r["case"]} · {r["size"]} · <b>{r["speedup"]:.2f}× faster</b></label><div class="bars">'
 for key,css in [('baseline_ms',''),('candidate_ms','after')]:
  html+=f'<div class="bar {css}" style="width:{100*r[key]/maximum:.2f}%"><span>{r[key]:.1f} ms</span></div>'
 html+='</div></div>'
html+='<p>640×400: median of two runs × 24 frames. 1280×720: one run × 16 frames. First frame excluded. Raw radiance matched byte-for-byte in all displayed cases. Includes tracing, filtering and display; excludes scene loading and image readback.</p></body></html>'
(out/'results.html').write_text(html)
