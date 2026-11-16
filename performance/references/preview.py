#!/usr/bin/env python3
"""Generate an inline viewer of the old and new actual reference renders."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'performance/references/2048-rr'
html='''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;color:var(--foreground);font-family:var(--font-sans)}p,label{font-size:13px}p{color:var(--muted-foreground);line-height:1.5}button,select{padding:8px;background:var(--secondary);color:var(--foreground);border:1px solid var(--border);border-radius:5px}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}img{display:block;width:100%}[hidden]{display:none!important}button[aria-pressed="true"]{background:var(--primary);color:var(--primary-foreground)}</style></head><body><p>Courtyard references · 640×400 · 8 bounces · no denoising<br>Application default: 3 bounces, roulette enabled. Reference bounce count stays at 8 for comparison with existing captures.</p><div class="controls"><label for="view">View</label><select id="view" onchange="showReferences()"><option value="1">Courtyard 1</option><option value="19">Courtyard 19</option></select><button id="new" aria-pressed="true" onclick="quality='new';showReferences()">2048 spp · RR on</button><button id="old" aria-pressed="false" onclick="quality='old';showReferences()">Previous 512 spp · RR off</button></div>'''
for idx in [1,19]:
 for quality,path in [('new',OUT/f'{idx:03d}.png'),('old',ROOT/f'training/runs/overnight/{idx:03d}_high.png')]:
  html+=f'<img class="reference" data-view="{idx}" data-quality="{quality}" src="{path}" alt="Courtyard {idx}: '+('2048 spp, roulette on' if quality=='new' else '512 spp, roulette off')+'"'+('' if idx==1 and quality=='new' else ' hidden')+'>'
html+='''<p>Both settings target the same eight-bounce lighting. Sample seeds differ. Noise is reduced rather than guaranteed absent.</p><script>let quality='new';function showReferences(){const selected=document.getElementById('view').value;document.querySelectorAll('.reference').forEach(e=>e.hidden=e.dataset.view!==selected||e.dataset.quality!==quality);document.getElementById('new').setAttribute('aria-pressed',quality==='new');document.getElementById('old').setAttribute('aria-pressed',quality==='old');}</script></body></html>'''
(OUT/'reference-preview.html').write_text(html)
