#!/usr/bin/env python3
"""Re-score existing captures against the completed 2048-spp RR references."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
references={e['view']:e for e in json.loads((ROOT/'performance/references/2048-rr/manifest.json').read_text())}
for experiment in ['roulette','sky']:
 path=ROOT/f'performance/{experiment}/results/summary.json'
 backup=path.with_name('summary-512-reference.json')
 if not backup.exists(): backup.write_bytes(path.read_bytes())
 rows=json.loads(path.read_text())
 spec=importlib.util.spec_from_file_location(f'metrics_{experiment}',ROOT/f'performance/{experiment}/experiment.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for row in rows:
  ref=references[row['view']];reference=np.fromfile(ref['raw'],dtype='<f4')
  row.update(reference=ref['raw'],reference_png=ref['png'],reference_spp=2048,reference_bounces=8,reference_russian_roulette=True)
  for run in row['runs']:
   for record in run.values():
    if isinstance(record,dict) and 'label' in record and 'display_mse' in record:
     module.metrics(record,ref,reference)
   baseline=run['fixed' if experiment=='roulette' else 'baseline']
   candidate=run['roulette' if experiment=='roulette' else 'sky']
   run['psnr_gain_db']=candidate['display_psnr_db']-baseline['display_psnr_db']
 path.write_text(json.dumps(rows,indent=2)+'\n')
 subprocess.run([sys.executable,str(ROOT/f'performance/{experiment}/report.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 print('Updated',experiment,'metrics and report against 2048 spp RR reference',flush=True)
