"""Same-count validation of combinations selected by the sweep."""
import json
from run import OUT,capture,equal_raw,save
screen=json.loads((OUT/'screen.json').read_text());combinations=json.loads((OUT/'combinations.json').read_text())
results=[]
for name in dict.fromkeys(r['variant'] for r in combinations):
 flags=next(r['candidate']['flags'] for r in combinations if r['variant']==name)
 for view in [1,19]:
  candidate=capture(f'identity_{name}_{view}',view,32,10000,flags)
  baseline=next(r for r in screen['controls'] if r['view']==view)
  results.append(dict(variant=name,view=view,flags=flags,raw_identical=equal_raw(baseline,candidate)))
save('combination-identities.json',results)
print(results)
