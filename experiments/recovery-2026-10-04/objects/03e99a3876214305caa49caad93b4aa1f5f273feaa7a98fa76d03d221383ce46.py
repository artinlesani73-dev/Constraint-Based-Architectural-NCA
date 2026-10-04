from pathlib import Path
import json,hashlib,zipfile,shutil,itertools,ast
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Variety-2026-10-04')
r=json.loads((P/'result.json').read_text());scenes=json.loads((P/'scenes.json').read_text())
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
tree=ast.parse(Path('render_g10.py').read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='iso');exec(compile(ast.Module(body=[func],type_ignores=[]),'<saved-renderer>','exec'))
im=Image.new('RGB',(1440,1280),'#f6f5ef');d=ImageDraw.Draw(im)
d.text((20,12),'Exploratory site / seed variety | 128 steps | 24% request | raw saved voxels',font=font(23),fill='#183d33')
for col,(label,seed) in enumerate(itertools.product(['G10','R3'],[2101,2102,2103])):d.text((col*240+10,50),f'{label} / seed {seed}',font=font(17),fill='#183d33')
for row,s in enumerate(scenes):
 y=90+row*290;sid=s['scene_id'];d.text((20,y),sid,font=font(19),fill='#183d33')
 for col,(label,seed) in enumerate(itertools.product(['G10','R3'],[2101,2102,2103])):
  rec=next(v for v in r['records'] if v['scene']==sid and v['model']==label and v['seed']==seed and v['steps']==128)
  if rec['status']=='evaluated':
   with np.load(P/'cases'/sid/f'{label}-{seed}-128.npz') as a:f=a['field']
   iso(d,f,s,(col*240+120,y+150),2.8)
   text=f"{sum(rec['score']['family_pass'].values())}/9 | growth {rec['growth64_128']:.1%}"
  else:text='Certificate failed'
  d.text((col*240+10,y+245),text,font=font(15),fill='#183d33')
im.save(P/'comparison.png')
rows=[]
for v in r['diversity']:
 vals=[p['jaccard_distance'] for p in v['pairs']];rows.append(f"| {v['scene']} | {v['model']} | {np.mean(vals):.3f} |" if vals else f"| {v['scene']} | {v['model']} | unavailable |")
failures=[dict(scene=v['scene'],model=v['model'],seed=v['seed'],steps=v['steps'],failed_families=[k for k,b in v.get('score',{}).get('family_pass',{}).items() if not b],status=v['status']) for v in r['records'] if v['status']!='evaluated' or not v['score']['contract_pass']]
(P/'failure-index.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
report='''# R3 variety assessment — exploratory, 2026-10-04

Decision: follow R3 as the practical experimental path; retain G10 comparison. Test geometry transfer and firing variation before increasing resolution. Nine families and whole-building-volume interpretation unchanged. MG7 is not replaced.

One compact CPU assessment: four constructed geometry changes, three firing seeds each, 24% request and saved horizons 64/128. There are only four sites, not twelve independent sites. These probes are exploratory, not a new held-out release gate; uniqueness against all historical training scenes was not established. Weights are unchanged. The optional firing-seed argument is the only adapter edit, and default-seed births, provenance and both full horizon states exactly match the independent-review saved case.

## Outcomes at 128

'''
for label,s in r['summary'].items():report+=f"- {label}: {s['family_pass']}/{s['total']} all-nine passes; {s['stable']}/{s['total']} within 5% late growth; max absolute volume error {s['max_volume_error']*100:.3f} percentage points.\n"
report+='''
## Variation between firing seeds

Mean pairwise occupied-voxel Jaccard distance at128. Zero means identical occupancy; larger means more different occupied sets. It is not an architectural-quality or useful-diversity metric. The deterministic planner is shared by all seeds on each site.

| Site | Model | Mean distance |
|---|---|---:|
'''+ '\n'.join(rows)+'''

Per-horizon family failures are preserved in failure-index.json; full metrics, traces, states, birth provenance and contexts are retained. Results do not justify tuning on these four probes and calling the same probes unseen afterward.

## Next

Use these results to set the boundary of the local R3 generation endpoint: report certificate failures explicitly, retain exact source/seed/scene/results per request, and keep raw G10 comparison. A passing result alone does not establish meaningful form diversity. Review the paired geometry plate before deciding whether a new learning objective or planner diversity mechanism is warranted. Do not increase resolution yet or start paid training.

Repository synchronization and off-device backup remain pending. No Drive access, remote push, publication or live-model promotion occurred. This milestone links to ../G11-R3-Preview-2026-10-04/RESUME.md. Original R3 source and all previous results remain unchanged.
'''
(P/'REVIEW.md').write_text(report,encoding='utf-8');shutil.copyfile(__file__,P/'review.py');shutil.copyfile('render_g10.py',P/'inherited-renderer.py')
print(report)
