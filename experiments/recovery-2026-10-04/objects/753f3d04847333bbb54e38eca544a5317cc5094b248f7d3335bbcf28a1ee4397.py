from pathlib import Path
import json,ast,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Modes-2026-10-04');r=json.loads((P/'transfer/result.json').read_text());assert len(r)==6
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
func=next(n for n in ast.parse(Path('render_g10.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='iso');exec(compile(ast.Module(body=[func],type_ignores=[]),'<renderer>','exec'))
im=Image.new('RGB',(1440,1120),'#f6f5ef');d=ImageDraw.Draw(im);d.text((20,20),'Route transfer | fixed seed2101 / 24% / 128 steps | exploratory exposed sites',font=font(24),fill='#183d33');pairs=[]
for row,scene in enumerate(dict.fromkeys(x['scene'] for x in r)):
 variants=[v for v in r if v['scene']==scene];s=variants[0]['result']['scene'];arrays={}
 for col,mode in enumerate(['original','low_y','high_y']):
  path=P.parent/'G11-R3-Variety-2026-10-04/cases'/scene/'R3-2101-128.npz' if mode=='original' else P/'runs'/next(v['id'] for v in variants if v['mode']==mode)/'R3-128.npz'
  x=col*480;y=85+row*330;d.text((x+20,y),scene.replace('r3-variety-','')+' / '+mode,font=font(20),fill='#183d33')
  if not path.exists():d.text((x+20,y+170),'No certified output',font=font(20),fill='#963721');continue
  with np.load(path) as a:f=a['field'].astype(bool)
  arrays[mode]=f;iso(d,f,s,(x+240,y+175),4.8);d.text((x+20,y+285),f'{f.sum()} voxels',font=font(17),fill='#183d33')
 for mode in ['low_y','high_y']:
  if mode in arrays:pairs.append(dict(scene=scene,mode=mode,jaccard_distance_from_original=float(1-(arrays[mode]&arrays['original']).sum()/(arrays[mode]|arrays['original']).sum())))
im.save(P/'transfer/comparison.png');(P/'transfer/diversity.json').write_text(json.dumps(pairs,indent=2))
summary=[]
for v in r:
 p=P/'runs'/v['id'];manifest=json.loads((p/'manifest.json').read_text());assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
 req=json.loads((p/'request.json').read_text());assert req['route_mode']==v['mode'] and v['result']['route_mode']==v['mode']
 summary.append(dict(scene=v['scene'],mode=v['mode'],status=v['status']['state']))
(P/'transfer/verification.json').write_text(json.dumps(dict(run_hashes=True,mode_provenance=True,summary=summary),indent=2));print(json.dumps(summary))
for n in ['review_r3_modes.py','check_r3_modes.py','render_g10.py']:shutil.copyfile(n,P/n)
