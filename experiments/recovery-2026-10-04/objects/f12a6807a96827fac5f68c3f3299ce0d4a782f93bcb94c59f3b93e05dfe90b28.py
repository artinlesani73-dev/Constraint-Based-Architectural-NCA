from pathlib import Path
import json,ast,sys,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Independent-Review-2026-10-04')
r=json.loads((OUT/'result.json').read_text());scenes={e['id']:e['scene'] for e in json.loads((OUT/'scene-index.json').read_text())['entries']}
tree=ast.parse(Path(__file__).with_name('render_g10.py').read_text())
func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='iso')
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
exec(compile(ast.Module(body=[func],type_ignores=[]),'<inherited-renderer>','exec'))
# Render all fresh cases plus every regression case with a family failure in either model.
cases=list(dict.fromkeys(x['case'] for x in r['observations'] if x['cohort']=='fresh' or x['status']!='evaluated' or not x.get('score',{}).get('contract_pass',False)))
(OUT/'visual-selection.json').write_text(json.dumps(dict(cases=cases,rule='all fresh plus all family/certificate failures in either model'),indent=2),encoding='utf-8')
v=OUT/'visuals';v.mkdir()
for page in range((len(cases)+5)//6):
 im=Image.new('RGB',(1600,1500),'#f6f5ef');d=ImageDraw.Draw(im)
 d.text((20,10),'Frozen evaluation | G10 raw versus R3 hybrid | no smoothing',font=font(23),fill='#183d33')
 for col,(model,step) in enumerate([('G10',64),('R3',64),('G10',128),('R3',128)]):
  d.text((col*400+25,48),f'{model} / {step} steps',font=font(20),fill='#183d33')
 for row,case in enumerate(cases[page*6:(page+1)*6]):
  y=85+row*230;scene=scenes[case.rsplit('-v',1)[0]]
  d.text((20,y),case,font=font(17),fill='#183d33')
  for col,(model,step) in enumerate([('G10',64),('R3',64),('G10',128),('R3',128)]):
   path=OUT/'cases'/case/f'{model}-{step}.npz'
   if not path.exists():
    d.text((col*400+20,y+100),'No certified output',font=font(18),fill='#9a3c31')
    continue
   with np.load(path) as a:field=a['field']
   rec=next(x for x in r['observations'] if x['case']==case and x['model']==model and x['steps']==step)
   iso(d,field,scene,(col*400+200,y+125),3.3)
   label=f"{sum(rec['score']['family_pass'].values())}/9 | {int(field.sum())} voxels"
   if model=='R3':label+=f" | planner {rec['planner_voxels']/max(1,int(field.sum())-1):.0%}"
   d.text((col*400+20,y+195),label,font=font(16),fill='#183d33')
 im.save(v/f'comparison-{page}.png')
shutil.copyfile(__file__,OUT/'render.py')
print(len(cases),'cases rendered')

