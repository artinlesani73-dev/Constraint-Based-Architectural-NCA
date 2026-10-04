from pathlib import Path
import ast,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Route-Options-2026-10-04');r=json.loads((P/'result.json').read_text());scene=json.loads((P/'baseline/scene.json').read_text())
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
func=next(n for n in ast.parse(Path('render_g10.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='iso');exec(compile(ast.Module(body=[func],type_ignores=[]),'<renderer>','exec'))
im=Image.new('RGB',(1440,660),'#f6f5ef');d=ImageDraw.Draw(im);d.text((25,20),'One site / one seed / one volume budget — planner route alternatives',font=font(26),fill='#183d33')
for i,name in enumerate(['baseline','low_y','high_y']):
 path=P/'baseline/R3-128.npz' if name=='baseline' else P/f'{name}-128.npz';x=i*480
 d.text((x+25,90),name.replace('_',' ').upper(),font=font(23),fill='#183d33')
 if not path.exists():d.text((x+25,170),'No certified output',font=font(20),fill='#9a3322');continue
 with np.load(path) as a:f=a['field']
 iso(d,f,scene,(x+240,345),6)
 d.text((x+25,530),f'{f.sum()} occupied voxels',font=font(20),fill='#183d33')
 if name!='baseline':
  v=next(v for v in r['variants'] if v['variant']==name);d.text((x+25,564),f"Checks {'pass' if v['passes'] else 'fail'} | planner {v['outputs']['128']['planner_share']:.1%}",font=font(20),fill='#183d33')
d.text((25,620),'128 steps · 24% request · firing seed2102 · unchanged trained weights · exploratory, single-site result',font=font(18),fill='#53635b');im.save(P/'comparison.png')
