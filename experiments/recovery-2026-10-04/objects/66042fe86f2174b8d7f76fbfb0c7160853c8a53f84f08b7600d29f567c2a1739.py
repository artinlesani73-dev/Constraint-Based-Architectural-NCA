from pathlib import Path
import ast,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Scale-2026-10-04');font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
s=Path('render_g10.py').read_text().replace('x==31','x==field.shape[2]-1').replace('y==31','y==field.shape[1]-1').replace('z==31','z==field.shape[0]-1');func=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='iso');exec(compile(ast.Module(body=[func],type_ignores=[]),'<variable-grid-renderer>','exec'))
im=Image.new('RGB',(1400,750),'#f6f5ef');d=ImageDraw.Draw(im);d.text((25,20),'Larger physical domain / same0.8 m voxels / same model',font=font(27),fill='#183d33')
for i,n in enumerate([32,40]):
 r=json.loads((P/f'{n}-result.json').read_text());scene=json.loads((P/f'{n}-scene.json').read_text());x=i*700
 with np.load(P/f'{n}-128.npz') as a:f=a['field']
 d.text((x+25,85),f'{n}³ cells | {n*.8:.1f} m domain edge',font=font(24),fill='#183d33');iso(d,f,scene,(x+350,340),6)
 d.text((x+25,560),f"Checks: {'pass' if r['passed'] else 'fail'} | growth64–128: {r['growth']:.2%}",font=font(20),fill='#183d33')
 d.text((x+25,600),f"128-step rollout: {r['times']['rollout128']:.2f}s | volume {f.sum()*.8**3:.1f} m³",font=font(20),fill='#183d33')
 d.text((x+25,640),f"Process peak working set: {r['peak_process_working_set_bytes']/2**20:.0f} MiB",font=font(20),fill='#183d33')
d.text((25,710),'Two single CPU timing samples; no GPU or training-memory estimate. Same drawing scale in both panels.',font=font(18),fill='#53635b');im.save(P/'comparison.png')
