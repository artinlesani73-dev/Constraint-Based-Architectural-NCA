from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];P=R/'.local-artifacts/r3-contribution-2026-10-08'
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
def iso(draw,field,scene,origin,scale=7):
 def p(x,y,z):return (origin[0]+scale*(x-y),origin[1]+scale*(.48*(x+y)-1.18*z))
 # Draw only three camera-facing exposed faces; far-to-near painter order.
 faces=[]
 for z,y,x in np.argwhere(field):
  if x==field.shape[2]-1 or not field[z,y,x+1]:faces.append((x+y+z,[(x+1,y,z),(x+1,y+1,z),(x+1,y+1,z+1),(x+1,y,z+1)],'#316958'))
  if y==field.shape[1]-1 or not field[z,y+1,x]:faces.append((x+y+z,[(x,y+1,z),(x+1,y+1,z),(x+1,y+1,z+1),(x,y+1,z+1)],'#438873'))
  if z==field.shape[0]-1 or not field[z+1,y,x]:faces.append((x+y+z,[(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)],'#7cb499'))
 for _,coords,color in sorted(faces,key=lambda f:f[0]):draw.polygon([p(*v) for v in coords],fill=color)
 # Context wireframes stay visible and cannot hide generated surfaces.
 for b in scene['buildings']:
  x0,x1=b['x'];y0,y1=b['y'];z0,z1=b['z']
  for z in [z0,z1]:draw.line([p(x0,y0,z),p(x1,y0,z),p(x1,y1,z),p(x0,y1,z),p(x0,y0,z)],fill='#adb3b1',width=1)
  for x in [x0,x1]:
   for y in [y0,y1]:draw.line([p(x,y,z0),p(x,y,z1)],fill='#adb3b1',width=1)
 for label,vec in [('X',(5,0,0)),('Y',(0,5,0)),('Z',(0,0,5))]:
  start=p(0,0,0);end=p(*vec);draw.line([start,end],fill='#747f7d',width=2);draw.text(end,label,font=font(13),fill='#52605c')

protocol=json.loads((P/'protocol.json').read_text())
for seed in protocol['seeds']:
 im=Image.new('RGB',(1500,1490),'#f6f5ef');d=ImageDraw.Draw(im)
 d.text((22,15),f'PARTIAL STUDY / fixed planner | seed {seed} | 128 steps',font=font(25),fill='#183d33')
 d.text((22,50),'Same 0.8m cells,24% request,original route. Neutral = constant positive score; not MG7.',font=font(18),fill='#53635b')
 for col,m in enumerate(protocol['methods']):d.text((col*500+24,85),{'G10':'G10 learned baseline','R3':'R3 learned + planner','neutral':'Neutral scores + planner'}[m],font=font(23),fill='#183d33')
 for row,s in enumerate(protocol['sites']):
  y=130+row*330;d.text((20,y),s['scene_id'],font=font(18),fill='#183d33')
  records=json.loads((P/f'case-{row}/results.json').read_text()) if (P/f'case-{row}/results.json').exists() else []
  for col,m in enumerate(protocol['methods']):
   rec=next((v for v in records if v['seed']==seed and v['method']==m),None);p=P/f'case-{row}/{m}-{seed}-128.npz'
   if not p.exists():d.text((col*500+30,y+145),('NOT RUN / INCOMPLETE' if rec is None else 'NO CERTIFIED OUTPUT'),font=font(20),fill='#9c4f2b');continue
   f=np.load(p)['field'];iso(d,f,s,(col*500+250,y+135),3.3)
   score=rec['scores']['128'];d.text((col*500+25,y+270),f"{sum(score['score']['family_pass'].values())}/9 | {int(f.sum())} cells | overall {'PASS' if rec['accepted'] else 'FAIL'}",font=font(18),fill='#183d33')
   d.text((col*500+25,y+296),f"Z/Y/X span {score['extent_cells_zyx']}",font=font(16),fill='#53635b')
 d.text((22,1460),'Exposed development cases. Different form does not establish superior architectural quality.',font=font(18),fill='#53635b');im.save(P/f'comparison-{seed}.png')
print('Saved both comparison plates')
