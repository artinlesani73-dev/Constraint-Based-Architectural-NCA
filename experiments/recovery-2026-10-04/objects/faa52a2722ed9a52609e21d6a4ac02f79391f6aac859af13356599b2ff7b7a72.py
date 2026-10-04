from pathlib import Path
import json,shutil,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G9-Final-Review-2026-10-04')
STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 64
V=OUT/f'visuals-final-{STEPS}';V.mkdir(exist_ok=False)
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
split=json.loads((OUT/'scene-index.json').read_text());scenes={r['id']:r['scene'] for r in split['entries']}
result=json.loads((OUT/'result.json').read_text())
records=[r for r in result['observations'] if r['steps']==STEPS]
def iso(draw,field,scene,origin,scale=7):
 def p(x,y,z):return (origin[0]+scale*(x-y),origin[1]+scale*(.48*(x+y)-1.18*z))
 # Draw only three camera-facing exposed faces; far-to-near painter order.
 faces=[]
 for z,y,x in np.argwhere(field):
  if x==31 or not field[z,y,x+1]:faces.append((x+y+z,[(x+1,y,z),(x+1,y+1,z),(x+1,y+1,z+1),(x+1,y,z+1)],'#316958'))
  if y==31 or not field[z,y+1,x]:faces.append((x+y+z,[(x,y+1,z),(x+1,y+1,z),(x+1,y+1,z+1),(x,y+1,z+1)],'#438873'))
  if z==31 or not field[z+1,y,x]:faces.append((x+y+z,[(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)],'#7cb499'))
 for _,coords,color in sorted(faces,key=lambda f:f[0]):draw.polygon([p(*v) for v in coords],fill=color)
 # Context wireframes stay visible and cannot hide generated surfaces.
 for b in scene['buildings']:
  x0,x1=b['x'];y0,y1=b['y'];z0,z1=b['z']
  for z in [z0,z1]:draw.line([p(x0,y0,z),p(x1,y0,z),p(x1,y1,z),p(x0,y1,z),p(x0,y0,z)],fill='#adb3b1',width=1)
  for x in [x0,x1]:
   for y in [y0,y1]:draw.line([p(x,y,z0),p(x,y,z1)],fill='#adb3b1',width=1)
 for label,vec in [('X',(5,0,0)),('Y',(0,5,0)),('Z',(0,0,5))]:
  start=p(0,0,0);end=p(*vec);draw.line([start,end],fill='#747f7d',width=2);draw.text(end,label,font=font(13),fill='#52605c')
def projection(draw,field,context,xy,kind,scale=8):
 if kind=='front':mass=field.any(1);existing=context[2].astype(bool).any(1);interfaces=context[5].astype(bool).any(1)
 else:mass=field.any(0);existing=context[2].astype(bool).any(0);interfaces=context[5].astype(bool).any(0)
 for j,i in np.ndindex(mass.shape):
  color='#2c7560' if mass[j,i] else '#d9ddda' if existing[j,i] else '#ffffff'
  y=31-j;box=(xy[0]+i*scale,xy[1]+y*scale,xy[0]+(i+1)*scale,xy[1]+(y+1)*scale)
  draw.rectangle(box,fill=color)
  if interfaces[j,i]:draw.rectangle(box,outline='#d79142',width=2)
 draw.rectangle((xy[0],xy[1],xy[0]+32*scale,xy[1]+32*scale),outline='#bec8c3')
 draw.text((xy[0],xy[1]+32*scale+6),'X  0 - 25.6 m    '+('Z up' if kind=='front' else 'Y up'),font=font(13),fill='#65746e')
for sid in dict.fromkeys(r['scene'] for r in records):
 im=Image.new('RGB',(1440,1020),'#f6f5ef');d=ImageDraw.Draw(im)
 d.text((28,18),sid.replace('g1-','').replace('g7-','').replace('g8-','').replace('_',' ').upper()+(' | G8 baseline' if sid.startswith('g8-baseline-') else ' | G9 frozen evaluation'),font=font(26),fill='#183d33')
 d.text((28,56),f'{STEPS} steps | Green: generated building volume | Gray: existing context | Orange: connection regions',font=font(18),fill='#53635b')
 d.text((28,85),'0.8 m voxels; no smoothing or filling. Orthographic projections show occupied extent, not interior sections.',font=font(16),fill='#53635b')
 for col,r in enumerate([r for r in records if r['scene']==sid]):
  x=col*480;case=r['case'];d.text((x+25,125),f"Requested {r['request']:.0%} | {sum(r['score']['family_pass'].values())}/9 families",font=font(23),fill='#183d33')
  if r['status']!='evaluated':continue
  with np.load(OUT/f'observations/{case}-{STEPS}.npz') as a:field=a['field']
  with np.load(OUT/f'contexts/{case}.npz') as a:context=a['context']
  iso(d,field,scenes[sid],(x+242,300),4.8)
  d.text((x+28,455),f"Actual {r['score']['volume_fraction']:.2%} | {r['score']['gross_volume_m3']:.1f} m3",font=font(18),fill='#183d33')
  z,y,xx=r['extent_zyx_cells'];d.text((x+28,483),f"Extent X/Y/Z: {xx*.8:.1f} / {y*.8:.1f} / {z*.8:.1f} m",font=font(17),fill='#53635b')
  d.text((x+28,520),'FRONT - projected X/Z occupancy',font=font(16),fill='#53635b');projection(d,field,context,(x+90,549),'front',6)
  d.text((x+28,786),'PLAN - projected X/Y occupancy',font=font(16),fill='#53635b');projection(d,field,context,(x+310,791),'plan',4)
  with np.load(OUT/f'observations/{case}-128.npz') as a:later=a['field']
  with np.load(OUT/f'observations/{case}-64.npz') as a:earlier=a['field']
  d.text((x+28,825),f"64 to 128 change:\n{int((later!=earlier).sum())} voxels",font=font(18),fill='#53635b')
 d.text((28,981),'Bounding extents do not imply uniform thickness. A passing pilot contract does not certify architectural quality.',font=font(16),fill='#53635b')
 im.save(V/(sid+'.png'))
shutil.copyfile(__file__,V/'visualization-script.py')
print('Saved geometry plates.')
