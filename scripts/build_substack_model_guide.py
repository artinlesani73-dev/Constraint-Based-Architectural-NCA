"""White, high-resolution model guide for the Substack article."""
from pathlib import Path
import argparse, csv, json, hashlib, textwrap
from PIL import Image, ImageDraw, ImageFont
ROWS=[
('Phase 1','Original experiment','Constraint-only training for pavilion-like extensions. Its 500-epoch record uses historical objectives that differ from the later massing tests.'),
('G8','Trained baseline','Learns from procedural teacher geometry using frontier classification, soft volume discrepancy and a global volume band. Reference for G9 and G10.'),
('G9','Trained ranking variant','Adds access ranking to the G8 base objective: favor teacher-positive growth that advances a connection. Both score groups receive ranking gradients.'),
('G10','Trained gradient variant','Uses the same ranking formula as G9, but holds the other-candidate scores fixed in that auxiliary gradient. A separately trained checkpoint.'),
('R3','Hybrid system','Reuses frozen G10 weights and adds route planning, coverage coordination and growth guards. No additional training or separate training-loss curve.'),
('Neutral','Procedural control','Uses the R3 planner and guards with constant positive occupancy logits and zero hidden updates. No trained network signal drives its growth.')]
def build(out):
 out.mkdir(parents=True,exist_ok=True)
 target=out/'00-model-guide.png'
 if target.exists(): raise FileExistsError('Preserve earlier output; select a new directory.')
 scale=4; W,H=1200,1160
 im=Image.new('RGB',(W*scale,H*scale),'white');d=ImageDraw.Draw(im)
 regular='C:/Windows/Fonts/arial.ttf';bold='C:/Windows/Fonts/arialbd.ttf'
 def text(x,y,s,size=23,color='#24383c',heavy=False):
  d.text((x*scale,y*scale),s,font=ImageFont.truetype(bold if heavy else regular,size*scale),fill=color)
 def line(y): d.line((40*scale,y*scale,1160*scale,y*scale),fill='#d9e2e4',width=scale)
 text(40,28,'A guide to the models and controls',32,heavy=True)
 text(40,77,'Experiment labels used in this article; they are not standard NCA architectures.',20)
 text(40,125,'NAME / ROLE',20,heavy=True);text(290,125,'WHAT IT DOES',20,heavy=True);line(160)
 for i,(name,role,desc) in enumerate(ROWS):
  y=180+i*150
  text(40,y,name,29,heavy=True);text(40,y+43,role,18,color='#52676b')
  for j,t in enumerate(textwrap.wrap(desc,width=65)):text(290,y+j*31,t)
  line(y+130)
 text(40,1105,'G8-G10 are trained candidates. R3 is a system around G10. Neutral is an ablation control.',20)
 im.save(target,dpi=(288,288))
 with (out/'model-guide.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['Name','Role','Introduction']);w.writerows(ROWS)
 (out/'model-guide.json').write_text(json.dumps({'rows':ROWS,'png_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'dimensions':im.size},indent=2))
 print(target)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
