"""Celestial publication aliases; original research identities remain unchanged."""
from pathlib import Path
import argparse,csv,json,hashlib,textwrap
from PIL import Image,ImageDraw,ImageFont
JOURNEY=[
('Vega-01','G1 / Seed growth','First learned generation of building volume from a small seed.'),
('Vega-02','G2 / Growth signal','Rebalanced the training loss to encourage growth; volume control remained a problem.'),
('Vega-03','G3 / Volume budget','Added remaining-volume feedback, a volume-band penalty and a hard admission cap.'),
('Vega-04','G4 / Cube growth','Grew overlapping 3 x 3 x 3 cubes instead of individual cells, addressing thin additions.'),
('Vega-05','G5 / Destination cues','Added distance and reachability cues toward the opposite interface; did not improve the tested result.'),
('Vega-06','G6 / Paced growth','Released the volume allowance gradually so growth did not spend its budget too early.'),
('Vega-07','G7 / Vertical variety','Broadened training examples to include more vertical variation; accuracy and stability regressed.')]
METHODS=[
('Vega-08','G8 / Trained baseline','Increased training exposure on the broader dataset. Uses frontier classification, soft volume discrepancy and a global volume band.'),
('Vega-09','G9 / Access ranking','Adds a learning signal favoring growth that advances an access connection. Both compared score groups receive ranking gradients.'),
('Vega-10','G10 / Gradient variant','Uses the same ranking formula, but treats other-candidate scores as a fixed reference in that auxiliary gradient. Separately trained.'),
('Lyra-03','G11-R3 / Hybrid system','Reuses frozen Vega-10 weights with route planning, coverage coordination and growth guards. No additional training-loss curve.'),
('Neutral','Procedural control','Keeps the Lyra-03 planner and guards, with constant positive occupancy logits and zero hidden updates. No learned signal drives growth.')]
def build(out):
 out.mkdir(parents=True,exist_ok=True)
 for name,title,subtitle,rows,footer in [
 ('00a-research-journey','The research journey: Vega-01 to Vega-07','Earlier experiments, including directions that did not improve the result.',JOURNEY,'Protocols evolved across stages. This is a history, not a shared benchmark leaderboard.'),
 ('00b-methods-compared','The methods compared in this article','Public names are paired with the original experiment IDs.',METHODS,'Lyra-03 reuses Vega-10 weights. Neutral is a control, not another trained model.')]:
  target=out/(name+'.png')
  if target.exists():raise FileExistsError('Use a new output directory.')
  scale=4;W=1200;H=240+150*len(rows)
  im=Image.new('RGB',(W*scale,H*scale),'white');d=ImageDraw.Draw(im)
  def text(x,y,s,size=23,heavy=False,color='#24383c'):
   d.text((x*scale,y*scale),s,font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if heavy else 'C:/Windows/Fonts/arial.ttf',size*scale),fill=color)
  def line(y):d.line((40*scale,y*scale,1160*scale,y*scale),fill='#d9e2e4',width=scale)
  text(40,28,title,32,True);text(40,77,subtitle,20)
  text(40,125,'NAME / ORIGINAL ID',20,True);text(320,125,'SHORT INTRODUCTION',20,True);line(160)
  for i,(label,role,desc) in enumerate(rows):
   y=180+i*150;text(40,y,label,29,True);text(40,y+43,role,18,color='#52676b')
   for j,t in enumerate(textwrap.wrap(desc,width=61)):text(320,y+31*j,t)
   line(y+130)
  text(40,H-48,footer,19);im.save(target,dpi=(288,288))
  with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
   w=csv.writer(f);w.writerow(['Name','Original ID / role','Introduction']);w.writerows(rows)
  (out/(name+'.json')).write_text(json.dumps({'rows':rows,'png_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'dimensions':im.size},indent=2))
  print(target)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
