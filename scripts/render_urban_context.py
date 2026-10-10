"""Plan diagram of the frozen representation, not generated building geometry."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]
OUT=Path('C:/Users/artin/Documents/Codex/outputs/NCA-Urban-Context-2026-10-10')
def main():
    OUT.mkdir(exist_ok=False)
    scene=json.loads((R/'.local-artifacts/urban-context-v1-2026-10-10/scene.json').read_text())
    image=Image.new('RGB',(1120,980),'#172025');draw=ImageDraw.Draw(image)
    regular=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
    title=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',30)
    draw.text((45,28),'URBAN CONTEXT / BLOCK 01',font=title,fill='#edf1e8')
    draw.text((45,76),'Representation only - no generated mass or trained-model result',font=regular,fill='#bbc5c4')
    scale=7;ox=65;oy=815
    def rect(b):return (ox+b['x'][0]*scale,oy-b['y'][1]*scale,ox+b['x'][1]*scale,oy-b['y'][0]*scale)
    draw.rectangle((ox,oy-96*scale,ox+96*scale,oy),outline='#697677',width=2)
    for i in range(0,97,12):
        x=ox+i*scale;y=oy-i*scale
        draw.line((x,oy-96*scale,x,oy),fill='#2c383d');draw.line((ox,y,ox+96*scale,y),fill='#2c383d')
    for key,color in [('opportunity_regions','#254e51'),('protected_regions','#72563a'),('buildings','#758181')]:
        for entry in scene[key]:
            bounds=rect(entry['bounds']);draw.rectangle(bounds,fill=color,outline='#bdc9bf',width=2)
            draw.text((bounds[0]+6,bounds[1]+6),entry['id'],font=regular,fill='#f2f2e8')
    buildings={b['id']:b for b in scene['buildings']};points={}
    for entry in scene['interfaces']:
        b=buildings[entry['building']]['bounds'];axis=entry['axis']
        plane=b[axis][1] if entry['side']=='max' else b[axis][0]
        other='y' if axis=='x' else 'x';span=entry['spans'][other]
        x,y=(plane,sum(span)/2) if axis=='x' else (sum(span)/2,plane)
        points[entry['id']]=(ox+x*scale,oy-y*scale)
    for link in scene['connections']:
        draw.line((*points[link['source']],*points[link['target']]),fill='#f2cc70',width=3)
    for x,y in points.values():draw.ellipse((x-6,y-6,x+6,y+6),fill='#ffe5a1')
    for i,(label,color) in enumerate([('Fixed buildings','#758181'),('Growth regions','#254e51'),('Protected courtyard','#72563a'),('Requested links','#f2cc70')]):
        draw.rectangle((790,180+i*60,810,200+i*60),fill=color)
        draw.text((825,175+i*60),label,font=regular,fill='#edf1e8')
    for i,text in enumerate(['XYZ: 96 x 96 x 40 cells','Voxel edge: 0.8 m','Extent: 76.8 x 76.8 x 32 m','3 buildings / 2 growth regions','3 facade interfaces / 2 links']):
        draw.text((760,475+i*42),text,font=regular,fill='#d8e0d8')
    draw.text((65,850),'Plan: X right, Y up. Z is vertical; individual heights are stored in the scene.',font=regular,fill='#bbc5c4')
    draw.text((65,892),'Yellow lines express requested relationships, not certified routes or proposed bridges.',font=regular,fill='#bbc5c4')
    image.save(OUT/'block-plan.png');print(OUT/'block-plan.png')
if __name__=='__main__':main()
