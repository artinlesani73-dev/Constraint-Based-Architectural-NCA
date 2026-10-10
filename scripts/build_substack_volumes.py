"""Render retained transfer fields for publication; no inference or training."""
from pathlib import Path
import argparse, ast, json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
def build(out):
    out.mkdir(parents=True,exist_ok=False)
    study=ROOT/'.local-artifacts/r3-transfer-v1-2026-10-08'
    protocol=json.loads((study/'protocol.json').read_text())
    analysis=json.loads((study/'analysis.json').read_text())
    # Reuse only the established pure renderer; never execute its plate-writing module.
    tree=ast.parse((ROOT/'scripts/render_r3_transfer.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='iso')
    env={'np':np,'font':lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n*3)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<retained-renderer>','exec'),env)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
    sources={}
    def panel(d,site,method,seed,x,w):
        scene=next(s for s in protocol['sites'] if s['scene_id']==site)
        i=protocol['sites'].index(scene)
        path=study/f'slot-{i}-{seed}-{method}'/f'{method}-{seed}-128.npz'
        sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as data: field=data['field']>.5
        rec=next(o for o in analysis['observations'] if o['site']==site and o['method']==method and o['seed']==seed)
        title={'G10':'Vega-10','R3':'Lyra-03','neutral':'Neutral control'}[method]
        d.text((x+70,270),title,font=font(76),fill='#183d33')
        env['iso'](d,field,scene,(x+w/2,970),min((w-180)/(2*scene['grid_size']),15))
        d.text((x+70,1590),f"Seed {seed} | {int(field.sum()):,} occupied cells",font=font(48),fill='#53635b')
        d.text((x+70,1660),'ACCEPTED' if rec['accepted'] else 'REJECTED',font=font(52),fill='#183d33' if rec['accepted'] else '#a04429')
    configs=[('05-volume-example.png','Gross building volume / raised-obstacle site',[('transfer-raised-obstacle-40','R3',2201)],'Saved Lyra-03 output; green = generated mass, wireframe = existing context.'),('06-method-volumes.png','Same site and firing seed / three methods',[('transfer-narrow-offset-32',m,2201) for m in ['G10','R3','neutral']],'Narrow-offset site. Equal mass budgets do not guarantee equal geometry or acceptance.'),('07-seed-volumes.png','Same method and site / two firing seeds',[('transfer-raised-obstacle-40','R3',seed) for seed in [2201,2202]],'Lyra-03 on the raised-obstacle site. Geometric variation is not a quality ranking.')]
    for name,title,panels,footer in configs:
        im=Image.new('RGB',(4800,1900),'white');d=ImageDraw.Draw(im)
        d.text((70,45),title,font=font(92),fill='#183d33')
        d.text((70,165),'Actual saved fields | 128 steps | 0.8 m voxels | 24% requested occupancy',font=font(48),fill='#53635b')
        for j,args in enumerate(panels):panel(d,*args,j*4800/len(panels),4800/len(panels))
        d.text((70,1800),footer,font=font(48),fill='#53635b');im.save(out/name)
    record={'source_fields':sources,'renderer_sha256':hashlib.sha256((ROOT/'scripts/render_r3_transfer.py').read_bytes()).hexdigest(),'note':'Publication renders only. Original fields unchanged. Chosen examples are not the complete study; complete plates remain in showcase.'}
    (out/'volume-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
