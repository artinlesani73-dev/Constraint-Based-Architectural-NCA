from pathlib import Path
import shutil,json,hashlib
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G11-R3-Studio-2026-10-04';P=BASE/'G11-R3-Editor-2026-10-04';P.mkdir(exist_ok=False)
for n in ['source','model']:shutil.copytree(OLD/n,P/n)
for n in ['config.json','scenes.json','data.js','index.html','studio.js']:shutil.copyfile(OLD/n,P/n)
s=(OLD/'server.py').read_text();s=s.replace('8014','8015').replace('from nca.contract import entrance_masks','from nca.contract import entrance_masks,validate_scene')
s=s.replace("s=byid[request['scene']]", "s=request.get('custom_scene') or byid[request['scene']]")
s=s.replace("'/studio.js':'studio.js'", "'/studio.js':'studio.js','/editor.js':'editor.js'")
s=s.replace('0<size<=4096','0<size<=16384')
s=s.replace("if set(req)!= {'scene','volume','seed'}", "if set(req) not in [{'scene','volume','seed'},{'scene','volume','seed','custom_scene'}]")
s=s.replace("  except (ValueError,TypeError,KeyError):return self.reply(400,{'error':'Invalid site, volume or seed.'})", """   if 'custom_scene' in req:
    custom=req['custom_scene']
    if not isinstance(custom,dict) or custom.get('grid_size')!=32 or custom.get('voxel_size_m')!=.8 or custom.get('street_levels')!=6 or custom.get('ceiling_z') is not None or custom.get('legacy_relaxations')!=[]:raise ValueError('Custom sites retain the32-cell grid,0.8m voxels,6-cell street band and no relaxations.')
    if len(custom.get('buildings',[])) not in [2,3] or len(custom.get('entrances',[]))!=2:raise ValueError('Use two facing buildings, two connections and at most one obstacle.')
    custom=validate_scene(custom)
    left,right=custom['buildings'][:2];west,east=custom['entrances']
    if left['side']!='left' or right['side']!='right' or left['x'][1]>=right['x'][0] or left['gap_facing_x']!=left['x'][1] or right['gap_facing_x']!=right['x'][0]:raise ValueError('Buildings must face across a positive X gap.')
    if west['id']!='E_west' or east['id']!='E_east' or west['x']!=left['x'][1] or east['x']!=right['x'][0]-2 or any(e['kind']!='facade' or e['extent']!=2 for e in [west,east]):raise ValueError('Use the two supported facade connections.')
    for b,e in [(left,west),(right,east)]:
     if not(b['y'][0]<=e['y'] and e['y']+2<=b['y'][1] and b['z'][0]<=e['z'] and e['z']+2<=b['z'][1]):raise ValueError('Each connection must fit fully on its building facade.')
    req['custom_scene']=custom
  except (ValueError,TypeError,KeyError,AttributeError) as error:return self.reply(400,{'error':str(error) or 'Invalid site, volume or seed.'})""")
s=s.replace("write(p/'scene.json',byid[req['scene']])", "write(p/'scene.json',req.get('custom_scene') or byid[req['scene']])")
(P/'server.py').write_text(s,encoding='utf-8')
j=(P/'studio.js').read_text();j=j.replace("seed})", "seed,...(window.editedScene?{custom_scene:window.editedScene()}: {})})");(P/'studio.js').write_text(j,encoding='utf-8')
h=(P/'index.html').read_text(encoding='utf-8').replace('local generation studio','custom-site studio').replace('</html>','<script src="editor.js"></script></html>');(P/'index.html').write_text(h,encoding='utf-8')
shutil.copyfile('r3_editor.js',P/'editor.js')
identity=dict(version='R3-custom-site-studio-v1',parent=str(OLD),files={p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file()})
(P/'identity.json').write_text(json.dumps(identity,indent=2));shutil.copyfile(__file__,P/'build.py')
(P/'RESUME.md').write_text('Custom editor implementation prepared; integration and invalid scene checks pending. No promotion. Prior: ../G11-R3-Studio-2026-10-04/RESUME.md.\n')
print(P)
