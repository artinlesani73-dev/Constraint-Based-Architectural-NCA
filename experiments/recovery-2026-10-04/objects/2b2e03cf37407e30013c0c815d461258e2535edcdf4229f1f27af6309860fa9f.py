from pathlib import Path
import json,shutil,hashlib
B=Path('C:/Users/artin/Documents/Codex/outputs');OLD=B/'G11-R3-Revisit-2026-10-04';P=B/'G11-R3-Modes-2026-10-04';P.mkdir(exist_ok=False)
for n in ['source','model','runs','drafts']:shutil.copytree(OLD/n,P/n)
for n in ['config.json','scenes.json','data.js','index.html','studio.js','editor.js','revisit.js','server.py']:shutil.copyfile(OLD/n,P/n)
shutil.copyfile(B/'G11-R3-Route-Options-2026-10-04/source/r3_route_options.py',P/'source/r3_route_options.py')
def edit(n,f):
 p=P/n;p.write_text(f(p.read_text(encoding='utf-8')),encoding='utf-8')
edit('server.py',lambda s:s.replace('8016','8017').replace('from context_route import route_from_context','from context_route import route_from_context\nfrom r3_route_options import route_via').replace("route=route_from_context(c);W=None", "mode=request.get('route_mode','original');route,route_meta=(route_from_context(c),{}) if mode=='original' else route_via(c,mode);write(p/'route.json',dict(mode=mode,details=route_meta));W=None").replace("result=dict(id=runid,", "result=dict(route_mode=mode,id=runid,").replace("   parent=req.pop('parent_run',None)","   mode=req.pop('route_mode','original')\n   if mode not in ['original','low_y','high_y']:raise ValueError('Unknown route mode')\n   parent=req.pop('parent_run',None)").replace("  if parent is not None:req['parent_run']=parent", "  req['route_mode']=mode\n  if parent is not None:req['parent_run']=parent"))
edit('studio.js',lambda s:s.replace('<button id="generate">','<label>Route mode<select id="routeMode"><option value="original">Original R3 · default</option><option value="low_y">Low Y · experimental</option><option value="high_y">High Y · experimental</option></select></label><button id="generate">').replace("seed,...(window.parentRun", "seed,route_mode:$('routeMode').value,...(window.parentRun").replace(' · seed ${r.seed} ·',' · ${r.route_mode||"original"} · seed ${r.seed} ·').replace(' · seed ${r.request.seed} ·',' · ${r.request.route_mode||"original"} · seed ${r.request.seed} ·'))
edit('revisit.js',lambda s:s.replace("$('seed').value=q.seed;", "$('routeMode').value=q.route_mode||'original';$('seed').value=q.seed;").replace("seed:$('seed').value,", "route_mode:$('routeMode').value,seed:$('seed').value,"))
edit('index.html',lambda s:s.replace('Experimental hybrid · CPU generation','Experimental route options · CPU').replace('Local experimental preview · MG7', 'Route alternatives alter the procedural planner, not the trained weights. Availability and quality may differ by site. Original R3 remains the default. Local experimental preview · MG7'))
shutil.copyfile(__file__,P/'build.py')
identity=dict(version='R3-route-modes-v1',parent=str(OLD),files={p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and not set(['runs','drafts']).intersection(p.relative_to(P).parts)})
(P/'identity.json').write_text(json.dumps(identity,indent=2));(P/'RESUME.md').write_text('Route selector implemented; bounded transfer and UI verification pending. Original default retained.\n')
print(P)
