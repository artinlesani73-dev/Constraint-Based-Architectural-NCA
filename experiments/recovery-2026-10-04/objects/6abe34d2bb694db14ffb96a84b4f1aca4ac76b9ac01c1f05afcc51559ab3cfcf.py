from pathlib import Path
import json,shutil,hashlib
B=Path('C:/Users/artin/Documents/Codex/outputs');OLD=B/'G11-R3-Editor-2026-10-04';P=B/'G11-R3-Revisit-2026-10-04';P.mkdir(exist_ok=False)
for n in ['source','model','runs']:shutil.copytree(OLD/n,P/n)
for n in ['config.json','scenes.json','data.js','index.html','studio.js','editor.js','server.py']:shutil.copyfile(OLD/n,P/n)
def edit(n,f):
 p=P/n;p.write_text(f(p.read_text(encoding='utf-8')),encoding='utf-8')
edit('server.py',lambda s:s.replace('8015','8016').replace("if path=='/api/scenes':", "if path=='/api/drafts':return self.reply(200,[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'drafts').glob('*.json'),reverse=True)])\n  if path=='/api/scenes':").replace("status=json.loads((p/'status.json').read_text()),result=", "request=json.loads((p/'request.json').read_text()),status=json.loads((p/'status.json').read_text()),result=").replace("'/editor.js':'editor.js'", "'/editor.js':'editor.js','/revisit.js':'revisit.js'").replace("  if self.path!='/api/generate':", """  if self.path=='/api/drafts':
   try:
    size=int(self.headers.get('Content-Length','0'))
    if not 0<size<=16384:raise ValueError('Draft too large')
    value=json.loads(self.rfile.read(size))
    if not isinstance(value,dict) or value.get('site') not in byid or not isinstance(value.get('values'),dict) or not isinstance(value.get('base'),dict):raise ValueError('Invalid draft')
    rid=uuid.uuid4().hex;folder=ROOT/'drafts';folder.mkdir(exist_ok=True);saved=dict(id=rid,saved_at=now(),draft=value);write(folder/(rid+'.json'),saved);return self.reply(201,saved)
   except (ValueError,TypeError):return self.reply(400,{'error':'Invalid draft. Nothing saved.'})
  if self.path!='/api/generate':""").replace("   if set(req) not in", """   parent=req.pop('parent_run',None)
   if parent is not None:
    if not isinstance(parent,str) or len(parent)!=32 or any(c not in '0123456789abcdef' for c in parent) or not (RUNS/parent/'request.json').exists():raise ValueError('Parent run not found')
   if set(req) not in""").replace("  if not lock.acquire", "  if parent is not None:req['parent_run']=parent\n  if not lock.acquire"))
edit('studio.js',lambda s:s.replace("seed,...(window.editedScene", "seed,...(window.parentRun?{parent_run:window.parentRun}:{}),...(window.editedScene").replace("async function watch(id){", "async function watch(id){window.selectedRun=id;").replace("if(v.result)show(v.result);", "if(v.result)show(v.result);if(window.revisitRunSelected)window.revisitRunSelected(id);"))
edit('editor.js',lambda s:s.replace("s.scene_id='custom-'+draft.scene_id", "s.scene_id=draft.scene_id.startsWith('custom-')?draft.scene_id:'custom-'+draft.scene_id").replace("$('site').addEventListener('change',()=>loadDraft()", "$('site').addEventListener('change',()=>{window.parentRun=null;return loadDraft() ").replace("textContent=e.message));loadDraft()", "textContent=e.message)});loadDraft()"))
edit('index.html',lambda s:s.replace('</html>','<script src="revisit.js"></script></html>'))
shutil.copyfile('r3_revisit.js',P/'revisit.js');shutil.copyfile(__file__,P/'build.py')
ident=dict(version='R3-revisit-v1',parent=str(OLD),files={p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and 'runs' not in p.relative_to(P).parts})
(P/'identity.json').write_text(json.dumps(ident,indent=2),encoding='utf-8');(P/'RESUME.md').write_text('Prepared revisit/draft workflow; validation pending. Previous milestone ../G11-R3-Editor-2026-10-04/RESUME.md. No scientific source change.\n')
print(P)
