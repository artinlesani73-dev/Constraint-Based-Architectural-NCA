from pathlib import Path
import shutil,json,hashlib
B=Path('C:/Users/artin/Documents/Codex/outputs');OLD=B/'G11-R3-Modes-2026-10-04';P=B/'G11-R3-Skins-2026-10-04';P.mkdir(exist_ok=False)
for n in ['source','model','runs','drafts']:shutil.copytree(OLD/n,P/n)
for n in ['server.py','config.json','scenes.json','data.js','index.html','studio.js','editor.js','revisit.js']:shutil.copyfile(OLD/n,P/n)
for n in ['studio_skins.js','studio_skins.css']:shutil.copyfile(n,P/n)
p=P/'studio_skins.js';p.write_text(p.read_text(encoding='utf-8').replace("history.replaceState(null", "window.history.replaceState(null"),encoding='utf-8')
p=P/'index.html';p.write_text(p.read_text(encoding='utf-8').replace('</style>','</style><link rel="stylesheet" href="studio_skins.css">').replace('</html>','<script src="studio_skins.js"></script></html>'),encoding='utf-8')
p=P/'server.py';s=p.read_text(encoding='utf-8').replace('8017','8018').replace("'/revisit.js':'revisit.js'", "'/revisit.js':'revisit.js','/studio_skins.js':'studio_skins.js','/studio_skins.css':'studio_skins.css'").replace("else 'text/javascript; charset=utf-8'", "else 'text/css; charset=utf-8' if name.endswith('css') else 'text/javascript; charset=utf-8'");p.write_text(s,encoding='utf-8')
identity=dict(version='R3-visual-skins-v1',parent=str(OLD),files={p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and not set(['runs','drafts']).intersection(p.relative_to(P).parts)})
(P/'identity.json').write_text(json.dumps(identity,indent=2));shutil.copyfile(__file__,P/'build.py');print(P)
