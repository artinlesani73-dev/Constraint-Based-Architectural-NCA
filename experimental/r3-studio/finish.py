from pathlib import Path
import json,hashlib,shutil
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Skins-2026-10-04')
rev=P/'implementation-history';rev.mkdir(exist_ok=True)
for n in ['studio_skins.js','studio_skins.css','identity.json']:shutil.copyfile(P/n,rev/n)
p=P/'studio_skins.js';s=p.read_text(encoding='utf-8').replace("wire?(dark?'#84d8d4':'#26798b')", "wire?`rgb(${rgb})`");p.write_text(s,encoding='utf-8')
p=P/'studio_skins.css';p.write_text(p.read_text(encoding='utf-8')+'\nbody[data-skin="porcelain"] .status[data-failed="true"]{color:#97441f!important}\n',encoding='utf-8')
ident=json.loads((P/'identity.json').read_text());
for n in ['studio_skins.js','studio_skins.css']:ident['files'][n]=hashlib.sha256((P/n).read_bytes()).hexdigest()
(P/'identity.json').write_text(json.dumps(ident,indent=2));shutil.copyfile(__file__,P/'finish.py')
