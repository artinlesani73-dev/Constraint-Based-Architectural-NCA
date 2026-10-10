"""Build a separate research history UI without modifying scientific generation."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parents[1];BASE=R/'.local-artifacts/r3-private-jobs-v3-2026-10-10';OUT=R/'.local-artifacts/r3-research-history-v3-2026-10-10'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((BASE/'private-manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(BASE/n)==h,n
    shutil.copytree(BASE,OUT,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('annotations.py','research.js'):shutil.copyfile(R/'experimental/r3-private-studio'/name,OUT/'app'/name)
    p=OUT/'app/server.py';s=p.read_text(encoding='utf-8').replace('from export_bundle import bundle','from export_bundle import bundle\nimport annotations')
    s=s.replace("  if path=='/api/drafts':", "  if path.startswith('/api/annotations/'):\n   rid=path.removeprefix('/api/annotations/')\n   if len(rid)!=32 or any(c not in '0123456789abcdef' for c in rid) or not (RUNS/rid/'request.json').is_file():return self.reply(404,{'error':'Unknown run'})\n   return self.reply(200,annotations.latest(DATA_ROOT,rid))\n  if path=='/api/drafts':",1)
    s=s.replace("  if self.path=='/api/drafts':", "  if self.path.startswith('/api/annotations/'):\n   try:\n    size=int(self.headers.get('Content-Length','0'))\n    if not 0<size<=65536:raise ValueError('Notes too large')\n    value=json.loads(self.rfile.read(size))\n    if not isinstance(value,dict):raise ValueError('Invalid notes')\n    return self.reply(201,annotations.save(DATA_ROOT,self.path.removeprefix('/api/annotations/'),value))\n   except (ValueError,TypeError):return self.reply(400,{'error':'Notes invalid or changed in another tab. Reopen the run; existing text remains saved.'})\n  if self.path=='/api/drafts':",1)
    s=s.replace("'/workflow.js':'workflow.js'", "'/research.js':'research.js','/workflow.js':'workflow.js'")
    p.write_text(s,encoding='utf-8')
    p=OUT/'app/index.html';s=p.read_text(encoding='utf-8');s+='\n<script src="/research.js"></script>\n';p.write_text(s,encoding='utf-8')
    identity_path=OUT/'app/identity.json';identity=json.loads(identity_path.read_text());identity['version']='R3-research-history-v1'
    identity['files'].update({'annotations.py':'','research.js':''});identity['files']={n:sha(OUT/'app'/n) for n in identity['files']};identity_path.write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    manifest.update(version='R3-research-history-v1',files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='private-manifest.json'})
    (OUT/'private-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(OUT)
if __name__=='__main__':main()
