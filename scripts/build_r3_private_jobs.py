"""Versioned worker overlay over the verified private storage app."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parents[1]
BASE=R/'.local-artifacts/r3-private-storage-v2-2026-10-09'
OUT=R/'.local-artifacts/r3-private-jobs-v3-2026-10-10'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((BASE/'private-manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(BASE/n)==h,n
    OUT.mkdir(exist_ok=False)
    shutil.copytree(BASE/'app',OUT/'app',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('launch.py','storage.py'):shutil.copyfile(BASE/name,OUT/name)
    for name in ('jobs.py','worker.py'):shutil.copyfile(R/'experimental/r3-private-studio'/name,OUT/name)
    path=OUT/'launch.py';s=path.read_text(encoding='utf-8')
    s=s.replace('from storage import import_release, open_store','from storage import import_release, open_store\nfrom jobs import ownership, recover')
    old="    runpy.run_path(str(ROOT/'app/server.py'), run_name='__main__')"
    new="    with ownership(data/'studio.lock'):\n        with ownership(data/'worker.lock'):\n            recover(data)\n        runpy.run_path(str(ROOT/'app/server.py'), run_name='__main__')"
    assert s.count(old)==1;s=s.replace(old,new);path.write_text(s,encoding='utf-8')
    path=OUT/'app/server.py';s=path.read_text(encoding='utf-8')
    begin=s.index("for p in RUNS.glob('*/status.json'):");end=s.index('\ndef generate(',begin)
    s=s[:begin]+s[end:]
    old="threading.Thread(target=generate,args=(rid,req),daemon=True).start()"
    assert s.count(old)==1
    s=s.replace(old,"__import__('jobs').start(rid,req,lock,ROOT,DATA_ROOT)")
    path.write_text(s,encoding='utf-8')
    identity_path=OUT/'app/identity.json';identity=json.loads(identity_path.read_text())
    identity.update(version='R3-private-jobs-v1',parent_version='R3-private-storage-v1')
    identity['files']={n:sha(OUT/'app'/n) for n in identity['files']}
    identity_path.write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    for n in identity['files']:
        if n.startswith(('source/','model/')):assert sha(OUT/'app'/n)==sha(BASE/'app'/n)
    manifest.update(version='R3-private-jobs-v1',files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
    (OUT/'private-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(OUT)
if __name__=='__main__':main()
