"""Prepare a versioned 32/40-cell studio from verified local evidence."""
from pathlib import Path
import hashlib,json,shutil,argparse

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='r3-scale-studio-2026-10-06')
    args=parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.name):parser.error('Use lowercase letters, numbers and hyphens for the runtime name')
    repo=Path(__file__).resolve().parents[1]
    index_root=repo/'experiments/recovery-2026-10-04'
    index=json.loads((index_root/'index.json').read_text(encoding='utf-8'))
    base=repo/'.local-artifacts/recovered/G11-R3-Skins-2026-10-04'
    if not base.exists():raise RuntimeError('First restore G11-R3-Skins-2026-10-04 with scripts/restore_recovered_snapshot.py')
    for name,row in index['snapshots'][base.name].items():
        p=base/name
        if not p.is_file() or digest(p)!=row['sha256']:raise RuntimeError('Original snapshot changed: '+name)
    out=repo/'.local-artifacts'/args.name
    if out.exists():raise FileExistsError('Choose a fresh runtime name; existing evidence is never overwritten')
    shutil.copytree(base,out,ignore=shutil.ignore_patterns('__pycache__','server.log','server-error.log'))
    sources=repo/'experimental/r3-scale-studio'
    for p in sources.iterdir():
        if p.suffix in {'.py','.js','.css','.html'}:shutil.copyfile(p,out/p.name)
    scenes=json.loads((out/'scenes.json').read_text(encoding='utf-8'))
    for n in [32,40]:
        row=index['snapshots']['G11-R3-Scale-2026-10-04'][f'{n}-scene.json']
        raw=(index_root/row['tracked_object']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Scale fixture hash differs')
        scenes.append(json.loads(raw))
    (out/'scenes.json').write_text(json.dumps(scenes,indent=2),encoding='utf-8')
    files={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and (p.relative_to(out).parts[0] in {'source','model'} or p.parent==out and p.name in {'server.py','index.html','studio.js','editor.js','revisit.js','studio_skins.js','studio_skins.css','data.js','scenes.json','config.json'})}
    identity={'version':'R3-scale-studio-v1','parent_snapshot':base.name,'date':'2026-10-06','grids':[32,40],'voxel_size_m':.8,'files':files}
    (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print(out)

if __name__=='__main__':main()
