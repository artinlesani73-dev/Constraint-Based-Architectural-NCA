"""Materialize an immutable snapshot; verify every file before exposing it."""
from pathlib import Path
import argparse,hashlib,json,shutil

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--artifact-root',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    recovery=repo/'experiments/recovery-2026-10-04'
    index=json.loads((recovery/'index.json').read_text(encoding='utf-8'))
    if args.snapshot not in index['snapshots']:parser.error('Unknown snapshot')
    root=args.artifact_root or Path(index['artifact_root'])
    output=repo/'.local-artifacts/recovered'/args.snapshot
    files=[]
    for name,row in index['snapshots'][args.snapshot].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe snapshot path')
        source=recovery/row['tracked_object'] if 'tracked_object' in row else root/args.snapshot/rel
        if not source.is_file() or digest(source)!=row['sha256']:
            raise RuntimeError(f'Missing or changed payload: {source}. Restore the original evidence first.')
        files.append((source,rel,row))
    if args.check_only:
        print(f'Verified {len(files)} files: {args.snapshot}');return
    if output.exists():raise FileExistsError(f'Refusing to overwrite existing evidence: {output}')
    output.mkdir(parents=True)
    for source,rel,row in files:
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if digest(target)!=row['sha256']:raise RuntimeError(f'Copy verification failed: {target}')
    print(f'Restored and verified {len(files)} files: {output}')

if __name__=='__main__':main()
