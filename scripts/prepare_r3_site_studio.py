"""Prepare the site studio without modifying the reference runtimes."""
from pathlib import Path
import subprocess,sys,json,hashlib,shutil,argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',default='r3-site-studio-2026-10-08');a=p.parse_args()
 repo=Path(__file__).resolve().parents[1]
 subprocess.run([sys.executable,str(repo/'scripts/prepare_r3_scale_studio.py'),'--name',a.name],check=True)
 out=repo/'.local-artifacts'/a.name;src=repo/'experimental/r3-site-studio'
 for f in src.iterdir():
  if f.suffix in {'.py','.js','.css','.html'}:shutil.copyfile(f,out/f.name)
 scenes=json.loads((out/'scenes.json').read_text());scenes+=json.loads((src/'site-fixtures.json').read_text());(out/'scenes.json').write_text(json.dumps(scenes,indent=2),encoding='utf-8')
 identity=json.loads((out/'identity.json').read_text());identity.update(version='R3-site-studio-v1',date='2026-10-08',parent_version='R3-scale-studio-v1')
 identity['files']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in identity['files']}
 (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8');print(out)
if __name__=='__main__':main()
