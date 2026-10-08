"""Local release launcher: verify immutable app and included evidence before serving."""
from pathlib import Path
import argparse,hashlib,json,os,platform,runpy,socket,sys
ROOT=Path(__file__).resolve().parent

def verify():
 manifest=json.loads((ROOT/'release-manifest.json').read_text(encoding='utf-8'))
 for name,h in manifest['files'].items():
  p=ROOT/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('Missing or changed release file: '+name)
 import numpy,torch
 actual=dict(python=platform.python_version(),numpy=numpy.__version__,torch=torch.__version__)
 if actual!=manifest['runtime']:raise RuntimeError('Runtime differs from tested versions. Expected '+str(manifest['runtime'])+'; found '+str(actual)+'. Use the documented project environment.')
 print('Verified release files and included saved history.',flush=True)
 return manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8022);p.add_argument('--check',action='store_true');a=p.parse_args()
 if not 1024<=a.port<=65535:p.error('Choose a port from 1024 to65535')
 verify()
 if a.check:return
 with socket.socket() as probe:
  try:probe.bind(('127.0.0.1',a.port))
  except OSError:raise RuntimeError('Local port is busy or inaccessible. Choose --port 8023. No existing server was stopped.')
 os.environ['NCA_STUDIO_PORT']=str(a.port);sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'app'))
 print(f'Open http://127.0.0.1:{a.port}/#skin=graphite',flush=True)
 print('Keep this terminal open. Press Ctrl+C to stop. New runs and drafts stay in app/.',flush=True)
 runpy.run_path(str(ROOT/'app/server.py'),run_name='__main__')
if __name__=='__main__':
 try:main()
 except KeyboardInterrupt:print('Studio stopped. Saved files retained.')
 except Exception as e:print('Cannot start: '+str(e),file=sys.stderr);sys.exit(1)
