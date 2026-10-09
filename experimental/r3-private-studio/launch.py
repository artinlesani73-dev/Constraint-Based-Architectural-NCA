"""Local-only studio with explicitly selected persistent research storage."""
from pathlib import Path
import argparse, hashlib, json, os, platform, runpy, socket, sys
from storage import import_release, open_store
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--import-release', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--port', type=int, default=8023)
    a = p.parse_args()
    if not 1024 <= a.port <= 65535:
        p.error('Port must be between 1024 and65535')
    manifest = json.loads((ROOT/'private-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('App file differs: '+name)
    import numpy, torch
    actual = dict(python=platform.python_version(), numpy=numpy.__version__, torch=torch.__version__)
    if actual != manifest['runtime']:
        raise ValueError('Use the documented tested Python environment: '+str(manifest['runtime']))
    if a.import_release:
        record = import_release(a.import_release, a.data_dir, ROOT/'app')
        print('Imported without modifying source: '+str(record['runs'])+' runs, '+str(record['drafts'])+' drafts', flush=True)
    data = open_store(a.data_dir, ROOT/'app')
    print('Research data: '+str(data), flush=True)
    if a.check:
        return
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', a.port))
    os.environ['NCA_STUDIO_PORT'] = str(a.port)
    os.environ['NCA_STUDIO_DATA'] = str(data)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT/'app'))
    print(f'Open http://127.0.0.1:{a.port}/#skin=graphite', flush=True)
    runpy.run_path(str(ROOT/'app/server.py'), run_name='__main__')

if __name__ == '__main__':
    main()
