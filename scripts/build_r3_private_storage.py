"""Create a new local app version; preserve release1 and all its history."""
from pathlib import Path
import hashlib, json, shutil
R = Path(__file__).resolve().parents[1]
BASE = R/'.local-artifacts/r3-local-release-2026-10-08'
OUT = R/'.local-artifacts/r3-private-storage-v2-2026-10-09'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest = json.loads((BASE/'release-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        assert sha(BASE/name) == expected, name
    OUT.mkdir(exist_ok=False)
    shutil.copytree(BASE/'app', OUT/'app', ignore=shutil.ignore_patterns('runs','drafts','__pycache__'))
    for name in ('launch.py','storage.py'):
        shutil.copyfile(R/'experimental/r3-private-studio'/name, OUT/name)
    path = OUT/'app/server.py'
    source = path.read_text(encoding='utf-8')
    old = "RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)"
    new = "DATA_ROOT=Path(os.environ['NCA_STUDIO_DATA']).resolve();RUNS=DATA_ROOT/'runs';RUNS.mkdir(exist_ok=True)"
    assert source.count(old) == 1
    source = source.replace(old,new).replace("ROOT/'drafts'", "DATA_ROOT/'drafts'")
    assert source.count("DATA_ROOT/'drafts'") == 2
    path.write_text(source,encoding='utf-8')
    identity_path = OUT/'app/identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    identity.update(version='R3-private-storage-v1',parent_version='R3-local-release-v1')
    identity['files'] = {name:sha(OUT/'app'/name) for name in identity['files']}
    identity_path.write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    scientific_unchanged = all(sha(OUT/'app'/name)==sha(BASE/'app'/name)
                              for name in identity['files'] if name.startswith(('source/','model/')))
    assert scientific_unchanged
    result = dict(version='R3-private-storage-v1',runtime=manifest['runtime'],
                  scientific_source_model_unchanged=scientific_unchanged,
                  files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    (OUT/'private-manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(OUT)
if __name__=='__main__':main()
