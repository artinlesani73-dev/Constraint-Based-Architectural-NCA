"""Versioned local data boundary; never merge or overwrite an existing store."""
from pathlib import Path
import hashlib, json, shutil, uuid

SCHEMA = 1

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory(root):
    return {p.relative_to(root).as_posix(): digest(p)
            for category in ('runs', 'drafts')
            for p in sorted((root / category).rglob('*')) if p.is_file()}

def validate_history(root):
    for run in (root / 'runs').iterdir():
        if not run.is_dir():
            raise ValueError('Unexpected entry in runs: ' + run.name)
        status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
        if status['state'] in ('queued', 'running'):
            raise ValueError('Stop active jobs before importing history')
        manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest.items():
            path = (run / name).resolve()
            if not path.is_relative_to(run.resolve()) or digest(path) != expected:
                raise ValueError('Run evidence differs: ' + str(path))

def open_store(path, app):
    path, app = Path(path).resolve(), Path(app).resolve()
    if path == app or path.is_relative_to(app) or app.is_relative_to(path):
        raise ValueError('Data must be separate from the application directory')
    if not path.is_dir():
        raise ValueError('Initialize the data directory with --import-release first')
    schema = json.loads((path / 'store.json').read_text(encoding='utf-8'))
    if schema.get('schema') != SCHEMA:
        raise ValueError('Unsupported data schema; no automatic migration performed')
    if not all((path / name).is_dir() for name in ('runs', 'drafts')):
        raise ValueError('Incomplete data store')
    return path

def import_release(source, target, app):
    source, target = Path(source).resolve(), Path(target).resolve()
    app = Path(app).resolve()
    if target.exists():
        raise ValueError('Import destination already exists; refusing to overwrite')
    if any(target == p or target.is_relative_to(p) or p.is_relative_to(target)
           for p in (source, app)):
        raise ValueError('Import destination must be separate from source and app')
    manifest = json.loads((source / 'release-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        p = (source / name).resolve()
        if not p.is_relative_to(source) or digest(p) != expected:
            raise ValueError('Release verification failed: ' + name)
    baseline = source / 'app'
    validate_history(baseline)
    before = inventory(baseline)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name + '.import-' + uuid.uuid4().hex)
    stage.mkdir()
    # On failure retain staging evidence for inspection; never delete source.
    for category in ('runs', 'drafts'):
        shutil.copytree(baseline / category, stage / category)
    after = inventory(stage)
    if before != after or inventory(baseline) != before:
        raise ValueError('Import inventory changed; staging retained: ' + str(stage))
    validate_history(stage)
    record = dict(schema=SCHEMA, source=str(source), files=after,
                  runs=len(list((stage / 'runs').iterdir())),
                  drafts=len(list((stage / 'drafts').glob('*.json'))))
    (stage / 'import-receipt.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (stage / 'store.json').write_text(json.dumps(dict(schema=SCHEMA))+'\n', encoding='utf-8')
    stage.rename(target)
    return record
