"""NR2 package integrity and safe extraction; no network or Drive operations."""
from hashlib import sha256
from pathlib import Path,PurePosixPath
import json
import zipfile


def safe_name(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or str(p)!=name:
        raise ValueError('Unsafe/noncanonical archive path')
    return p


def extract_checked(archive,destination,expected_sha256):
    archive=Path(archive);destination=Path(destination)
    if sha256(archive.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Package checksum differs')
    if destination.exists():raise FileExistsError('Use a fresh extraction directory')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP entries')
        manifest=json.loads(z.read('manifest.json'))
        if set(names)!=set(manifest['files'])|{'manifest.json'}:raise ValueError('Unlisted/missing ZIP files')
        if sum(i.file_size for i in z.infolist())>100_000_000:raise ValueError('Package expands beyond 100 MB')
        for name in names:
            safe_name(name)
            if ((z.getinfo(name).external_attr>>16)&0o170000)==0o120000:raise ValueError('Symlink archive entry')
            if name!='manifest.json' and sha256(z.read(name)).hexdigest()!=manifest['files'][name]:raise ValueError('Member checksum differs')
        destination.mkdir(parents=True)
        for name in names:
            p=destination.joinpath(*safe_name(name).parts);p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(z.read(name))
    return verify_package(destination)


def verify_package(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_bytes())
    if manifest['version']!='NR2_preflight_package_v1':raise ValueError('Unknown package')
    for name,h in manifest['files'].items():
        safe_name(name);p=root.joinpath(*PurePosixPath(name).parts)
        if p.is_symlink() or not p.resolve().is_relative_to(root) or sha256(p.read_bytes()).hexdigest()!=h:
            raise ValueError('Package file changed: '+name)
    data=json.loads((root/'dataset.json').read_bytes())
    if len(data['rows'])!=81 or any(x['split']!='train' for x in data['rows']):raise ValueError('Exactly 81 TRAIN examples required')
    if len({x['case'] for x in data['rows']})!=27 or len({x['arrays'] for x in data['rows']})!=81:
        raise ValueError('Dataset case/example counts differ')
    expected_data={x['arrays'] for x in data['rows']}
    if {n for n in manifest['files'] if n.startswith('data/')}!=expected_data:raise ValueError('Unexpected data files')
    for x in data['rows']:
        if manifest['files'].get(x['arrays'])!=x['arrays_sha256']:raise ValueError('Data row checksum differs')
    return manifest,data
