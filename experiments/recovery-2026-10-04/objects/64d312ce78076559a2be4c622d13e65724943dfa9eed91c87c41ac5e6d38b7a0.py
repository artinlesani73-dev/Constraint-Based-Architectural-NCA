"""CGR2 package integrity; no cloud access."""
from pathlib import Path
from hashlib import sha256
import json,zipfile
from nca.colab_package import safe_name
from nca.bulk_repair import SETTINGS


def verify(root):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_bytes())
    if m['version']!='CGR2_bulk_package_v1':raise ValueError('Wrong package version')
    for name,h in m['files'].items():
        safe_name(name);p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Changed package member: '+name)
    config=json.loads((root/'study.json').read_bytes())
    if config!=SETTINGS:raise ValueError('Frozen study changed')
    data=json.loads((root/'dataset.json').read_bytes())
    rows=data['rows']
    if len(rows)!=81 or any(x['split']!='train' for x in rows) or len({x['case'] for x in rows})!=27:raise ValueError('TRAIN split differs')
    if {x['arrays'] for x in rows}!={n for n in m['files'] if n.startswith('data/')}:raise ValueError('Unexpected example arrays')
    if any(m['files'][x['arrays']]!=x['arrays_sha256'] for x in rows):raise ValueError('Dataset digest differs')
    return m,data


def extract(archive,destination,expected):
    archive=Path(archive);dest=Path(destination)
    if sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('Archive hash differs')
    if dest.exists():raise FileExistsError('Fresh extraction required')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();m=json.loads(z.read('manifest.json'))
        if len(names)!=len(set(names)) or set(names)!=set(m['files'])|{'manifest.json'}:raise ValueError('Duplicate/unlisted archive member')
        if sum(x.file_size for x in z.infolist())>100_000_000:raise ValueError('Expanded size limit')
        for name in names:
            safe_name(name)
            if ((z.getinfo(name).external_attr>>16)&0o170000)==0o120000:raise ValueError('Symlink member')
            if name!='manifest.json' and sha256(z.read(name)).hexdigest()!=m['files'][name]:raise ValueError('Member checksum differs')
        dest.mkdir(parents=True)
        for name in names:
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(z.read(name))
    return verify(dest)
