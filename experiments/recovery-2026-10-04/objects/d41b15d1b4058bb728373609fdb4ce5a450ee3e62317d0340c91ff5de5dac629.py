from pathlib import Path
import json,hashlib
def verify(root):
 root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
 for name,digest in m['files'].items():
  p=(root/name).resolve()
  if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Package changed: '+name)
 data=json.loads((root/'data.json').read_text())
 if len(data['rows'])!=27 or any(r['split']!='train' for r in data['rows']):raise ValueError('TRAIN27 only')
 for row in data['rows']:
  name=row['arrays']
  from pathlib import PurePosixPath
  parts=PurePosixPath(name)
  if not isinstance(name,str) or '\\' in name or ':' in name or parts.is_absolute() or '..' in parts.parts or name!=parts.as_posix():raise ValueError('Nonportable dataset path')
  if name not in m['files'] or m['files'][name]!=row['arrays_sha256']:raise ValueError('Dataset row missing from manifest or hash mismatch')
 return m,data
