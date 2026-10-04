from pathlib import Path
import json,hashlib
def verify(root):
 root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
 for name,digest in m['files'].items():
  p=(root/name).resolve()
  if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Package changed: '+name)
 data=json.loads((root/'data.json').read_text())
 if len(data['rows'])!=27 or any(r['split']!='train' for r in data['rows']):raise ValueError('TRAIN27 only')
 return m,data
