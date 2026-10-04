from pathlib import Path
import json,hashlib,shutil
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G7-Vertical-Training-2026-10-04';OUT=BASE/'G7-Vertical-Training-2026-10-04-v2';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
for name in ['environment.json','split-manifest.json','prepare-data.py']:shutil.copyfile(OLD/name,OUT/name)
shutil.copytree(OLD/'source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__'))
data=json.loads((OLD/'data-preparation.json').read_text());lineage=[]
for row in data['rows']:
 parent=OLD/row['arrays'];assert sha(parent.read_bytes())==row['arrays_sha256']
 with np.load(parent,allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
 arrays['damaged']=arrays['seed'].copy()
 p=OUT/row['arrays'];p.parent.mkdir(exist_ok=True)
 with p.open('xb') as f:np.savez_compressed(f,**arrays)
 lineage.append(dict(case=row['id'],parent_sha256=row['arrays_sha256'],new_sha256=sha(p.read_bytes()),change='Add damaged alias equal to independent single seed;all original array values unchanged'))
 row['arrays_sha256']=sha(p.read_bytes())
 with p.with_suffix('.json').open('x') as f:json.dump(row,f,indent=2)
with (OUT/'data-preparation.json').open('x') as f:json.dump(data,f,indent=2)
record=dict(parent=str(OLD),failed_run='20261004T102252Z_f43319ef9edf',completed_updates=0,reason='Inherited loader requires damaged;new labels supplied seed only',geometry_or_model_changed=False,lineage=lineage,old_evidence_preserved=True)
with (OUT/'packaging-correction.json').open('x') as f:json.dump(record,f,indent=2)
with (OUT/'RESUME-preparation.json').open('x') as f:json.dump(dict(status='Schema correction prepared;build and rehearse v2 before approval',previous=str(OLD/'RESUME.json'),paid_run_authorized=False,repository_sync_pending=True),f,indent=2)
shutil.copyfile(__file__,OUT/'fix-packaging.py')
