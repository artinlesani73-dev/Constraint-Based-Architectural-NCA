from pathlib import Path
import hashlib,json,zipfile,shutil
import numpy as np
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Variety-2026-10-04');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((P/'freeze.json').read_text());assert all(sha(P/n)==h for n,h in frozen.items() if n!='RESUME.json')
r=json.loads((P/'result.json').read_text());assert len(r['records'])==48
for rec in r['records']:
 if rec['status']!='evaluated':continue
 p=P/'cases'/rec['scene'];label=rec['model'];seed=rec['seed'];step=rec['steps']
 with np.load(p/f'{label}-{seed}-{step}.npz') as a:f=a['field']
 assert int(f.sum())==rec['score']['occupied_voxels']
 with np.load(p/f'{label}-{seed}-trajectory.npz') as a:
  for n in a.files:assert np.isfinite(a[n]).all()
  b=a['births'];assert b.shape[0]==128 and int(b.sum(0).max())<=1
  if label=='R3':assert np.array_equal(b,a['provenance']>0) and np.array_equal(f,a[f'state{step}'][0,0].astype(bool))
(P/'audit.json').write_text(json.dumps(dict(frozen_hashes_unchanged=True,records=48,finite_trajectories=True,unique_voxel_births=True,r3_provenance_and_states_match=True,visual_review='All 24 final fields inspected on comparison.png'),indent=2),encoding='utf-8')
(P/'RESUME.json').write_text(json.dumps(dict(status='exploratory assessment complete and visually reviewed',next='R3 local generation endpoint with persistent evidence and explicit failure reporting; retain raw G10 comparison. See REVIEW.md for geometry limits.',live_model='MG7 unchanged',previous='../G11-R3-Preview-2026-10-04/RESUME.md',repo_sync_pending=True,off_device_backup_pending=True),indent=2),encoding='utf-8')
shutil.copyfile(__file__,P/'close.py')
files={p.relative_to(P).as_posix():sha(p) for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(P/'manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8');files['manifest.json']=sha(P/'manifest.json')
zpath=P.with_suffix('.verified.zip')
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(P/n,n)
with zipfile.ZipFile(zpath) as z:
 assert len(z.namelist())==len(files)
 assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
receipt=dict(sha256=sha(zpath),bytes=zpath.stat().st_size,payloads=len(files),verified=True)
zpath.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(receipt)
