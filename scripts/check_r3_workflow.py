"""One browser-generated run: exact export geometry, parity and integrity guards."""
from pathlib import Path
import argparse,hashlib,io,json,sys,time,urllib.request,urllib.error,zipfile,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'.local-artifacts/r3-workflow-studio-2026-10-08'
sys.path.insert(0,str(P));from export_bundle import bundle,obj

def verify_zip(raw):
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  manifest=json.loads(z.read('export-manifest.json'));assert set(z.namelist())==set(manifest)|{'export-manifest.json'}
  for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  meta=json.loads(z.read('metadata.json'))
  for n in z.namelist():
   if not n.endswith('.obj'):continue
   label=Path(n).stem
   with np.load(io.BytesIO(z.read('evidence/'+label+'.npz'))) as a:f=a['field']
   v=[];q=[]
   for line in z.read(n).decode().splitlines():
    if line.startswith('v '):v.append(list(map(float,line.split()[1:])))
    if line.startswith('f '):q.append([int(x)-1 for x in line.split()[1:]])
   v=np.array(v);q=np.array(q);surface=0
   for axis in range(3):surface+=int(np.abs(np.diff(np.pad(f.astype(int),1),axis=axis)).sum())
   assert len(q)==surface
   xyz=np.argwhere(f)[:,::-1];s=meta['scene']['voxel_size_m']
   assert np.allclose(v.min(0),xyz.min(0)*s) and np.allclose(v.max(0),(xyz.max(0)+1)*s)
   volume=0
   for a,b,c,d in v[q]:volume+=np.dot(a,np.cross(b,c))/6+np.dot(a,np.cross(c,d))/6
   assert np.isclose(volume,f.sum()*s**3),(label,volume,f.sum()*s**3)
  return meta,len(manifest)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('run');a=parser.parse_args();out=P/'workflow-check';out.mkdir(exist_ok=False);report=dict(run=a.run,passed=False)
 try:
  folder=P/'runs'/a.run;deadline=time.monotonic()+180
  while not (folder/'manifest.json').exists():
   if time.monotonic()>deadline:raise TimeoutError('Inspect original job; do not regenerate')
   time.sleep(.5)
  assert json.loads((folder/'status.json').read_text())['state']=='completed'
  req=json.loads((folder/'request.json').read_text());assert req['seed']==2102 and req['scene']=='site-staggered-court-40'
  prior=R/'.local-artifacts/r3-site-studio-2026-10-08/runs/957327d1ee4449cf99e4c41159517e0f'
  for m in ['G10','R3']:
   for h in [64,128]:
    name=f'{m}-{h}.npz';assert np.array_equal(np.load(folder/name)['field'],np.load(prior/name)['field'])
  with urllib.request.urlopen('http://127.0.0.1:8021/api/export/'+a.run,timeout=30) as response:
   assert response.headers['Content-Type']=='application/zip';raw=response.read()
  (out/'download.zip').write_bytes(raw);meta,count=verify_zip(raw)
  assert meta['run_id']==a.run and meta['units']=='metres' and meta['request']==req
  assert meta['outputs']['R3']['accepted_both_horizons']
  # Export a prior blocked result without running or changing it.
  blocked=R/'.local-artifacts/r3-site-studio-2026-10-08/runs/c34b8f5553944f37869f06bb7d8fa4d9'
  b=bundle(blocked);bm,_=verify_zip(b);(out/'blocked-evidence.zip').write_bytes(b)
  assert not bm['outputs']['R3']['accepted_both_horizons']
  with zipfile.ZipFile(io.BytesIO(b)) as z:assert not any(n.startswith('geometry/R3') for n in z.namelist())
  corrupt=out/'corrupt-fixture';shutil.copytree(folder,corrupt);(corrupt/'request.json').write_text('{}')
  try:bundle(corrupt);raise AssertionError('Corruption accepted')
  except ValueError as error:assert 'hash differs' in str(error)
  report.update(passed=True,all_four_fields_exact=True,export_payloads=count,all_export_hashes_verified=True,obj_surface_count_bounds_signed_volume_verified=True,blocked_no_R3_geometry=True,corruption_rejected=True,download_sha256=hashlib.sha256(raw).hexdigest(),download_bytes=len(raw))
 finally:
  (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(R/'experiments/reports/R3-workflow-check-2026-10-08.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
