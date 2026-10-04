from pathlib import Path, PurePosixPath
import json,hashlib,zipfile,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G1-Training-2026-10-03';OUT=BASE/'G1-Portable-Paths-2026-10-03'
OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
run='20261003T172855Z_dafe9fb14084';download=Path('C:/Users/artin/Downloads')
receipt=json.loads((download/(run+'.receipt.json')).read_text());raw=(download/(run+'.zip')).read_bytes();assert sha(raw)==receipt['sha256']
with zipfile.ZipFile(download/(run+'.zip')) as z:
 evidence=json.loads(z.read('evidence-manifest.json'))
 assert len(evidence)==receipt['files'] and len(z.namelist())==len(set(z.namelist()))==len(evidence)+1
 assert set(z.namelist())==set(evidence)|{'evidence-manifest.json'}
 assert all(sha(z.read(k))==v for k,v in evidence.items())
 result=json.loads(z.read('result.json'));request=json.loads(z.read('request.json'))
 assert request['manifest_sha256']==sha((OLD/'package/manifest.json').read_bytes())
 for k in evidence:
  dest=OUT/'failed-run'/k;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(k))
for suffix in ('.zip','.receipt.json'):shutil.copyfile(download/(run+suffix),OUT/(run+suffix))
with zipfile.ZipFile(OLD/'NCA-G1-Generation-Package.zip') as z:
 payload={k:z.read(k) for k in z.namelist() if k!='manifest.json'}
original=dict(payload);data=json.loads(payload['data.json'])
bad_before=sum('\\' in r['arrays'] for r in data['rows'])
for row in data['rows']:row['arrays']=row['arrays'].replace('\\','/')
payload['data.json']=json.dumps(data,indent=2).encode()
old_verify=payload['nca/generation_package.py'].decode()
extra=''' for row in data['rows']:
  name=row['arrays']
  from pathlib import PurePosixPath
  parts=PurePosixPath(name)
  if not isinstance(name,str) or '\\\\' in name or ':' in name or parts.is_absolute() or '..' in parts.parts or name!=parts.as_posix():raise ValueError('Nonportable dataset path')
  if name not in m['files'] or m['files'][name]!=row['arrays_sha256']:raise ValueError('Dataset row missing from manifest or hash mismatch')
'''
payload['nca/generation_package.py']=old_verify.replace(' return m,data',extra+' return m,data').encode()
compile(payload['nca/generation_package.py'],'package verifier','exec')
manifest={'version':'g1_package_portable_paths_v2','files':{k:sha(v) for k,v in payload.items()}}
archive=OUT/'NCA-G1-Generation-Portable-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1
 for k,h in manifest['files'].items():assert sha(z.read(k))==h
 for row in data['rows']:
  # Exact ZIP member lookup models POSIX separators independently of Windows Path.
  assert '\\' not in row['arrays'] and sha(z.read(row['arrays']))==row['arrays_sha256']
 root=OUT/'verification-package';z.extractall(root)
scope={};exec(payload['nca/generation_package.py'],scope);scope['verify'](root)
# Negative regression: old Windows paths must fail even on a Windows machine.
valid_data=(root/'data.json').read_bytes();valid_manifest=(root/'manifest.json').read_bytes()
bad=json.loads(valid_data);bad['rows'][0]['arrays']=bad['rows'][0]['arrays'].replace('/','\\')
(root/'data.json').write_text(json.dumps(bad));bad_m=json.loads(valid_manifest);bad_m['files']['data.json']=sha((root/'data.json').read_bytes());(root/'manifest.json').write_text(json.dumps(bad_m))
try:scope['verify'](root)
except ValueError as exc:assert 'Nonportable dataset path' in str(exc)
else:raise AssertionError('Windows path accepted')
(root/'data.json').write_bytes(valid_data);(root/'manifest.json').write_bytes(valid_manifest);scope['verify'](root)
# Same exact labels, model, runner, curriculum and protocol as the prior package.
changed=sorted(k for k in payload if payload[k]!=original[k]);assert changed==['data.json','nca/generation_package.py']
oldreceipt=json.loads((OLD/'package-receipt.json').read_text());nb=json.loads((OLD/'NCA-G1-Generation.ipynb').read_text())
for c in nb['cells']:
 text=''.join(c['source']).replace(oldreceipt['sha256'],sha(archive.read_bytes())).replace(oldreceipt['archive'],archive.name)
 if c['cell_type']=='code':compile(text,'notebook','exec')
 c['source']=text.splitlines(True)
(OUT/'NCA-G1-Generation-Portable.ipynb').write_text(json.dumps(nb,indent=2))
shutil.copyfile(OLD/'PROTOCOL.md',OUT/'PROTOCOL.md')
verification=dict(failed_run=run,failed_payloads_verified=len(evidence),completed_updates=result['worker']['completed'],controlled_seconds=result['wall_seconds'],original_windows_paths=bad_before,
 corrected_rows=len(data['rows']),exact_zip_member_checks=True,original_bad_path_rejected=True,changed_package_members=changed,
 scientific_payloads_unchanged=True,new_gpu_run_started=False,package_sha256=sha(archive.read_bytes()),manifest_sha256=sha((root/'manifest.json').read_bytes()))
(OUT/'VERIFICATION.json').write_text(json.dumps(verification,indent=2))
(OUT/'package-receipt.json').write_text(json.dumps({'archive':archive.name,'sha256':sha(archive.read_bytes()),'payloads':len(payload)},indent=2))
shutil.copyfile(__file__,OUT/'fix-paths.py');print(json.dumps(verification,indent=2))
