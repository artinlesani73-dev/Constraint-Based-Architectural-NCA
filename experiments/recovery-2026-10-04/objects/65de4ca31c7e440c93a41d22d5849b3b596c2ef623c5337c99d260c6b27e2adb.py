from pathlib import Path
import hashlib,json,zipfile,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G1-Portable-Paths-2026-10-03';OUT=BASE/'G2-Balanced-Growth-2026-10-03';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(OLD/'NCA-G1-Generation-Portable-Package.zip') as z:
 original={k:z.read(k) for k in z.namelist() if k!='manifest.json'}
 payload=dict(original)
s=payload['nca/connected_repair.py'].decode()
assert s.count("'frontier_positive':.5")==1 and s.count('front=(.5*F.softplus')==1
s=s.replace("VERSION='connected_constructive_repair_v2'","VERSION='connected_generation_positive_balance_v1'")
s=s.replace("'frontier_positive':.5","'frontier_positive':1.").replace('front=(.5*F.softplus','front=(1.*F.softplus')
payload['nca/connected_repair.py']=s.encode()
s=payload['nca/generation_training.py'].decode().replace('G1 fresh seed generation','G2 positive-balanced seed generation').replace("VERSION='seed_generation_training_v1'","VERSION='seed_generation_training_v2_positive_balance'")
payload['nca/generation_training.py']=s.encode()
s=payload['scripts/colab_generation.py'].decode().replace('RGR1 cu130 compatibility, recovery and cleanup timing; three optimizer steps.','G2 bounded generation pilot, with embedded exact recovery checks.').replace('G1 generation training evidence','G2 generation training evidence')
payload['scripts/colab_generation.py']=s.encode()
protocol=payload['PROTOCOL.md'].decode().replace('G1 bounded generation pilot','G2 balanced-growth comparison').replace('G1 seed generation semantics','G2 seed generation semantics').replace('Keep CGR1 loss: positive0.5,negative1,','Change only G1 positive frontier weight0.5 to1.0; retain negative1,').replace('one G1 job','one G2 job').replace('APPROVED_G1_JOB','APPROVED_G2_JOB')
protocol+='''

G1 control evidence is completed run20261003T184158Z_42fc06fd2028,not a new arm.
G2 starts fresh; no G1 checkpoint is resumed. All27 TRAIN array bytes, row order,
sampler/start schedule, initialization seed, network, optimizer, rollout horizon,
negative and volume weights, firing RNG and evaluation gates remain unchanged.
Increased positive weight may worsen false additions and stability; report them.
The same nine development requests have been reused; do not call them fresh
held-out evidence. Reserved targets stay unopened. One model seed cannot establish
robustness. Compare final checkpoints under identical CPU review settings and
report differences without selecting checkpoints,thresholds,horizons or seeds.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
 if k.endswith('.py'):compile(v,k,'exec')
data=json.loads(payload['data.json'])
for row in data['rows']:
 assert '\\' not in row['arrays'] and sha(payload[row['arrays']])==row['arrays_sha256']
assert payload['data.json']==original['data.json']
changed=sorted(k for k in payload if payload[k]!=original[k])
assert changed==['PROTOCOL.md','nca/connected_repair.py','nca/generation_training.py','scripts/colab_generation.py']
manifest={'version':'g2_positive_balance_package_v1','files':{k:sha(v) for k,v in payload.items()}}
archive=OUT/'NCA-G2-Balanced-Growth-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1
 assert all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
oldreceipt=json.loads((OLD/'package-receipt.json').read_text());nb=json.loads((OLD/'NCA-G1-Generation-Portable.ipynb').read_text())
for c in nb['cells']:
 text=''.join(c['source']).replace(oldreceipt['sha256'],sha(archive.read_bytes())).replace(oldreceipt['archive'],archive.name).replace('APPROVED_G1_JOB','APPROVED_G2_JOB').replace('one G1 job','one G2 job').replace('nca-g1-','nca-g2-')
 if c['cell_type']=='markdown':text=protocol
 else:compile(text,'notebook','exec')
 c['source']=text.splitlines(True)
(OUT/'NCA-G2-Balanced-Growth.ipynb').write_text(json.dumps(nb,indent=2))
(OUT/'PROTOCOL.md').write_text(protocol)
(OUT/'package-receipt.json').write_text(json.dumps(dict(archive=archive.name,sha256=sha(archive.read_bytes()),payloads=len(payload)),indent=2))
(OUT/'change-audit.json').write_text(json.dumps(dict(changed_payloads=changed,training_rows_unchanged=True,all27_arrays_unchanged=True,scientific_change='positive frontier weight 0.5 to1.0',manifest_sha256=sha((OUT/'package/manifest.json').read_bytes())),indent=2))
shutil.copyfile(__file__,OUT/'build-package.py');print(OUT)
