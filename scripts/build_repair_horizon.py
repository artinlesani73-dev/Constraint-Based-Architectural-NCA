"""Build the disarmed, TRAIN-only NR5 notebook/package locally."""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.experiments import RunStore,read_json,write_once,digest
from nca.repair_horizon import SETTINGS
from scripts.build_colab_preflight import SOURCES


def build(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'.local-artifacts/runs'/SETTINGS['dataset_run']
    if RunStore(source.parent).verify(source.name):raise ValueError('NL0 integrity failed')
    study=read_json(source/'study.json');rows=[]
    sources=[p for p in SOURCES if p!='docs/next-phase/COLAB_PREFLIGHT_PROTOCOL.md']+[
        'nca/repair_preservation.py','nca/repair_horizon.py','nca/horizon_package.py','scripts/colab_repair_horizon.py',
        'docs/next-phase/HORIZON_PROTOCOL.md']
    payload={p:(ROOT/p).read_bytes() for p in sources}
    for row in sorted((x for x in study['examples'] if x['split']=='train'),key=lambda x:(x['case'],x['damage'])):
        name='data/'+Path(row['arrays']).name;raw=(source/row['arrays']).read_bytes()
        if sha256(raw).hexdigest()!=row['arrays_sha256']:raise ValueError('Training bytes differ')
        payload[name]=raw;rows.append({k:row[k] for k in ('case','damage','split','arrays_sha256')});rows[-1]['arrays']=name
    payload['dataset.json']=json.dumps({'source_run':source.name,'source_study_sha256':digest(source/'study.json'),'rows':rows},indent=2).encode()
    payload['study.json']=json.dumps(SETTINGS,indent=2).encode()
    guide=(ROOT/'docs/next-phase/HORIZON_PROTOCOL.md').read_text(encoding='utf-8')
    payload['README.md']=guide.encode()
    m={'version':'NR5_horizon_package_v1','files':{p:sha256(b).hexdigest() for p,b in payload.items()},
       'train_rows':81,'heldout_rows':0,'gpu_job_executed':False}
    archive=out/'NCA-NR5-Horizon-Package.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('manifest.json',json.dumps(m,indent=2))
    archive_hash=digest(archive);cells=[]
    def cell(kind,s):
        d={'cell_type':kind,'id':f'nr5-{len(cells):02d}','metadata':{},'source':s.splitlines(True)}
        if kind=='code':compile(s,'nr5-cell','exec');d.update(execution_count=None,outputs=[])
        cells.append(d)
    cell('markdown','# NR5 horizon-alignment trial\n\nOne approved seed at a time. Disarmed by default. Read START-HERE before allocating a GPU. No Drive mount or automatic continuation.\n')
    cell('code',f'''from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib, zipfile, json, uuid, subprocess, sys
EXPECTED_SHA256 = {archive_hash!r}
ARCHIVE_NAME = 'NCA-NR5-Horizon-Package.zip'
uploaded = files.upload()
if set(uploaded) != {{ARCHIVE_NAME}}: raise ValueError('Select exactly the supplied NR5 ZIP')
raw = uploaded[ARCHIVE_NAME]
if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256: raise ValueError('Package checksum differs')
source = Path('/content') / ('nr5-upload-' + uuid.uuid4().hex + '.zip')
with source.open('xb') as f: f.write(raw)
PACKAGE = Path('/content') / ('nca-nr5-' + uuid.uuid4().hex)
with zipfile.ZipFile(source) as z:
    manifest = json.loads(z.read('manifest.json'))
    names = z.namelist()
    if len(names) != len(set(names)) or set(names) != set(manifest['files']) | {{'manifest.json'}}: raise ValueError('Unexpected members')
    if sum(x.file_size for x in z.infolist()) > 100_000_000: raise ValueError('Expanded size limit')
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\\\' in name or ':' in name or str(p) != name: raise ValueError('Unsafe path')
        if ((z.getinfo(name).external_attr >> 16) & 0o170000) == 0o120000: raise ValueError('Symlink member')
        if name != 'manifest.json' and hashlib.sha256(z.read(name)).hexdigest() != manifest['files'][name]: raise ValueError('Member hash differs')
    PACKAGE.mkdir()
    for name in names:
        target = PACKAGE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f: f.write(z.read(name))
subprocess.run([sys.executable, '-c', 'from nca.horizon_package import verify; verify(".")'], cwd=PACKAGE, check=True)
print('Package verified:', PACKAGE)
''')
    cell('code','''MODEL_SEED = 1201  # Only seed1201 is proposed.
APPROVED_SEED_JOB = False
if not APPROVED_SEED_JOB: raise RuntimeError('Stop until this seed job and backup plan are approved')
result = subprocess.run([sys.executable, str(PACKAGE / 'scripts/colab_repair_horizon.py'),
    '--seed', str(MODEL_SEED), '--device', 'cuda:0', '--approved-seed-job', '--seconds', '600'], cwd=PACKAGE)
print('Exit code:', result.returncode, '- download evidence next, including failures.')
''')
    cell('code','''exports = sorted((PACKAGE / 'horizon-runs').glob('*.zip'))
if not exports: raise RuntimeError('No export. Preserve runtime and send the error; do not rerun.')
for archive in exports:
    files.download(str(archive))
    files.download(str(archive.with_suffix('.receipt.json')))
print('Return ZIP and receipt for local verification; disconnect idle GPU after verified download.')
print('No additional seed or automatic retry is planned.')
''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','name':'python3','language':'python'}},'cells':cells}
    write_once(out/'NCA-NR5-Horizon.ipynb',nb)
    with (out/'START-HERE.md').open('x',encoding='utf-8') as f:f.write(guide)
    receipt={'archive':archive.name,'archive_sha256':archive_hash,'archive_bytes':archive.stat().st_size,
             'manifest_sha256':sha256(json.dumps(m,indent=2).encode()).hexdigest(),'payload_files':len(payload),
             'notebook_sha256':digest(out/'NCA-NR5-Horizon.ipynb'),'train_rows':81,'heldout_rows':0,'settings':SETTINGS}
    write_once(out/'package-receipt.json',receipt);print(json.dumps(receipt,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);build(p.parse_args().output)
