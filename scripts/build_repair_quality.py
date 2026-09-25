"""Build the disarmed, TRAIN-only NR3 notebook/package locally."""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.experiments import RunStore,read_json,write_once,digest
from nca.repair_quality import SETTINGS
from scripts.build_colab_preflight import SOURCES


def build(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'.local-artifacts/runs'/SETTINGS['dataset_run']
    if RunStore(source.parent).verify(source.name):raise ValueError('NL0 integrity failed')
    study=read_json(source/'study.json');rows=[]
    sources=[p for p in SOURCES if p!='docs/next-phase/COLAB_PREFLIGHT_PROTOCOL.md']+[
        'nca/repair_quality.py','nca/quality_package.py','scripts/colab_repair_quality.py',
        'docs/next-phase/REPAIR_QUALITY_PROTOCOL.md']
    payload={p:(ROOT/p).read_bytes() for p in sources}
    for row in sorted((x for x in study['examples'] if x['split']=='train'),key=lambda x:(x['case'],x['damage'])):
        name='data/'+Path(row['arrays']).name;raw=(source/row['arrays']).read_bytes()
        if sha256(raw).hexdigest()!=row['arrays_sha256']:raise ValueError('Training bytes differ')
        payload[name]=raw;rows.append({k:row[k] for k in ('case','damage','split','arrays_sha256')});rows[-1]['arrays']=name
    payload['dataset.json']=json.dumps({'source_run':source.name,'source_study_sha256':digest(source/'study.json'),'rows':rows},indent=2).encode()
    payload['study.json']=json.dumps(SETTINGS,indent=2).encode()
    guide='''# NR3: first bounded repair-quality study

LOCAL PREPARATION ONLY. Do not run until the specific seed job and backup plan are
approved. Keep the NR2 notebook as historical evidence; this is a separate notebook.

Each of three fresh models (seeds1201,1202,1203) gets256 updates. One selected seed
per invocation, at most600 controlled seconds, one attempt. No longer training,
automatic next seed, package installation or Drive mount. Current admitted GPU is
Tesla T4 with the exact recorded NR2 software stack; mismatches stop for review.
GPU allocation time includes setup and idle time outside the code's job timer.
Proposed total allocation maximum60 minutes, not an automatically enforced bill.

1. After explicit notebook save/readback approval, save the new notebook only in
   project folder1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H. Editing/saving notebook changes
   in Colab needs an explicit scope too. Nothing has been uploaded by this build.
2. After the particular GPU job is approved, open that notebook in Colab, choose
   a GPU runtime, and upload NCA-NR3-Quality-Package.zip using its first code cell.
3. Set MODEL_SEED to the approved seed, set APPROVED_SEED_JOB=True and run once.
   Do not change losses, steps, thresholds or software when an error occurs.
4. Run the download cell even after an ordinary failure. Preserve ZIP and receipt.
   Return both so the assistant can verify hashes and archive locally. Disconnect
   the GPU once downloads are verified; an idle GPU may still consume credits.
5. Request a separate exact Drive save+readback batch for that results ZIP/receipt.
   Only after verified local AND Drive copies and new compute approval, start the
   next seed. Do not delete markers or re-extract the package to bypass a failed job.

There is no automatic protection against Colab deleting the whole VM before a
job export. At most one seed job is at risk; a lost/incomplete model is preserved
as a failed attempt and needs reviewed retry permission. Checkpoints do not imply
cross-VM exact recovery. Downloaded data and the notebook are different backups.

All81 training examples are included. Validation/test data stays local. After all
three final models are fixed, CPU evaluation checks the actual voxel geometry,
all nine families and frozen baselines. The first eight updates are not a quality
result. Neither lower loss nor successful execution alone passes the study.
'''
    payload['README.md']=guide.encode()
    m={'version':'NR3_quality_package_v1','files':{p:sha256(b).hexdigest() for p,b in payload.items()},
       'train_rows':81,'heldout_rows':0,'gpu_job_executed':False}
    archive=out/'NCA-NR3-Quality-Package.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('manifest.json',json.dumps(m,indent=2))
    archive_hash=digest(archive);cells=[]
    def cell(kind,s):
        d={'cell_type':kind,'id':f'nr3-{len(cells):02d}','metadata':{},'source':s.splitlines(True)}
        if kind=='code':compile(s,'nr3-cell','exec');d.update(execution_count=None,outputs=[])
        cells.append(d)
    cell('markdown','# NR3 repair-quality study\n\nOne approved seed at a time. Disarmed by default. Read START-HERE before allocating a GPU. No Drive mount or automatic continuation.\n')
    cell('code',f'''from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib, zipfile, json, uuid, subprocess, sys
EXPECTED_SHA256 = {archive_hash!r}
ARCHIVE_NAME = 'NCA-NR3-Quality-Package.zip'
uploaded = files.upload()
if set(uploaded) != {{ARCHIVE_NAME}}: raise ValueError('Select exactly the supplied NR3 ZIP')
raw = uploaded[ARCHIVE_NAME]
if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256: raise ValueError('Package checksum differs')
source = Path('/content') / ('nr3-upload-' + uuid.uuid4().hex + '.zip')
with source.open('xb') as f: f.write(raw)
PACKAGE = Path('/content') / ('nca-nr3-' + uuid.uuid4().hex)
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
subprocess.run([sys.executable, '-c', 'from nca.quality_package import verify; verify(".")'], cwd=PACKAGE, check=True)
print('Package verified:', PACKAGE)
''')
    cell('code','''MODEL_SEED = 1201  # Change only to the specifically approved seed:1201,1202,1203.
APPROVED_SEED_JOB = False
if not APPROVED_SEED_JOB: raise RuntimeError('Stop until this seed job and backup plan are approved')
result = subprocess.run([sys.executable, str(PACKAGE / 'scripts/colab_repair_quality.py'),
    '--seed', str(MODEL_SEED), '--device', 'cuda:0', '--approved-seed-job', '--seconds', '600'], cwd=PACKAGE)
print('Exit code:', result.returncode, '- download evidence next, including failures.')
''')
    cell('code','''exports = sorted((PACKAGE / 'quality-runs').glob('*.zip'))
if not exports: raise RuntimeError('No export. Preserve runtime and send the error; do not rerun.')
for archive in exports:
    files.download(str(archive))
    files.download(str(archive.with_suffix('.receipt.json')))
print('Return ZIP and receipt for local verification; disconnect idle GPU after verified download.')
print('The next seed needs verified local/Drive backups and separate approval.')
''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','name':'python3','language':'python'}},'cells':cells}
    write_once(out/'NCA-NR3-Quality-Study.ipynb',nb)
    with (out/'START-HERE.md').open('x',encoding='utf-8') as f:f.write(guide)
    receipt={'archive':archive.name,'archive_sha256':archive_hash,'archive_bytes':archive.stat().st_size,
             'manifest_sha256':sha256(json.dumps(m,indent=2).encode()).hexdigest(),'payload_files':len(payload),
             'notebook_sha256':digest(out/'NCA-NR3-Quality-Study.ipynb'),'train_rows':81,'heldout_rows':0,'settings':SETTINGS}
    write_once(out/'package-receipt.json',receipt);print(json.dumps(receipt,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);build(p.parse_args().output)
