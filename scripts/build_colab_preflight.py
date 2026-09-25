"""Build a local, disarmed NR2 notebook and verified training-only dependency bundle."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import sys
import zipfile

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,digest,read_json,write_once

DATASET='20260925T094341Z_316cff241020'
SOURCES=['nca/__init__.py','nca/contract.py','nca/evaluation.py','nca/volumetric.py',
    'nca/experiments.py','nca/recovery.py','nca/repair_benchmark.py','nca/repair_training.py',
    'nca/repair_portable.py','nca/colab_package.py','deploy/studio_process.py',
    'scripts/colab_repair_preflight.py','docs/next-phase/COLAB_PREFLIGHT_PROTOCOL.md']


def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    store=RunStore(REPO/'.local-artifacts/runs')
    if store.verify(DATASET):raise ValueError('NL0 evidence corrupt')
    source=store.path(DATASET);study=read_json(source/'study.json')
    rows=[];payload={name:(REPO/name).read_bytes() for name in SOURCES}
    for row in sorted((x for x in study['examples'] if x['split']=='train'),key=lambda x:(x['case'],x['damage'])):
        name='data/'+Path(row['arrays']).name;raw=(source/row['arrays']).read_bytes()
        if sha256(raw).hexdigest()!=row['arrays_sha256']:raise ValueError('NL0 sample changed')
        payload[name]=raw;rows.append({k:row[k] for k in ('case','damage','split','arrays_sha256')})
        rows[-1]['arrays']=name
    if len(rows)!=81 or len({x['case'] for x in rows})!=27:raise ValueError('Wrong training split')
    dataset={'source_run':DATASET,'source_study_sha256':digest(source/'study.json'),'rows':rows,
             'sampler':'sorted case/damage; shuffled epochs without replacement using private CPU seed1203'}
    payload['dataset.json']=(json.dumps(dataset,indent=2)+'\n').encode()
    guide='''# NCA GPU restart preflight

Prepared locally. No GPU execution or Drive access has occurred.
This package checks restarting training; it does not run the longer quality study.
It contains81 training examples from27 volumes, no validation/test examples.

## Before starting

Wait for approval of this exact one-attempt job:8 unique/16 executed updates,
one GPU,600-second execution cap. The cap does not stop Colab billing for setup,
uploads/downloads or idle GPU allocation. No automatic installations, Drive mount,
sync or further training. Keep both original local files and all downloaded results.

## Files and steps after approval

1. Notebook storage needs separate permission from GPU execution. After that exact
   save action is approved, place NCA-NR2-GPU-Preflight.ipynb directly inside project
   Drive folder1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H and open that file with Colab.
   Do not use a default upload/save workflow that may create it outside this folder.
   Nothing has been uploaded by preparing these local files.
2. Select one GPU runtime. Availability/type varies. Have the ZIP ready so the
   GPU is not left idle. Use its existing PyTorch and NumPy; do not install/change
   versions if an error occurs. Send us the error/evidence instead.
3. Run the package-upload cell and choose NCA-NR2-Preflight-Package.zip. Its checksum
   is embedded in the notebook. Extraction refuses tampering and overwriting.
4. ONLY after job approval, set APPROVED_GPU_PREFLIGHT=True in the run cell and run
   it once. It checks actual CUDA availability, records the environment, runs an
   uninterrupted copy, interrupts a second copy after checkpoint4, and resumes a
   new process. An exact replay or supported-operation failure stops the experiment.
5. Run the download cell even after an ordinary failure. It downloads the evidence
   ZIP and checksum receipt, containing checkpoints, RNG/optimizer/sampler state,
   all traces, raw outputs and logs. Return both files for local verification.
6. Confirm the downloads are present on your computer before disconnecting/deleting
   the Colab runtime. A successful files.download request alone is not a verified
   local backup. Do not rerun a failed preflight: the one-attempt marker blocks it.

Colab can delete the runtime and its files. A checkpoint on that runtime alone
does not protect against whole-VM deletion. Downloaded evidence can be preserved,
but a replacement VM/GPU is not automatically compatible with a checkpoint.
Extended training needs a separately approved backup arrangement and allowance.

Expected result: result.json reports gpu_recovery_passed=true only for an actual
successful CUDA run. A CPU rehearsal is explicitly false. Exact recovery is not
proof that the generated geometry is good. A passing job does not launch more work.

References: https://research.google.com/colaboratory/faq.html and
https://docs.pytorch.org/docs/stable/notes/randomness.html .
'''
    payload['README.md']=guide.encode()
    manifest={'version':'NR2_preflight_package_v1','files':{n:sha256(b).hexdigest() for n,b in payload.items()},
              'training_rows':81,'validation_rows':0,'test_rows':0,'gpu_validated':False}
    archive=output/'NCA-NR2-Preflight-Package.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items():z.writestr(name,raw)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    checksum=digest(archive)
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True)})
    def code(text):
        compile(text,'notebook-cell','exec')
        cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(True)})
    md('# NCA GPU restart check\n\nOne approved preflight only. This notebook is disarmed and never mounts Drive.\nRead the accompanying guide before allocating a GPU. The600-second job cap does\nnot cap billing while a runtime remains connected. Download results before closing.\n')
    code(f'''# Upload the locally supplied ZIP; no Google Drive access.
from google.colab import files
from pathlib import Path
import hashlib, zipfile, uuid
EXPECTED_SHA256 = {checksum!r}
ARCHIVE_NAME = 'NCA-NR2-Preflight-Package.zip'
uploaded = files.upload()
if set(uploaded) != {{ARCHIVE_NAME}}:
    raise ValueError('Select exactly the supplied preflight ZIP')
raw = uploaded[ARCHIVE_NAME]
if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
    raise ValueError('Package checksum differs; stop')
source = Path('/content') / ('nca-upload-' + uuid.uuid4().hex + '.zip')
with source.open('xb') as f: f.write(raw)
namespace = {{}}
with zipfile.ZipFile(source) as z:
    # This bootstrap is read only AFTER verification of the entire trusted ZIP.
    exec(compile(z.read('nca/colab_package.py'), 'verified-package-bootstrap', 'exec'), namespace)
PACKAGE = Path('/content') / ('nca-nr2-' + uuid.uuid4().hex)
namespace['extract_checked'](source, PACKAGE, EXPECTED_SHA256)
print('Package verified:', PACKAGE)
''')
    md('## Run only after approving this preflight\n\nNo long training begins. Leave the flag false until the job is approved.\nAfter an error, download the evidence; do not enable nondeterministic operations or retry.\n')
    code('''APPROVED_GPU_PREFLIGHT = False
if not APPROVED_GPU_PREFLIGHT:
    raise RuntimeError('Stop here until this one GPU preflight has been approved')
import subprocess, sys, os
command = [sys.executable, str(PACKAGE / 'scripts/colab_repair_preflight.py'),
           '--device', 'cuda:0', '--allow-gpu-preflight', '--seconds', '600']
result = subprocess.run(command, cwd=PACKAGE,
    env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
print('Exit code:', result.returncode, '- download evidence next, even after failure.')
''')
    code('''# Download does not mean local verification: return ZIP and receipt for checking.
exports = sorted((PACKAGE / 'runs').glob('*.zip'))
if not exports:
    raise RuntimeError('No completed export. Keep this runtime and send the error; do not delete evidence.')
for archive in exports:
    files.download(str(archive))
    files.download(str(archive.with_suffix('.receipt.json')))
print('Confirm both files arrived locally, then disconnect the GPU runtime.')
''')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    for i,cell in enumerate(cells):cell['id']=f'nr2-{i:02d}'
    write_once(output/'NCA-NR2-GPU-Preflight.ipynb',notebook)
    with (output/'START-HERE.md').open('x',encoding='utf-8') as f:f.write(guide)
    write_once(output/'package-receipt.json',{'archive':archive.name,'archive_sha256':checksum,'bytes':archive.stat().st_size,
        'notebook_sha256':digest(output/'NCA-NR2-GPU-Preflight.ipynb'),'files':len(payload),'training_examples':81,'gpu_executed':False})
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(build(p.parse_args().output))
