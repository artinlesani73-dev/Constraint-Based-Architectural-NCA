from pathlib import Path
import json,hashlib,zipfile,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');PREP=BASE/'G1-Preparation-2026-10-03';OUT=BASE/'G1-Training-2026-10-03';OUT.mkdir(exist_ok=False)
ROOT=OUT/'package';ROOT.mkdir();(ROOT/'scripts').mkdir();(ROOT/'examples').mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=BASE/'RGR1-Paired-Comparison/rehearsal'
# Preserve the tested supervisor's dependency closure, but expose only the G1 runner.
for folder in ('nca','deploy'):
 for p in (old/folder).rglob('*.py'):
  dest=ROOT/p.relative_to(old);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
shutil.copyfile(old/'scripts/colab_repair_preflight.py',ROOT/'scripts/colab_repair_preflight.py')
shutil.copyfile(PREP/'generation_data.py',ROOT/'nca/generation_data.py')
shutil.copyfile(Path(__file__).with_name('g1_session.py'),ROOT/'nca/generation_training.py')
import numpy as np
data=json.loads((PREP/'dataset.json').read_text());rows=[]
for row in data['rows']:
 if row['split']!='train':continue
 assert row['admissible'] and sha(PREP/row['arrays'])==row['arrays_sha256']
 with np.load(PREP/row['arrays'],allow_pickle=False) as a:
  p=ROOT/row['arrays'];np.savez_compressed(p,condition=a['context'],damaged=a['seed'],target=a['target'],distance=a['distance'])
 rows.append(dict(id=row['id'],split='train',arrays=row['arrays'],arrays_sha256=sha(p)))
(ROOT/'data.json').write_text(json.dumps({'rows':rows,'source_manifest_sha256':sha(PREP/'manifest.json')},indent=2))
verify='''from pathlib import Path
import json,hashlib
def verify(root):
 root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
 for name,digest in m['files'].items():
  p=(root/name).resolve()
  if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Package changed: '+name)
 data=json.loads((root/'data.json').read_text())
 if len(data['rows'])!=27 or any(r['split']!='train' for r in data['rows']):raise ValueError('TRAIN27 only')
 return m,data
'''
(ROOT/'nca/generation_package.py').write_text(verify)
source=(old/'scripts/colab_reversible_pair.py').read_text();tail=source[source.index('def main(a):'):]
tail=tail.replace('paired-runs','generation-runs').replace('paired-attempt','generation-attempt')
tail=tail.replace("'updates_per_arm':2 if a.cpu_rehearsal else 256,'arms':['CGR1','RGR1'],","'updates':3 if a.cpu_rehearsal else 256,'replayed_updates':2,")
tail=tail.replace('Paired training evidence; no heldout quality evaluation or automatic deployment.','G1 generation training evidence; no quality acceptance or automatic deployment.')
header=source[:source.index('def worker(a):')].replace('from nca.reversible_package import verify','from nca.generation_package import verify').replace('from nca.reversible_repair import SEEDS,SETTINGS','from nca.generation_training import SEEDS,SETTINGS')
worker='''def worker(a):
    if sys.stdin.readline()!='GO\\n':raise RuntimeError('Owned start required')
    threading.Thread(target=watch_parent,daemon=True).start()
    import numpy as np,torch
    from nca.generation_training import GenerationSession,equal_tree
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);status='completed';error=None;t=time.monotonic();session=None
    try:
        _,data=verify(ROOT)
        identity={'manifest_sha256':digest(ROOT/'manifest.json'),'settings':SETTINGS}
        def fresh():return GenerationSession(ROOT,data['rows'],identity,device=a.device,seed=a.seed)
        session=fresh();write_once(out/'identity.json',session.identity)
        if a.device=='cuda:0':
            expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
            if any(session.identity['runtime'][k]!=v for k,v in expected.items()):raise ValueError('Runtime changed; stop')
            torch.cuda.reset_peak_memory_stats()
        session.save(out/'checkpoint-0000.pt')
        end=3 if a.cpu_rehearsal else 256
        for update in range(1,end+1):
            trace,state=session.step();session.save(out/f'checkpoint-{update:04d}.pt')
            with (out/f'training-{update:04d}.npz').open('xb') as f:np.savez_compressed(f,state=state)
            write_once(out/f'update-{update:04d}.json',trace)
            if update in (2,3):
                expected_payload=session.payload();clone=fresh();clone.restore(out/f'checkpoint-{update-1:04d}.pt')
                replay,replay_state=clone.step()
                checks={'update':update,'full_payload_equal':equal_tree(expected_payload,clone.payload()),'state_equal':bool(np.array_equal(state,replay_state))}
                write_once(out/f'recovery-{update:04d}.json',checks)
                if not checks['full_payload_equal'] or not checks['state_equal']:raise AssertionError('Exact recovery failed')
                session=clone
            if update==end:
                with (out/'seed-only-boundary.npz').open('xb') as f:np.savez_compressed(f,**session.evaluate(0))
            if a.device=='cuda:0' and torch.cuda.max_memory_reserved()>.8*torch.cuda.get_device_properties(0).total_memory:raise MemoryError('Memory cap')
    except BaseException:error=traceback.format_exc();status='failed'
    result={'status':status,'failure':error,'completed':session.completed if session else 0,'cpu_rehearsal':a.cpu_rehearsal,'wall_seconds':time.monotonic()-t,'peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda:0' else None,'quality_evaluated':False}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1

'''
runner=header+worker+tail;compile(runner,'runner','exec');(ROOT/'scripts/colab_generation.py').write_text(runner)
protocol='''# G1 bounded generation pilot

Fresh CGR1 network; G1 seed generation semantics. TRAIN27 only from the verified
G1 preparation. One seed1201,256 retained updates plus2 exact-recovery replay
updates,64 steps per update,batch1,float32,Adam lr0.001,gradient clip1.
Alternate seed-only and teacher-stage starts,128 each; stage depth determined by
SHA256(completed-update:row-index) modulo maximum teacher distance minus one.
Teacher starts have zero hidden state. Keep CGR1 loss: positive0.5,negative1,
local3-cube volume0.25; no intact examples. Detached hard births; no topology
gradient. This is a generation baseline,not a controlled repair ablation.

ONE Tesla T4 job capped at600 controlled wall seconds including recovery replay
and output writes. Setup,upload,export,idle are extra. Exact expected stack:
Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Stop on mismatch,failed recovery,nonfinite values or reserved GPU memory>80%.
No automatic retry. Checkpoint every completed update; full evidence ZIP+receipt.
Recovery checks replay both a teacher-stage update and a seed-start update,
comparing the full payload and generated state. Completed-update recovery only;
not mid-rollout recovery or portability across different runtimes.

Freeze review now: final256 checkpoint only,CPUfloat32,firing2101,all9 G1
development requests,seed-only64-step outputs. Also report fixed128-step outputs
as a stability diagnostic,never choose the more favorable horizon or checkpoint.
No teacher stages,cleanup,reroll or teacher input during evaluation. Primary pilot
gate:9/9 pass all unchanged massing_targets_v1 families at64steps; median absolute
requested fraction error<=0.02,and maximum<=0.04. Stability gate:9/9 remain valid
at128steps and each changes occupied count by<=5% of its64-step count. Report
all failures,per-family results,teacher IoU diagnostic,volume error and timings.
These are newly preregistered engineering targets,not established quality claims
or new constraint families. MG7 reference is9/9 valid on these same requests.
No reserved labels or evaluation in this pilot; passing development does not
authorize deployment or establish generalization. MG7 remains live.

Open notebook and upload matching ZIP. Keep APPROVED_G1_JOB=False until this exact
budget is approved. After approval run once,download full ZIP and receipt even
if it fails. No Drive access,external sync,public deployment or model promotion.
'''
(ROOT/'PROTOCOL.md').write_text(protocol);(OUT/'PROTOCOL.md').write_text(protocol)
manifest={'version':'g1_package_v1','files':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=OUT/'NCA-G1-Generation-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(ROOT.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
 assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest['files'].items())
receipt={'archive':archive.name,'sha256':sha(archive),'payloads':len(manifest['files'])}
(OUT/'package-receipt.json').write_text(json.dumps(receipt,indent=2))
cells=[]
def cell(kind,text):cells.append(dict(cell_type=kind,metadata={},source=text.splitlines(True),**({'execution_count':None,'outputs':[]} if kind=='code' else {})))
cell('markdown',protocol)
cell('code',f'''from google.colab import files
from pathlib import Path
import hashlib,zipfile,io,json,uuid
uploaded=files.upload()
if len(uploaded)!=1:raise ValueError('Upload exactly the supplied G1 package ZIP')
raw=next(iter(uploaded.values()))
if hashlib.sha256(raw).hexdigest()!={receipt['sha256']!r}:raise ValueError('Wrong package bytes')
PACKAGE=Path('/content')/('nca-g1-'+uuid.uuid4().hex)
PACKAGE.mkdir()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        if not (PACKAGE/name).resolve().is_relative_to(PACKAGE.resolve()):raise ValueError('Unsafe path')
    z.extractall(PACKAGE)
m=json.loads((PACKAGE/'manifest.json').read_text())
for name,h in m['files'].items():
    if hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
print('Package verified:',PACKAGE)
''')
cell('code',"""APPROVED_G1_JOB=False
if not APPROVED_G1_JOB:raise RuntimeError('Stop until this one G1 job is approved')
import subprocess,sys,os
command=[sys.executable,str(PACKAGE/'scripts/colab_generation.py'),'--device','cuda:0','--seed','1201','--seconds','600','--approved-seed-job']
result=subprocess.run(command,cwd=PACKAGE,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
print('Exit code:',result.returncode,'Download evidence even after failure.')
""")
cell('code',"""archives=sorted((PACKAGE/'generation-runs').glob('*.zip'))
if len(archives)!=1:raise RuntimeError('Expected one evidence ZIP; inspect run directory')
files.download(str(archives[0]))
files.download(str(archives[0].with_suffix('.receipt.json')))
""")
for c in cells:
 if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
(OUT/'NCA-G1-Generation.ipynb').write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),indent=2))
shutil.copyfile(__file__,OUT/'build-package.py')
print(json.dumps(receipt))
