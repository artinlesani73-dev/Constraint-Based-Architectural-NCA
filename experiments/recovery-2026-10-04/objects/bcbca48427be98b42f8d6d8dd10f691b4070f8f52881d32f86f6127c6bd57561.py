from pathlib import Path
import json,zipfile,hashlib,sys
base=Path('C:/Users/artin/Documents/Codex/outputs');repo=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');impl=base/'Reversible-Repair-Implementation-2026-10-03';sys.path[:0]=[str(impl),str(repo)]
from reversible_repair import SETTINGS
out=base/'RGR1-cu130-Preflight';out.mkdir(exist_ok=False)
old=base/'CGR3-cu130-Preflight'
s=(old/'compatibility-runner.py').read_text().replace('nca.curriculum_package','nca.reversible_package').replace('nca.curriculum_repair','nca.reversible_repair').replace('CGR3_cu130_recovery_v1','RGR1_cu130_recovery_v1').replace('before-augmented','before-replay').replace('after-augmented','after-update').replace('expected_start=session.last_start.copy()','expected_start=session.tensors(0)[0].detach().cpu().numpy()[0,0].copy()').replace('start=replay.last_start','start=replay.tensors(0)[0].detach().cpu().numpy()[0,0]').replace("'augmented_visit':expected_trace['start']['mode']=='intermediate',",'').replace('np.array_equal(expected_start,replay.last_start)','np.array_equal(expected_start,replay.tensors(0)[0].detach().cpu().numpy()[0,0])')
needle="        if not all(checks.values()):raise AssertionError('Compatibility/recovery checks failed')"
extra='''
        # Direct device-side rule probe: deletion followed by rebirth, no target.
        from nca.reversible_repair import transition,ConnectedRepair as Reversible
        from nca.connected_repair import ConnectedRepair as Control
        anchor=torch.zeros(1,1,5,5,5,dtype=torch.bool,device=a.device);anchor[:,:,2,2,2]=True
        legal=torch.ones_like(anchor);high=torch.ones_like(anchor,dtype=torch.float32)
        grown=transition(anchor,anchor,legal,legal,high)[0]
        deleted=transition(anchor,grown,legal,legal,high*0)[0]
        regrown=transition(anchor,deleted,legal,legal,high)[0]
        if not torch.equal(deleted,anchor) or not torch.equal(grown,regrown):raise AssertionError('Device delete/rebirth failed')
        write_once(out/'transition-check.json',{'delete_rebirth_exact':True})
        # One matched high-birth timing probe per model, plus captured learned output.
        occupancy,features,allowed,_=session.tensors(0);timings=[]
        for label,cls in [('CGR1',Control),('RGR1',Reversible)]:
            torch.manual_seed(1201);model=cls().float().to(a.device)
            with torch.no_grad():
                model.last.bias[0]=2.
                if a.device=='cuda:0':torch.cuda.synchronize()
                start=time.monotonic();values=model.rollout(occupancy,features,allowed,torch.Generator(device=a.device).manual_seed(2101),32)
                if a.device=='cuda:0':torch.cuda.synchronize()
                elapsed=time.monotonic()-start
            timings.append({'model':label,'seconds':elapsed,'occupied':int(values['field'].sum()),'interpretation':'Untrained forced-positive32-step probe; includes cleanup and transfers; single timing.'})
        write_once(out/'timings.json',timings)
        with (out/'learned-boundary.npz').open('xb') as f:np.savez_compressed(f,**replay.evaluate(0))
'''
s=s.replace(needle,needle+extra).replace('CGR3 cu130 bounded compatibility/recovery check; three optimizer steps including replay.','RGR1 cu130 compatibility, recovery and cleanup timing; three optimizer steps.')
payload={}
with zipfile.ZipFile(old/'NCA-CGR3-cu130-Preflight-Package.zip') as z:
 payload={n:z.read(n) for n in z.namelist() if n!='manifest.json'}
payload['nca/reversible_repair.py']=(impl/'reversible_repair.py').read_bytes()
verifier=(repo/'nca/curriculum_package.py').read_text().replace('nca.curriculum_repair','nca.reversible_repair').replace('CGR3_curriculum_package_v1','RGR1_reversible_package_v1')
payload['nca/reversible_package.py']=verifier.encode()
payload['scripts/colab_reversible_compatibility.py']=s.encode();compile(s,'runner','exec')
payload['study.json']=json.dumps(SETTINGS,indent=2).encode()
guide='''# Reversible repair GPU compatibility and timing check

One proposed T4 check,120controlled seconds maximum; setup/upload/export/idle extra.
Two updates on one TRAIN cube5 example plus restore/replay of update2: three
optimizer steps total. Exact trace,start,state and full checkpoint payload must
match. Includes a separate device deletion/rebirth probe,one untrained32-step
positive-growth timing probe for CGR1 and RGR1,and one captured learned rollout.
Timing includes CPU connectivity flood fill and device transfers. This is not
quality evaluation or a512-update paired experiment. No automatic continuation.
Expected T4,Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Changed runtime stops; deterministic algorithms remain enabled.

Open the supplied notebook; upload this package only. Leave APPROVED_SEED_JOB=False
until this exact120s check is approved,then set True and execute once. Download
full ZIP and receipt,including failures; return both. No Drive access or retry.
Do not execute other historical runners included as source dependencies.
Original CGR1 and prior experiments remain unchanged. Reversible update semantics
and CPU recovery are locally tested; GPU correctness and timing remain unverified.
'''
payload['README.md']=guide.encode();payload['docs/next-phase/RGR1_PREFLIGHT.md']=guide.encode()
m=dict(version='RGR1_reversible_package_v1',files={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()},train_rows=81,heldout_rows=0,gpu_job_executed=False,purpose='compatibility_recovery_timing_only')
a=out/'NCA-RGR1-cu130-Preflight-Package.zip'
with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED) as z:
 for n,b in payload.items():z.writestr(n,b)
 z.writestr('manifest.json',json.dumps(m,indent=2))
sha=lambda b:hashlib.sha256(b).hexdigest();oldreceipt=json.loads((old/'package-receipt.json').read_bytes());nb=json.loads((old/'NCA-CGR3-cu130-Preflight.ipynb').read_bytes())
for c in nb['cells']:
 text=''.join(c['source']).replace(oldreceipt['sha256'],sha(a.read_bytes())).replace(oldreceipt['archive'],a.name).replace('nca.curriculum_package','nca.reversible_package').replace('colab_curriculum_compatibility.py','colab_reversible_compatibility.py').replace('nca-cgr3-compat-','nca-rgr1-compat-')
 if c['cell_type']=='markdown':text=guide
 if c['cell_type']=='code':compile(text,'notebook','exec')
 c['source']=text.splitlines(True)
(out/'NCA-RGR1-cu130-Preflight.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
(out/'START-HERE.md').write_text(guide,encoding='utf-8')
(out/'package-receipt.json').write_text(json.dumps(dict(archive=a.name,sha256=sha(a.read_bytes()),manifest_sha256=sha(json.dumps(m,indent=2).encode()),notebook_sha256=sha((out/'NCA-RGR1-cu130-Preflight.ipynb').read_bytes()),payload_files=len(payload),approved=False,controlled_seconds=120),indent=2),encoding='utf-8')
# Use the newly packaged verifier for independent extraction before rehearsal.
(out/'reversible_package.py').write_text(verifier,encoding='utf-8')
print(out)
