from pathlib import Path
import json,hashlib,zipfile
base=Path('C:/Users/artin/Documents/Codex/outputs');old=base/'RGR1-cu130-Preflight';out=base/'RGR1-Paired-Comparison';out.mkdir(exist_ok=False)
s=(old/'rehearsal/scripts/colab_reversible_compatibility.py').read_text();a=s.index('def worker(a):');b=s.index('\ndef main(a):')
worker='''def worker(a):
    if sys.stdin.readline()!='GO\\n':raise RuntimeError('Owned start required')
    threading.Thread(target=watch_parent,daemon=True).start()
    import numpy as np
    import torch,hashlib,gc
    from nca.connected_repair import ConnectedSession as Control
    from nca.reversible_repair import ConnectedSession as Candidate
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);status='completed';error=None;t=time.monotonic();arms={};initial_hashes={};orders={};firing_states={};session=None
    def parameter_hash(model):
        h=hashlib.sha256()
        for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    try:
        _,data=verify(ROOT)
        plan=read_json(ROOT/'paired-plan.json')
        expected_plan={'version':'RGR1_paired_comparison_v1','arms':['CGR1','RGR1'],'updates_per_arm':256,'total_updates':512,'shared_seconds_cap':600,'seed':1201,'steps':32,'automatic_retry':False,'test_evaluation':False}
        if plan!=expected_plan:raise ValueError('Paired plan differs')
        end=2 if a.cpu_rehearsal else 256
        for label,cls in [('CGR1',Control),('RGR1',Candidate)]:
            arm=out/label;arm.mkdir();start=time.monotonic()
            session=cls(ROOT,data['rows'],{'paired_plan':plan,'arm':label,'manifest_sha256':digest(ROOT/'manifest.json')},device=a.device,seed=a.seed)
            write_once(arm/'identity.json',session.identity)
            if a.device=='cuda:0':
                expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
                if any(session.identity['runtime'][k]!=v for k,v in expected.items()):raise ValueError('Runtime changed; stop')
                torch.cuda.reset_peak_memory_stats()
            initial_hashes[label]=parameter_hash(session.model)
            session.save(arm/'checkpoint-0000.pt')
            def boundary():
                with (arm/f'boundary-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,**session.evaluate(0))
            boundary()
            while session.completed<end:
                step_start=time.monotonic();trace,state=session.step()
                session.save(arm/f'checkpoint-{session.completed:04d}.pt')
                with (arm/f'training-{session.completed:04d}.npz').open('xb') as f:np.savez_compressed(f,state=state)
                write_once(arm/f'update-{session.completed:04d}.json',dict(trace,wall_seconds=time.monotonic()-step_start))
                if session.completed%64==0 or session.completed==end:boundary()
                if a.device=='cuda:0' and torch.cuda.max_memory_reserved()>.8*torch.cuda.get_device_properties(0).total_memory:raise MemoryError('Memory cap')
            orders[label]=[x['row_index'] for x in session.trace];firing_states[label]=session.firing.get_state().cpu()
            arms[label]={'completed':session.completed,'wall_seconds':time.monotonic()-start,'initial_parameter_sha256':initial_hashes[label],'peak_reserved':torch.cuda.max_memory_reserved() if a.device=='cuda:0' else None}
            write_once(arm/'result.json',arms[label]);session=None;gc.collect()
            if a.device=='cuda:0':torch.cuda.empty_cache()
        checks={'same_initial_parameters':initial_hashes['CGR1']==initial_hashes['RGR1'],'same_row_order':orders['CGR1']==orders['RGR1'],'same_final_firing_rng':torch.equal(firing_states['CGR1'],firing_states['RGR1'])}
        write_once(out/'pairing-checks.json',checks)
        if not all(checks.values()):raise AssertionError('Pairing mismatch')
    except BaseException:error=traceback.format_exc();status='failed'
    result={'status':status,'failure':error,'arms':arms,'in_progress_completed':session.completed if session else None,'cpu_rehearsal':a.cpu_rehearsal,'wall_seconds':time.monotonic()-t,'interpretation':'Training evidence; final quality evaluated later. Partial arms are not a completed comparison.'}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1

'''
s=s[:a]+worker+s[b:];s=s.replace('<=120','<=600').replace('Maximum120s','Maximum600s').replace('default=120','default=600').replace('compatibility-runs','paired-runs').replace('compatibility-attempt','paired-attempt').replace("'updates':2,'optimizer_steps_including_replay':3,","'updates_per_arm':2 if a.cpu_rehearsal else 256,'arms':['CGR1','RGR1'],").replace('Compatibility and exact recovery check only; not a quality trial or full training run.','Paired training evidence; no heldout quality evaluation or automatic deployment.')
compile(s,'paired-worker','exec')
with zipfile.ZipFile(old/'NCA-RGR1-cu130-Preflight-Package.zip') as z:payload={n:z.read(n) for n in z.namelist() if n!='manifest.json'};m=json.loads(z.read('manifest.json'))
plan=dict(version='RGR1_paired_comparison_v1',arms=['CGR1','RGR1'],updates_per_arm=256,total_updates=512,shared_seconds_cap=600,seed=1201,steps=32,automatic_retry=False,test_evaluation=False)
payload['scripts/colab_reversible_pair.py']=s.encode();payload['paired-plan.json']=json.dumps(plan,indent=2).encode()
guide='''# Paired CGR1 and reversible repair comparison

One proposed job: CGR1 control then RGR1 candidate,256updates each,512total,
seed1201,32steps. Shared600-second controlled cap covers BOTH arms,not10minutes
each. Setup/upload/export/idle extra. T4,Torch2.11.0+cu130,Python3.13.15,
NumPy2.1.3,CUDA13.0,cuDNN92700. Any mismatch stops. No retry or extra seeds.
Original TRAIN81 only; no curriculum. Models start fresh with identical parameters,
row order and firing RNG. Different update rule and supervised set: system-level
comparison,not a pure single-factor deletion ablation. Global CPU connectivity
cleanup remains enabled and included in candidate timing. A single timing probe
cannot guarantee the pair finishes under cap; partial evidence must be retained.

Open supplied notebook,upload matching ZIP. Keep APPROVED_SEED_JOB=False until
approval for this entire512-update job. After approval,set True and run once.
Download FULL ZIP+receipt,including failures. No Drive or automatic continuation.
Do not run the older compatibility or historical training scripts in the package.
Each arm retains per-update checkpoints,state,trace and boundary captures; RGR1
captures proposals,births,direct/cleanup removals,candidate and projected fields.
No quality-model claims follow from training completion.

Frozen review: both final256checkpoints only,CPUfloat32,32steps,firing2101,
original27development inputs, no TEST or threshold/horizon search. Report accepted
projected output plus pre-cleanup candidate diagnostics. Retain all original
acceptance gates: all9intact IoU>=.99 and valid; damaged validity>=17/18,
medianIoU>=.9705768039313023,excess<=325,recovered>=1945,median absolute volume
error<=19; zero original-input removals. Report regressions against same-runtime
control,repair deletions,cleanup deletions,oscillation,walltime and memory.
MG7 stays live. One seed does not establish generalization. No automatic admission.
'''
payload['README.md']=guide.encode();payload['docs/next-phase/RGR1_PAIRED_PROTOCOL.md']=guide.encode()
sha=lambda b:hashlib.sha256(b).hexdigest();m['files']={n:sha(b) for n,b in payload.items()};m['purpose']='paired_training_comparison';m['runtime_evidence']='20261003T134845Z_4974049d006e'
a=out/'NCA-RGR1-Paired-Package.zip'
with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED) as z:
 for n,b in payload.items():z.writestr(n,b)
 z.writestr('manifest.json',json.dumps(m,indent=2))
oldreceipt=json.loads((old/'package-receipt.json').read_bytes());nb=json.loads((old/'NCA-RGR1-cu130-Preflight.ipynb').read_bytes())
for c in nb['cells']:
 text=''.join(c['source']).replace(oldreceipt['sha256'],sha(a.read_bytes())).replace(oldreceipt['archive'],a.name).replace('colab_reversible_compatibility.py','colab_reversible_pair.py').replace('compatibility-runs','paired-runs').replace("'120'","'600'").replace('nca-rgr1-compat-','nca-rgr1-pair-')
 if c['cell_type']=='markdown':text=guide
 if c['cell_type']=='code':compile(text,'notebook','exec')
 c['source']=text.splitlines(True)
(out/'NCA-RGR1-Paired.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8');(out/'START-HERE.md').write_text(guide,encoding='utf-8')
(out/'package-receipt.json').write_text(json.dumps(dict(archive=a.name,sha256=sha(a.read_bytes()),manifest_sha256=sha(json.dumps(m,indent=2).encode()),notebook_sha256=sha((out/'NCA-RGR1-Paired.ipynb').read_bytes()),plan=plan,payload_files=len(payload),approved=False),indent=2),encoding='utf-8')
# Checked fresh extraction for local rehearsal.
dest=out/'rehearsal'
with zipfile.ZipFile(a) as z:
 assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(payload)|{'manifest.json'}
 for n in z.namelist():
  q=Path(n);assert not q.is_absolute() and '..' not in q.parts and ':' not in n
  if n!='manifest.json':assert sha(z.read(n))==m['files'][n]
  p=dest/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(z.read(n))
print(out)
