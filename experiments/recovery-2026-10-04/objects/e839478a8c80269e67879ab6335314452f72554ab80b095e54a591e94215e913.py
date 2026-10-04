from pathlib import Path
import json,zipfile,hashlib
repo=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');out=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-cu130-Preflight');out.mkdir(exist_ok=False)
old=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum')
s=(repo/'scripts/colab_curriculum_repair.py').read_text()
start=s.index('def worker(a):');end=s.index('\ndef main(a):')
worker='''def worker(a):
    if sys.stdin.readline()!='GO\\n':raise RuntimeError('Owned start required')
    threading.Thread(target=watch_parent,daemon=True).start()
    import numpy as np
    import torch
    from nca.curriculum_repair import ConnectedSession
    from nca.recovery import tree_equal
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);session=None;status='completed';error=None;t=time.monotonic();checks={}
    try:
        _,data=verify(ROOT)
        row=next(r for r in data['rows'] if r['damage']=='cube5')
        identity={'preflight':'CGR3_cu130_recovery_v1','manifest_sha256':digest(ROOT/'manifest.json')}
        session=ConnectedSession(ROOT,[row],identity,device=a.device,seed=a.seed)
        write_once(out/'identity.json',session.identity);write_once(out/'selected-row.json',row)
        if a.device=='cuda:0':
            rt=session.identity['runtime']
            expected={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
            differences={k:{'expected':v,'actual':rt[k]} for k,v in expected.items() if rt[k]!=v}
            if differences:raise ValueError('Observed runtime changed: '+json.dumps(differences))
            torch.cuda.reset_peak_memory_stats()
        session.save(out/'checkpoint-0000.pt')
        first,_=session.step();session.save(out/'before-augmented.pt')
        expected_trace,expected_state=session.step();expected_start=session.last_start.copy();expected_payload=session.payload()
        session.save(out/'after-augmented.pt')
        with (out/'expected.npz').open('xb') as f:np.savez_compressed(f,start=expected_start,state=expected_state)
        replay=ConnectedSession(ROOT,[row],identity,device=a.device,seed=a.seed)
        replay.restore(out/'before-augmented.pt');actual_trace,actual_state=replay.step()
        replay.save(out/'replayed.pt')
        with (out/'replayed.npz').open('xb') as f:np.savez_compressed(f,start=replay.last_start,state=actual_state)
        checks={'augmented_visit':expected_trace['start']['mode']=='intermediate','trace_exact':expected_trace==actual_trace,'start_exact':np.array_equal(expected_start,replay.last_start),'state_exact':np.array_equal(expected_state,actual_state),'full_payload_exact':tree_equal(expected_payload,replay.payload())}
        write_once(out/'checks.json',checks)
        if not all(checks.values()):raise AssertionError('Compatibility/recovery checks failed')
    except BaseException:error=traceback.format_exc();status='failed'
    memory={}
    if session is not None and a.device=='cuda:0':memory={'peak_reserved':torch.cuda.max_memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated()}
    result={'status':status,'failure':error,'completed':session.completed if session else 0,'optimizer_steps_executed':3 if all(checks.values()) and checks else None,'checks':checks,'device':a.device,'cpu_rehearsal':a.cpu_rehearsal,'wall_seconds':time.monotonic()-t,'memory':memory}
    write_once(out/'result.json',result);print(json.dumps(result),flush=True)
    return 0 if status=='completed' else 1

'''
s=s[:start]+worker+s[end:]
s=s.replace('curriculum-runs','compatibility-runs').replace('curriculum-attempt','compatibility-attempt').replace('600','120').replace('eight-update rehearsal','compatibility rehearsal').replace("'updates':8 if a.cpu_rehearsal else 256","'updates':2,'optimizer_steps_including_replay':3").replace('Training evidence only. Final quality is evaluated separately on CPU with frozen validation-only rules.','Compatibility and exact recovery check only; not a quality trial or full training run.')
s=s.replace('One explicitly approved CGR3 training seed, or a bounded eight-update CPU rehearsal.','CGR3 cu130 bounded compatibility/recovery check; three optimizer steps including replay.')
with zipfile.ZipFile(old/'NCA-CGR3-Curriculum-Package.zip') as z:
 payload={n:z.read(n) for n in z.namelist() if n!='manifest.json'};m=json.loads(z.read('manifest.json'))
payload['scripts/colab_curriculum_compatibility.py']=s.encode()
guide='''# CGR3 CUDA13 compatibility check

One proposed T4 check capped at120controlled seconds; setup/upload/export/idle extra.
Two training updates on one TRAIN cube5 example, then restore and replay update2:
three optimizer steps total. Includes augmented start and exact full-payload check.
No quality trial, no full256-update job, no automatic continuation or retry.
Expected observed stack: T4,Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,
cuDNN92700. Strict deterministic algorithms remain on. Changed stack stops again.
Open notebook,upload supplied ZIP,leave APPROVED_SEED_JOB=False until explicit
approval of this check. After approval set True and run once. Download ZIP+receipt,
including failure,then return both. Local download only,no Drive access.
This new package preserves the old training script unchanged; do not run it.
A successful check does not establish numerical equivalence with CUDA12.8 or
approve another training attempt. No checkpoint from this check is a quality model.
'''
payload['README.md']=guide.encode();payload['docs/next-phase/CGR3_CU130_PREFLIGHT.md']=guide.encode()
m['files']={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()};m['purpose']='cu130_compatibility_only'
a=out/'NCA-CGR3-cu130-Preflight-Package.zip'
with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED) as z:
 for n,b in payload.items():z.writestr(n,b)
 z.writestr('manifest.json',json.dumps(m,indent=2))
h=hashlib.sha256(a.read_bytes()).hexdigest()
nb=json.loads((old/'NCA-CGR3-Curriculum.ipynb').read_bytes())
oldhash=json.loads((old/'package-receipt.json').read_bytes())['archive_sha256']
for c in nb['cells']:
 text=''.join(c['source']).replace(oldhash,h).replace('NCA-CGR3-Curriculum-Package.zip',a.name).replace('colab_curriculum_repair.py','colab_curriculum_compatibility.py').replace('curriculum-runs','compatibility-runs').replace("'600'","'120'").replace('cgr3-upload','cgr3-compat-upload').replace('nca-cgr3-','nca-cgr3-compat-')
 if c['cell_type']=='markdown':text=guide
 if c['cell_type']=='code':compile(text,'cell','exec')
 c['source']=text.splitlines(True)
(out/'NCA-CGR3-cu130-Preflight.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
(out/'START-HERE.md').write_text(guide,encoding='utf-8')
(out/'compatibility-runner.py').write_text(s,encoding='utf-8')
(out/'package-receipt.json').write_text(json.dumps(dict(archive=a.name,sha256=h,manifest_sha256=hashlib.sha256(json.dumps(m,indent=2).encode()).hexdigest(),purpose='compatibility_only',controlled_seconds=120,optimizer_steps=3,approved=False),indent=2),encoding='utf-8')
print(out);print(h)
