from pathlib import Path
import sys,json,time,hashlib,shutil,subprocess,copy
B=Path('C:/Users/artin/Documents/Codex/outputs');P=B/'G11-R3-Scale-2026-10-04';OLD=B/'G11-R3-Skins-2026-10-04'
def save(n,v):(P/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
if len(sys.argv)==1:
 P.mkdir(exist_ok=False)
 for n in ['source','model']:shutil.copytree(OLD/n,P/n)
 shutil.copyfile(__file__,P/'benchmark.py');shutil.copyfile(OLD/'config.json',P/'base-config.json')
 scene=json.loads((OLD/'runs/bbf64c65230c4531bf50a887ade4f59d/scene.json').read_text());save('base-scene.json',scene)
 save('protocol.json',dict(grids=[32,40],worker_timeout_seconds=180,steps=128,seed=2102,request=.24,threads=2,route='original R3',change='XY building positions/extents and connection Y scaled by1.25; connection X reattached; heights and voxel0.8m unchanged; larger physical domain, NOT higher resolution',limits='two single timing samples; new condition changes random-mask shape; not same per-voxel randomness or statistical scaling benchmark',no_training=True))
 save('RESUME.json',dict(status='benchmark running',next='Inspect each result and controller log before retry; timeout is preserved, no automatic retry.'))
 save('freeze.json',{p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file()})
 for n in [32,40]:
  with (P/f'{n}-log.txt').open('w',encoding='utf-8') as log:
   try:
    r=subprocess.run([sys.executable,str(P/'benchmark.py'),str(n)],stdout=log,stderr=subprocess.STDOUT,timeout=180);save(f'{n}-controller.json',dict(exit_code=r.returncode,timed_out=False))
   except subprocess.TimeoutExpired:save(f'{n}-controller.json',dict(timed_out=True,timeout_seconds=180))
  print(n,'finished',flush=True)
 sys.exit()
import numpy as np,torch,ctypes
sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'source'))
from context_route import route_from_context
from g11_reservation import witness,rollout
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.paced_generation import PacedNCA
from nca.repair_portable import read_portable
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets,neighbors
from nca.contract import entrance_masks,validate_scene
from nca.repair_benchmark import condition
from nca.budget_reference import budget
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
n=int(sys.argv[1]);s=json.loads((P/'base-scene.json').read_text());cfg=json.loads((P/'base-config.json').read_text());cfg['grid_size']=n;s['grid_size']=n;s['scene_id']=f'scale-{n}'
if n!=32:
 for b in s['buildings']:
  for axis in ['x','y']:b[axis]=[round(v*n/32) for v in b[axis]]
  if b['side']:b['gap_facing_x']=b['x'][1 if b['side']=='left' else 0]
 for i,e in enumerate(s['entrances']):e['y']=round(e['y']*n/32);e['x']=s['buildings'][i]['x'][1] if i==0 else s['buildings'][i]['x'][0]-2
s=validate_scene(s);save(f'{n}-scene.json',s);save(f'{n}-config.json',cfg)
model=PacedNCA();model.load_state_dict(read_portable(P/'model/checkpoint-0427.pt',json.loads((P/'model/identity.json').read_text()))['model']);model.eval();times={};t=time.perf_counter()
fields,domain,_=target_context(s,cfg);c=condition(s,fields,domain,.24);x=seed_inputs(c);_,C=budget(int(domain.sum()),.24,3);np.savez_compressed(P/f'{n}-context.npz',condition=c);times['context']=time.perf_counter()-t;t=time.perf_counter()
route=route_from_context(c);times['route']=time.perf_counter()-t
if route is None:save(f'{n}-result.json',dict(status='no_route',times=times));sys.exit()
t=time.perf_counter();W,plan=witness(route,x['allowed'],s,fields,domain,C);score,_=evaluate_targets(W,s,fields,domain);times['witness_and_certificate']=time.perf_counter()-t;save(f'{n}-certificate.json',dict(score=score,plan=plan));np.savez_compressed(P/f'{n}-witness.npz',field=W,route=route)
if not(score['contract_pass'] and W.sum()<=C):save(f'{n}-result.json',dict(status='certificate_failed',times=times));sys.exit()
contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool);face=neighbors(fields['existing'].astype(bool));ends=entrance_masks(s)
for e in s['entrances']:
 if e['kind']=='facade':contact &= ~(ends[e['id']]&face&x['allowed'])
t=time.perf_counter();hy=rollout(model,torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],W,contact,firing_seed=2102);times['rollout128']=time.perf_counter()-t
np.savez_compressed(P/f'{n}-trajectory.npz',births=hy['births'],provenance=hy['provenance'],state64=hy['states'][64],state128=hy['states'][128]);save(f'{n}-trace.json',hy['trace']);scores={}
for step in [64,128]:
 f=hy['states'][step][0,0].astype(bool);score,_=evaluate_targets(f,s,fields,domain);scores[str(step)]=dict(score=score,volume_error=float(abs(f.sum()/domain.sum()-.24)));np.savez_compressed(P/f'{n}-{step}.npz',field=f)
growth=float((hy['states'][128][0,0].sum()-hy['states'][64][0,0].sum())/hy['states'][64][0,0].sum())
class Memory(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
memory=Memory();memory.cb=ctypes.sizeof(memory);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),ctypes.byref(memory),memory.cb)
save(f'{n}-result.json',dict(status='evaluated',times=times,scores=scores,growth=growth,passed=all(v['score']['contract_pass'] and v['volume_error']<=.04 for v in scores.values()) and growth<=.05,peak_process_working_set_bytes=memory.PeakWorkingSetSize if ok else None,runtime=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,threads=2,device='cpu')))
print('complete',n,times,flush=True)
