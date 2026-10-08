"""Prospectively frozen transfer evaluation; no training or case selection by result."""
from pathlib import Path
import argparse, copy, ctypes, inspect, json, shutil, subprocess, sys, time
import continue_r3_contribution as capture
ROOT=capture.ROOT
PARENT=capture.OLD
OUT=ROOT/'.local-artifacts/r3-transfer-v1-2026-10-08'
sha=capture.sha
save=capture.save

def memory():
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    k=ctypes.WinDLL('kernel32',use_last_error=True);p=ctypes.WinDLL('psapi',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE
    p.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    c=Counters();c.cb=ctypes.sizeof(c)
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
    return dict(peak_working_set_bytes=c.PeakWorkingSetSize,working_set_bytes=c.WorkingSetSize,peak_commit_bytes=c.PeakPagefileUsage,scope='Whole worker process through artifact writing, not rollout-only; Windows process counters.')

def freeze():
    capture.verify_inputs()
    parent=json.loads((PARENT/'protocol.json').read_text())
    base=parent['sites'][0]
    def site(name,n,west,east,entries,extras):
        s=copy.deepcopy(base);s.update(scene_id=name,grid_size=n,description=name,notes=['Prospectively authored transfer fixture, frozen before generation. No tuning or replacement based on outcomes.'])
        s['buildings'][0].update(x=[0,west[0]],y=west[1],z=[0,west[2]],gap_facing_x=west[0])
        s['buildings'][1].update(x=[east[0],n],y=east[1],z=[0,east[2]],gap_facing_x=east[0])
        for e,x,yz in zip(s['entrances'],[west[0],east[0]-2],entries):e.update(x=x,y=yz[0],z=yz[1])
        s['buildings'] += [dict(id=f'B_context_{i}',x=x,y=y,z=z,side=None,gap_facing_x=None) for i,(x,y,z) in enumerate(extras)]
        return s
    scenes=[
        site('transfer-narrow-offset-32',32,(9,[3,27],24),(23,[5,30],28),[(10,8),(19,14)],[]),
        site('transfer-flanking-blocks-32',32,(6,[2,27],27),(25,[4,31],24),[(14,11),(17,9)],[([11,15],[4,10],[0,20]),([18,22],[24,29],[0,22])]),
        site('transfer-raised-obstacle-40',40,(8,[3,35],30),(31,[6,38],29),[(11,15),(26,18)],[([18,22],[17,23],[0,26])]),
        site('transfer-sealed-gap-40',40,(10,[4,36],28),(29,[2,38],30),[(12,10),(24,15)],[([20,23],[0,40],[0,40])]),
    ]
    sys.path.insert(0,str(PARENT/'source'))
    from nca.contract import validate_scene
    for s in scenes:validate_scene(s)
    def geometry(s):return json.dumps({k:s[k] for k in ['grid_size','buildings','entrances']},sort_keys=True)
    assert len(set(map(geometry,scenes)))==4
    assert not set(map(geometry,scenes)) & set(map(geometry,parent['sites']))
    assert memory()['peak_working_set_bytes']>0
    OUT.mkdir(exist_ok=False)
    for name in ['source','model']:shutil.copytree(PARENT/name,OUT/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['config.json','frozen-study.py']:shutil.copyfile(PARENT/name,OUT/name)
    for name in ['r3_transfer_study.py','continue_r3_contribution.py']:shutil.copyfile(ROOT/'scripts'/name,OUT/name)
    protocol=copy.deepcopy(parent)
    protocol.update(version='r3-transfer-v1',sites=scenes,seeds=[2201,2202],references={},limits=dict(worker_seconds=180,total_seconds=1200,device='cpu',threads=2),
        parent_manifest_sha256=sha(PARENT/'input-manifest.json'),
        novelty='New authored fixtures, exact geometry differs from preceding contribution fixtures. Not an exhaustive training-history exclusion audit or statistically independent benchmark.',
        classification='Report certification separately from acceptance. Intended sealed barrier kept regardless of result; retain every other failure/refusal.',
        measurement_scope='Rollout wall time from original runner; Windows peak working set/commit for whole fresh worker, includes setup and capture. No GPU extrapolation.',
        decision_rule='No tuning or case replacement. If R3 fails feasible cases, inspect cause before release or training. If neutral matches validity, continue to describe learning as form selection rather than necessary constraint satisfaction.')
    save(OUT/'protocol.json',protocol)
    save(OUT/'input-manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
    save(ROOT/'experiments/reports/R3-transfer-protocol-2026-10-08.json',dict(protocol=protocol,input_manifest_sha256=sha(OUT/'input-manifest.json'),artifact_root=str(OUT)))
    print('Frozen four sites, 24 method/request slots; no rollouts yet.')

def verify():
    for n,h in json.loads((OUT/'input-manifest.json').read_text()).items():assert sha(OUT/n)==h,n

def worker(index,seed,method):
    capture.OLD=OUT;capture.OUT=OUT
    src=inspect.getsource(capture.worker)
    old="ns['worker'](index)"
    assert src.count(old)==1
    src=src.replace(old,"ns['OUT']=OLD; ns['worker'](index)")
    ns=dict(vars(capture));exec(compile(src,'<transfer-slot-worker>','exec'),ns)
    ns['worker'](index,seed,method,'slot')
    d=OUT/f'slot-{index}-{seed}-{method}'
    save(d/'memory.json',memory())
    (d/'effective-worker.py').write_text(src,encoding='utf-8')
    save(d/'manifest.json',{p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='manifest.json'})

def run():
    verify()
    assert sha(Path(__file__))==sha(OUT/'r3_transfer_study.py')
    assert sha(ROOT/'scripts/continue_r3_contribution.py')==sha(OUT/'continue_r3_contribution.py')
    status=OUT/'controller.json'
    if status.exists():raise RuntimeError('Existing attempt; inspect, do not overwrite or silently retry')
    protocol=json.loads((OUT/'protocol.json').read_text());records=[];start=time.monotonic()
    jobs=[(i,s,m) for i in range(4) for s in protocol['seeds'] for m in protocol['methods']]
    for i,s,m in jobs:
        save(status,dict(state='running',records=records,current=[i,s,m]))
        remaining=1200-(time.monotonic()-start)
        if remaining<=0:break
        t=time.monotonic()
        with (OUT/f'slot-{i}-{s}-{m}.log').open('w') as stream:
            try:
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--index',str(i),'--seed',str(s),'--method',m],stdout=stream,stderr=subprocess.STDOUT,timeout=min(180,remaining))
                state='completed' if p.returncode==0 else 'error'
            except subprocess.TimeoutExpired:state='timeout'
        records.append(dict(index=i,seed=s,method=m,state=state,worker_wall_seconds=time.monotonic()-t));print(records[-1],flush=True)
        if state!='completed':break
    complete=len(records)==len(jobs) and all(v['state']=='completed' for v in records)
    save(status,dict(state='completed' if complete else 'stopped',records=records,seconds=time.monotonic()-start));verify()
    if not complete:raise RuntimeError('Stopped; retain evidence and inspect before any new attempt')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','worker']);p.add_argument('--index',type=int);p.add_argument('--seed',type=int);p.add_argument('--method');a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:worker(a.index,a.seed,a.method)
