"""Versioned continuation: one slot per process, unused raw capture histories omitted.

Original inputs, runner, and partial results remain immutable. Exact saved-array
parity for all three methods precedes any missing-slot execution.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time, textwrap, inspect, shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / '.local-artifacts/r3-contribution-2026-10-08'
OUT = ROOT / '.local-artifacts/r3-contribution-continuation-v2-2026-10-08'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v): p.write_text(json.dumps(v, indent=2)+'\n', encoding='utf-8')

def verify_inputs():
    for n, h in json.loads((OLD/'input-manifest.json').read_text()).items():
        assert sha(OLD/n) == h, n

def worker(index, seed, method, phase):
    import numpy as np
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(OLD/'source'))
    import nca.paced_generation as paced
    original = textwrap.dedent(inspect.getsource(paced.PacedNCA.rollout))
    replacements = {
        'births.append(tensor(new_field&~field).cpu());proposals.append(q.detach().cpu())': 'births.append(tensor(new_field&~field).cpu())',
        'candidates.append(tensor(field|cube_union(eligible_np & copied[1].astype(bool) & (copied[0]>.5))).cpu())': 'pass # unused diagnostic candidate history omitted',
        ',proposals=torch.stack(proposals),pre_admission_candidates=torch.stack(candidates)': '',
    }
    modified = original
    for before, after in replacements.items():
        assert modified.count(before) == 1, before
        modified = modified.replace(before, after)
    namespace = dict(vars(paced))
    exec(compile(modified, '<capture-only-v2>', 'exec'), namespace)
    paced.PacedNCA.rollout = namespace['rollout']
    dest = OUT / f'{phase}-{index}-{seed}-{method}'
    code = (OLD/'frozen-study.py').read_text()
    edits = {
        "dest=OUT/f'case-{index}'": 'dest=DEST',
        "for seed in protocol['seeds']:": 'for seed in [SELECTED_SEED]:',
        "for method in protocol['methods']:": 'for method in [SELECTED_METHOD]:',
    }
    for before, after in edits.items():
        assert code.count(before) == 1, before
        code = code.replace(before, after)
    ns = dict(__file__=str(ROOT/'scripts/r3_contribution_study.py'), __name__='frozen_worker_v2',
              DEST=dest, SELECTED_SEED=seed, SELECTED_METHOD=method)
    exec(compile(code, '<frozen-worker-slot-v2>', 'exec'), ns)
    ns['worker'](index)
    (dest/'effective-rollout.py').write_text(modified, encoding='utf-8')
    if phase == 'parity':
        checked = []
        for p in dest.glob(f'{method}-{seed}-*.npz'):
            with np.load(p) as actual, np.load(OLD/f'case-{index}'/p.name) as expected:
                assert set(actual.files) == set(expected.files)
                for k in actual.files:
                    assert np.array_equal(actual[k], expected[k]), (p.name,k)
                    checked.append(f'{p.name}:{k}')
        assert checked
        a=json.loads((dest/'results.json').read_text())[0]
        b=next(r for r in json.loads((OLD/f'case-{index}'/'results.json').read_text()) if r['seed']==seed and r['method']==method)
        assert {k:v for k,v in a.items() if k!='rollout_seconds'} == {k:v for k,v in b.items() if k!='rollout_seconds'}
        save(dest/'parity.json',dict(exact_saved_array_and_metric_parity=True,arrays=checked))
    save(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'})

def run():
    verify_inputs()
    OUT.mkdir(exist_ok=False)
    shutil.copyfile(__file__, OUT/'frozen-continuation.py')
    protocol=json.loads((OLD/'protocol.json').read_text())
    save(OUT/'protocol.json',dict(parent=str(OLD),parent_manifest_sha256=sha(OLD/'input-manifest.json'),
        scientific_protocol=protocol,change='Fresh process per method/seed; omit unused raw proposal and candidate histories only.',
        parity='Case2 seed2102 all three methods; every saved array and metric exact.',limits=dict(worker_seconds=180,total_seconds=900)))
    jobs=[('parity',2,2102,m) for m in protocol['methods']]
    jobs += [('missing',2,2103,m) for m in protocol['methods']]
    jobs += [('missing',3,s,m) for s in protocol['seeds'] for m in protocol['methods']]
    records=[]; start=time.monotonic()
    for phase,i,s,m in jobs:
        save(OUT/'controller.json',dict(state='running',records=records,current=[phase,i,s,m]))
        remaining=900-(time.monotonic()-start)
        if remaining<=0: break
        with (OUT/f'{phase}-{i}-{s}-{m}.log').open('w') as log:
            try:
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--index',str(i),'--seed',str(s),'--method',m,'--phase',phase],stdout=log,stderr=subprocess.STDOUT,timeout=min(180,remaining))
                state='completed' if p.returncode==0 else 'error'
            except subprocess.TimeoutExpired: state='timeout'
        records.append(dict(phase=phase,index=i,seed=s,method=m,state=state))
        print(records[-1],flush=True)
        if state!='completed': break
    complete=len(records)==len(jobs) and all(r['state']=='completed' for r in records)
    save(OUT/'controller.json',dict(state='completed' if complete else 'stopped',records=records,seconds=time.monotonic()-start))
    verify_inputs()
    if not complete: raise RuntimeError('Continuation stopped; preserve evidence and inspect logs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker']);p.add_argument('--index',type=int);p.add_argument('--seed',type=int);p.add_argument('--method');p.add_argument('--phase');a=p.parse_args()
    if a.action=='run': run()
    else: worker(a.index,a.seed,a.method,a.phase)
