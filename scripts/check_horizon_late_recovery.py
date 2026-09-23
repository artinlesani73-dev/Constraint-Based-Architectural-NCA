"""F3L: repeat updates63/64 from every saved F3 checkpoint at62.

Extra recovery evidence only; never extend training or select models by quality.
"""
import hashlib
from pathlib import Path
import sys
import time
import traceback
import zipfile
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_horizon_training import STORE,records,record,launch
from nca.experiments import read_json,write_once,snapshot_source,provenance,digest
from nca.recovery import tree_equal


def verify(run,completed=True):
    assert not STORE.verify(run);d=STORE.path(run);p=records(run,'protocol')[0]
    if completed:assert read_json(d/'result.json')['status']=='completed'
    source=p['source_run'];sd=STORE.path(source)
    assert not STORE.verify(source) and read_json(sd/'result.json')['status']=='completed'
    original=records(source,'protocol')[0]
    assert original['mode']=='study' and original['protocol']=='F3_v1'
    assert p['members']==original['members'] and p['proposal']==original['proposal']
    counts={}
    for role in ('training_update','evaluation_record'):
        key=lambda r:(r['branch'],r['trace']['update'],r['trace']['steps'])
        expected={key(r):r for r in records(source,role) if r['trace']['update'] in ((63,64) if role=='training_update' else (64,))}
        rows=records(run,role)
        assert len(rows)==len(expected) and {key(r) for r in rows}==expected.keys()
        for row in rows:
            old=expected[key(row)];assert row['trace']==old['trace']
            a=torch.load(d/row['checkpoint']['path'],map_location='cpu',weights_only=True)
            b=torch.load(sd/old['checkpoint']['path'],map_location='cpu',weights_only=True)
            assert tree_equal(a,b),key(row)
            with np.load(d/row['fields']['path'],allow_pickle=False) as x,np.load(sd/old['fields']['path'],allow_pickle=False) as y:
                assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files),key(row)
        counts[role+'_exact_full_checkpoint_trace_field_matches']=len(rows)
    events=[read_json(f) for f in (d/'events').glob('*.json')]
    source_zip=next(e['details']['path'] for e in events if e['kind']=='artifact' and e['details']['role']=='source_snapshot')
    with zipfile.ZipFile(d/source_zip) as z:
        code=next(iter(p['members'].values()))['code_sha256']
        for name,expected in {**code,**p['wrapper_sha256']}.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    counts['source_hashes_verified']=len(code)+len(p['wrapper_sha256'])
    processes=records(run,'process_record')
    assert len(processes)==4 and all(r['returncode']==0 and not r['timed_out'] and not r['elapsed_cap_exceeded'] and r['seconds']<=r['cap_seconds'] for r in processes)
    cursors=records(run,'schedule_cursor')
    assert len(cursors)==8 and {(r['branch'],r['update']) for r in cursors}=={(b,u) for b in p['members'] for u in (63,64)}
    assert all(r['next_horizon']==(50 if r['update']==63 else None) for r in cursors)
    if completed:assert records(run,'summary')[0]['seconds']<=p['total_cap_seconds']
    counts.update(schedule_cursors_verified=8,worker_caps_met=True)
    return counts


def main():
    source=sys.argv[1];assert not STORE.verify(source)
    assert read_json(STORE.path(source)/'result.json')['status']=='completed'
    old=records(source,'protocol')[0];assert old['mode']=='study' and old['protocol']=='F3_v1'
    p={'protocol':'F3L_v1','mode':'late_recovery','source_run':source,
        'proposal':old['proposal'],'members':old['members'],
        'wrapper_sha256':{'scripts/check_horizon_late_recovery.py':digest(Path(__file__))},
        'worker_cap_seconds':120,'total_cap_seconds':360,'scope':'Four CPU update62->63->64 resumes; no added training exposure'}
    origin=provenance(REPO);run=STORE.create('F3 late recovery','horizon_late_recovery',p,0,origin,parent_run=source)
    print('RUN_ID='+run,flush=True);d=STORE.path(run);started=time.perf_counter();status,error='completed',None;checks={}
    try:
        record(run,'protocol',p,'protocol');snapshot_source(REPO,d/'source.zip');STORE.attach(run,d/'source.zip','source_snapshot')
        for branch in p['members']:
            prefix=next(r for r in records(source,'training_update') if r['branch']==branch and r['trace']['update']==62)
            launch(run,branch,['--worker','--run-id',run,'--branch',branch,'--stop-after','64',
                '--resume',str(STORE.path(source)/prefix['checkpoint']['path'])],min(p['worker_cap_seconds'],p['total_cap_seconds']-(time.perf_counter()-started)))
        checks=verify(run,completed=False)
        if time.perf_counter()-started>p['total_cap_seconds']:raise TimeoutError('Late recovery total cap exceeded')
    except (KeyboardInterrupt,TimeoutError):status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'source_run':source,'status':status,'error':error,'checks':checks,
        'seconds':time.perf_counter()-started,'recorded_updates':len(records(run,'training_update')),
        'evaluation_cases':len(records(run,'evaluation_record')),'provenance':origin}
    record(run,'summary',summary,'summary')
    if error:STORE.event(run,'error','Late recovery failed; evidence retained',traceback=error)
    STORE.finish(run,status,checks,p['scope']);write_once(REPO/'experiments/records'/(run+'.json'),summary)
    if status=='completed':
        write_once(REPO/'experiments/reports/F3L-verification.json',summary)
    print(summary,flush=True);return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
