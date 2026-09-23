"""Complete one interrupted F2 member in a linked run; retain all parent evidence."""
import copy
from pathlib import Path
import sys
import time
import traceback
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_access_training import STORE,records,record,launch
from nca.access_training import make_metadata
from nca.experiments import read_json,write_once,provenance,snapshot_source,digest
from deploy.checkpoints import load_model_c
from scripts.diagnostic_inputs import load_inputs


def main():
    parent=sys.argv[1];pd=STORE.path(parent)
    assert not STORE.verify(parent) and read_json(pd/'result.json')['status']=='interrupted'
    original=records(parent,'protocol')[0];assert original['mode']=='study' and original['protocol']=='F2_v1'
    rows=records(parent,'training_update');evaluation=records(parent,'evaluation_record')
    last={b:max([r['trace']['update'] for r in rows if r['branch']==b],default=0) for b in original['members']}
    unfinished=[b for b,u in last.items() if u<64]
    assert len(unfinished)==1,'This bounded continuation supports exactly one unfinished member'
    branch=unfinished[0];start=last[branch];assert 60<=start<64,'At most four remaining updates'
    cfg,_,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
    for m in original['members'].values():
        assert m==make_metadata(m['recipe'],m['scene'],inputs,cfg,checkpoint,m['objective_version'])
    p=copy.deepcopy(original);p['continuation']={'parent_run':parent,'branch':branch,'start_update':start,
        'cap_seconds':120,'wrapper_sha256':digest(Path(__file__)),
        'scope':'Complete missing planned updates only; timing-deviating parent retained, no clean timing claim'}
    origin=provenance(REPO);run=STORE.create('F2 linked completion','access_training_continuation',p,0,origin,parent_run=parent)
    print('RUN_ID='+run,flush=True);d=STORE.path(run);started=time.perf_counter();status,error='completed',None;checks={}
    try:
        record(run,'protocol',p,'protocol');snapshot_source(REPO,d/'source.zip');STORE.attach(run,d/'source.zip','source_snapshot')
        copied={};manifest=[]
        for role,items in [('training_update',rows),('evaluation_record',evaluation)]:
            for i,row in enumerate(items):
                new=copy.deepcopy(row)
                for key in ('checkpoint','fields'):
                    ref=row[key]
                    if ref['path'] not in copied:
                        copied[ref['path']]=STORE.attach(run,pd/ref['path'],ref['role'])
                    new[key]=copied[ref['path']];assert new[key]['sha256']==ref['sha256']
                record(run,f'import-{role}-{i:03d}',new,role)
                manifest.append({'role':role,'branch':row['branch'],'update':row['trace']['update'],
                    'source_fields':row['fields'],'copied_fields':new['fields'],
                    'source_checkpoint':row['checkpoint'],'copied_checkpoint':new['checkpoint']})
        record(run,'import-manifest',{'parent_run':parent,'records':manifest},'import_manifest')
        for i,row in enumerate(records(parent,'process_record')):
            record(run,f'parent-process-{i}',row,'process_record' if row['label']!=branch else 'interrupted_parent_process')
        STORE.attach(run,pd/'result.json','parent_result');STORE.attach(run,pd/'summary.json','parent_summary')
        prefix=next(r for r in records(run,'training_update') if r['branch']==branch and r['trace']['update']==start)
        launch(run,branch,['--worker','--run-id',run,'--branch',branch,'--stop-after','64',
            '--resume',str(d/prefix['checkpoint']['path'])],120)
        process=records(run,'process_record')[-1]
        if process['seconds']>120:raise TimeoutError('Continuation elapsed cap exceeded even though wait returned')
        training=records(run,'training_update');ev=records(run,'evaluation_record')
        expected={(b,u) for b in p['members'] for u in range(1,65)}
        assert len(training)==256 and {(r['branch'],r['trace']['update']) for r in training}==expected
        expected={(b,u,h) for b in p['members'] for u in p['proposal']['evaluation']['boundaries'] for h in (16,50)}
        assert len(ev)==56 and {(r['branch'],r['trace']['update'],r['trace']['steps']) for r in ev}==expected
        checks={'imported_training_updates':len(rows),'imported_evaluations':len(evaluation),
            'newly_executed_updates':64-start,'complete_training_matrix':True,'complete_evaluation_matrix':True}
    except (KeyboardInterrupt,TimeoutError):status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    elapsed=time.perf_counter()-started;parent_seconds=records(parent,'summary')[0]['seconds']
    summary={'run_id':run,'status':status,'error':error,'mode':'study','checks':checks,
        'recorded_updates':len(records(run,'training_update')),'evaluation_cases':len(records(run,'evaluation_record')),
        'seconds':parent_seconds+elapsed,'continuation_seconds':elapsed,'parent_seconds':parent_seconds,
        'provenance':origin,'admission':p['admission'],'parent_run':parent,
        'timing_note':'Cumulative elapsed includes interrupted parent. This is not a cap-compliant fresh study.'}
    record(run,'summary',summary,'summary')
    if error:STORE.event(run,'error','Continuation stopped; all evidence retained',traceback=error)
    STORE.finish(run,status,checks,p['continuation']['scope']);write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True);return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
