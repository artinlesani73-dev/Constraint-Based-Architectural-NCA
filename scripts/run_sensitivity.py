"""K2 actual-loop CPU recovery and fixed local sensitivity comparison."""
import argparse
from pathlib import Path
import subprocess, sys, time, traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,provenance,snapshot_source,read_json,write_once,digest
from nca.sensitivity import Session,make_metadata,contexts,PROPOSAL
from nca.recovery import save_checkpoint,tree_equal
from nca.objective import research_terms,weighted_total
from nca.losses import LossSpec
from nca.interventions import experimental_rollout
from nca.e0 import evaluate
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from scripts.diagnostic_inputs import load_inputs

STORE=RunStore(REPO/'.local-artifacts/runs')


def records(run,role):
    d=STORE.path(run)
    return [read_json(d/e['details']['path']) for p in sorted((d/'events').glob('*.json'))
            for e in [read_json(p)] if e['kind']=='artifact' and e['details']['role']==role]


def record(run,name,data,role):
    p=STORE.path(run)/(name+'.json');write_once(p,data)
    return STORE.attach(run,p,role)


def arrays(run,name,data):
    p=STORE.path(run)/(name+'.npz')
    with p.open('xb') as f:np.savez_compressed(f,**data)
    return STORE.attach(run,p,'fields')


def worker(args):
    d=STORE.path(args.run_id);protocol=read_json(d/'protocol.json');meta=protocol['members'][args.branch]
    session=Session(meta,args.resume)
    if not session.completed <= args.stop_after <= meta['updates']:
        raise ValueError('Invalid stop boundary')
    for _ in range(session.completed,args.stop_after):
        tick=time.perf_counter();trace,fields=session.step();u=session.completed
        path=d/f'{args.branch}-u{u:02d}.pt'
        save_checkpoint(path,session.model,session.optimizer,session.scheduler,session.generator,meta,u)
        ckpt=STORE.attach(args.run_id,path,'training_checkpoint')
        field=arrays(args.run_id,f'{args.branch}-u{u:02d}',fields)
        row={'branch':args.branch,'trace':trace,'checkpoint':ckpt,'fields':field,'seconds':time.perf_counter()-tick}
        record(args.run_id,f'{args.branch}-u{u:02d}',row,'training_update')
        print(f'{args.branch} update={u} scene={trace["scene"]} loss={trace["total_loss"]:.7g} seconds={row["seconds"]:.3f}',flush=True)
    return 0


def evaluation_worker(args):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    d=STORE.path(args.run_id);protocol=read_json(d/'protocol.json');proposal=protocol['proposal']
    cfg,weights,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
    scenes=proposal['training_scenes'];ctxs=contexts(inputs,cfg,scenes)
    if args.branch!='W1_procedural':
        model=UrbanPavilionNCA(dict(cfg))
        if args.branch!='original_checkpoint':
            rows=[r for r in records(args.run_id,'training_update') if r['branch']==args.branch]
            final=max(rows,key=lambda r:r['trace']['update'])
            payload=torch.load(d/final['checkpoint']['path'],weights_only=True,map_location='cpu')
            if payload['completed_updates']!=proposal['updates_per_run']:raise ValueError('Incomplete trained arm')
            if payload['metadata']!=protocol['members'][args.branch]:raise ValueError('Checkpoint metadata changed')
            weights=payload['model']
        elif digest(checkpoint)!=next(iter(protocol['members'].values()))['checkpoint_sha256']:
            raise ValueError('Original checkpoint changed')
        model.load_state_dict(weights);model.train()
    else:
        w1=protocol['witness_run'];assert not STORE.verify(w1)
        witness={r['scene_id']:r for r in records(w1,'witness_record')}
    with torch.no_grad():
        for i,name in enumerate(scenes):
            item=inputs[name];ctx,allowance=ctxs[name]
            horizons=[None] if args.branch=='W1_procedural' else proposal['evaluation']['horizons']
            for steps in horizons:
                started=time.perf_counter()
                if steps is None:
                    with np.load(STORE.path(w1)/witness[name]['fields']['path'],allow_pickle=False) as f:
                        p=torch.from_numpy(f['material'].copy()).float()
                    state=item['seed'].clone();state[:,cfg['ch_structure']]=p;raw=p
                else:
                    result=experimental_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',steps,
                                               torch.Generator().manual_seed(proposal['evaluation']['firing_seed']))
                    state,raw=result['state'],result['raw_material'];p=state[:,cfg['ch_structure']]
                seconds=time.perf_counter()-started
                values=research_terms(state,raw,ctx,cfg,allowance,LossSpec())
                if not bool(values['context_valid'][0]) or not torch.isfinite(state).all() or not torch.isfinite(raw).all():raise ValueError('Invalid evaluation')
                if not torch.equal(state[:,:cfg['n_frozen']],item['seed'][:,:cfg['n_frozen']]):raise ValueError('Frozen evaluation context changed')
                field=arrays(args.run_id,f'e-{args.branch}-{i:02d}-{steps}',{'material':p.numpy(),'raw':raw.numpy()})
                row={'branch':args.branch,'scene':name,'scene_hash':item['scene_hash'],'steps':steps,
                    'firing_seed':None if steps is None else proposal['evaluation']['firing_seed'],
                    'terms':{k:float(v[0]) for k,v in values['terms'].items()},
                    'regularizers':{k:float(v[0]) for k,v in values['regularizers'].items()},
                    'mass_ratio':float(values['mass_ratio'][0]),'metrics':evaluate(state,cfg,item['scene']),
                    'totals_under_both_recipes':{r:float(weighted_total(values,c['family_weights'],c['regularizer_weights'])) for r,c in proposal['recipes'].items()},
                    'fields':field,'forward_seconds':seconds,
                    'timing_scope':'saved-field load only; original construction cost is in W1' if steps is None else 'rollout only; excludes scoring and serialization'}
                record(args.run_id,f'e-{args.branch}-{i:02d}-{steps}',row,'evaluation_record')
            print(f'evaluate {args.branch} {i+1}/17 {name}',flush=True)
    return 0


def launch(run,label,command,cap):
    d=STORE.path(run);log=d/(label+'.log');tick=time.perf_counter();timed_out=False
    with log.open('x',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*command],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();code=process.returncode;timed_out=True
        except BaseException:
            process.kill();process.wait();raise
    STORE.attach(run,log,'worker_log')
    record(run,'process-'+label,{'label':label,'seconds':time.perf_counter()-tick,'returncode':code,'timed_out':timed_out,'cap_seconds':cap},'process_record')
    print(f'{label}: exit={code}, seconds={time.perf_counter()-tick:.2f}',flush=True)
    if timed_out:raise TimeoutError(label+' reached wall-time cap; completed checkpoints retained')
    if code:raise RuntimeError(label+' failed; see '+str(log))


def verify_recovery(run):
    assert not STORE.verify(run)
    d=STORE.path(run);rows=records(run,'training_update')
    branches={b:sorted([r for r in rows if r['branch']==b],key=lambda r:r['trace']['update']) for b in ('whole','prefix','resumed','repeat')}
    assert [r['trace']['update'] for r in branches['whole']]==[1,2,3]
    assert [r['trace']['update'] for r in branches['prefix']]==[1]
    checks={}
    def load(r):return torch.load(d/r['checkpoint']['path'],weights_only=True,map_location='cpu')
    checks['prefix_checkpoint']=tree_equal(load(branches['whole'][0]),load(branches['prefix'][0]))
    for b in ('resumed','repeat'):
        combined=branches['prefix']+branches[b]
        assert [r['trace']['update'] for r in combined]==[1,2,3]
        checks[b+'_traces']=[r['trace'] for r in combined]==[r['trace'] for r in branches['whole']]
        checks[b+'_checkpoint']=tree_equal(load(combined[-1]),load(branches['whole'][-1]))
        checks[b+'_fields']=True
        for a,c in zip(branches['whole'],combined):
            with np.load(d/a['fields']['path'],allow_pickle=False) as x,np.load(d/c['fields']['path'],allow_pickle=False) as y:
                checks[b+'_fields'] &= x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
    if not all(checks.values()):raise ValueError('Actual K2 recovery check failed: '+str(checks))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['recovery','study']);p.add_argument('--recovery-run');p.add_argument('--parent-run');p.add_argument('--resume-run')
    p.add_argument('--worker',choices=['train','evaluate']);p.add_argument('--run-id');p.add_argument('--branch');p.add_argument('--stop-after',type=int,default=17);p.add_argument('--resume')
    args=p.parse_args()
    if args.worker:return worker(args) if args.worker=='train' else evaluation_worker(args)
    if not args.mode:p.error('--mode required')
    if args.resume_run and args.mode!='study':p.error('--resume-run only for study')
    torch.set_num_threads(2)
    proposal=read_json(REPO/PROPOSAL);cfg,_,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
    assert not STORE.verify(proposal['source_calibration_run'])
    assert read_json(STORE.path(proposal['source_calibration_run'])/'result.json')['status']=='completed'
    members={}
    if args.mode=='recovery':
        m=make_metadata('mass_3',0,inputs,cfg,checkpoint);members={b:m for b in ('whole','prefix','resumed','repeat')}
    else:
        if not args.recovery_run:p.error('--recovery-run required')
        assert read_json(STORE.path(args.recovery_run)/'result.json')['status']=='completed'
        verify_recovery(args.recovery_run)
        gate=read_json(STORE.path(args.recovery_run)/'protocol.json')['members']['whole']
        assert gate==make_metadata('mass_3',0,inputs,cfg,checkpoint),'Recovery source/config differs'
        for recipe in ('mapped_30','mass_3'):
            for seed in proposal['training_seeds']:members[f'{recipe}-s{seed}']=make_metadata(recipe,seed,inputs,cfg,checkpoint)
    protocol={'protocol':'K2R_v1' if args.mode=='recovery' else 'K2_v1','proposal':proposal,'members':members,
        'recovery_gate':args.recovery_run,'witness_run':'20260923T084933Z_eb2603cd79f7','resume_run':args.resume_run,
        'evaluation_worker_cap_seconds':900,'paid_compute':False,
        'scope':'Local coefficient sensitivity on development scenes; no geometry generalization or architecture-quality claim'}
    origin=provenance(REPO);run=STORE.create('K2 '+args.mode,'sensitivity_'+args.mode,protocol,0,origin,parent_run=args.resume_run or args.parent_run)
    print('RUN_ID='+run,flush=True);d=STORE.path(run);status,error='completed',None;checks={};started=time.perf_counter()
    try:
        record(run,'protocol',protocol,'protocol');source=d/'source.zip';snapshot_source(REPO,source);STORE.attach(run,source,'source_snapshot')
        if args.resume_run:
            previous=STORE.path(args.resume_run);assert not STORE.verify(args.resume_run)
            assert read_json(previous/'result.json')['status'] in ('failed','interrupted')
            assert read_json(previous/'protocol.json')['members']==members
            for index,row in enumerate(records(args.resume_run,'training_update')):
                assert row['branch'] in members
                for key in ('checkpoint','fields'):row[key]=STORE.attach(run,previous/row[key]['path'],'imported_'+key)
                row['imported_from']=args.resume_run
                record(run,f'import-{index:03d}',row,'training_update')
        if args.mode=='recovery':
            for branch,stop in [('whole',3),('prefix',1),('resumed',3),('repeat',3)]:
                command=['--worker','train','--run-id',run,'--branch',branch,'--stop-after',str(stop)]
                if branch in ('resumed','repeat'):
                    row=next(r for r in records(run,'training_update') if r['branch']=='prefix')
                    command+=['--resume',str(d/row['checkpoint']['path'])]
                launch(run,branch,command,900)
            checks=verify_recovery(run)
        else:
            for branch in members:
                command=['--worker','train','--run-id',run,'--branch',branch,'--stop-after','17']
                prior=sorted([r for r in records(run,'training_update') if r['branch']==branch],key=lambda r:r['trace']['update'])
                if prior:
                    assert [r['trace']['update'] for r in prior]==list(range(1,len(prior)+1))
                    command+=['--resume',str(d/prior[-1]['checkpoint']['path'])]
                launch(run,'train-'+branch,command,proposal['cpu_time_cap_seconds_per_run'])
            for branch in [*members,'original_checkpoint','W1_procedural']:
                launch(run,'eval-'+branch,['--worker','evaluate','--run-id',run,'--branch',branch],900)
            assert len(records(run,'training_update'))==68
            assert len(records(run,'evaluation_record'))==187
            checks={'complete_training_matrix':True,'complete_evaluation_matrix':True}
    except (KeyboardInterrupt,TimeoutError):status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'mode':args.mode,'checks':checks,
        'recorded_updates':len(records(run,'training_update')),'evaluation_cases':len(records(run,'evaluation_record')),
        'seconds':time.perf_counter()-started,'provenance':origin,'artifact_location':f'.local-artifacts/runs/{run}'}
    record(run,'summary',summary,'summary')
    if error:STORE.event(run,'error','Attempt stopped; partial evidence and completed boundaries retained',traceback=error)
    STORE.finish(run,status,checks,protocol['scope']);write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True);return 0 if status=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
