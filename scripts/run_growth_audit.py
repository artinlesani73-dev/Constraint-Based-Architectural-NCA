"""H1 fixed growth/firing diagnostic and actual F2 gradients, no optimizer."""
import argparse
from pathlib import Path
import subprocess,sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.growth_common import CONFIG,registry,load_source,score,transition,vector_summary,cost_gate
from scripts.run_sensitivity import STORE,records,record,arrays
from nca.experiments import read_json,write_once,provenance,snapshot_source,digest
from nca.interventions import experimental_rollout
from nca.access_training import objective_pair
from nca.losses import LossSpec
from nca.objective import weighted_total


def growth(run,protocol,source_id):
    source=protocol['sources'][source_id];model,weights,item,ctx,allow=load_source(source)
    seeds=protocol['config']['pilot_firing_seeds'] if protocol['mode']=='pilot' else protocol['config']['firing_seeds']
    with torch.no_grad():
        for seed in seeds:
            previous=None;previous_horizon=None
            for horizon in protocol['config']['horizons']:
                tick=time.perf_counter()
                out=experimental_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',horizon,torch.Generator().manual_seed(seed))
                state,raw=out['state'],out['raw_material'];p=state[:,model.config['ch_structure']]
                scored=score(state,raw,item,ctx,allow,model.config,protocol['recipes'])
                anchor_match=None
                if seed==2 and str(horizon) in source['anchors']:
                    anchor=source['anchors'][str(horizon)]
                    with np.load(STORE.path(source['source_run'])/anchor['fields']['path'],allow_pickle=False) as f:
                        assert np.array_equal(p.numpy(),f['material']) and np.array_equal(raw.numpy(),f['raw'])
                    t=anchor['trace']
                    for key in ('terms','regularizers','mass_ratio','metrics','raw_saturation'):assert scored[key]==t[key]
                    assert scored['totals_v1']==t['totals_under_both_recipes']
                    anchor_match=True
                name=f'{source_id}-s{seed}-h{horizon}'
                field=arrays(run,name,{'material':p.numpy(),'raw':raw.numpy()})
                row={'source_id':source_id,'scene':source['scene'],'firing_seed':seed,'steps':horizon,
                    'score':scored,'fields':field,'anchor_exact':anchor_match,'previous_horizon':previous_horizon,
                    'change':transition(previous,p.numpy()) if previous is not None else None,
                    'seconds':time.perf_counter()-tick}
                record(run,name,row,'growth_case');previous=p.numpy().copy();previous_horizon=horizon
            print(f'{source_id} seed{seed}: all six horizons saved',flush=True)
    assert all(torch.equal(v,weights[n]) for n,v in model.state_dict().items())
    record(run,source_id+'-weights',{'source_id':source_id,'frozen_weights_unchanged':True},'weight_check')


def gradient(run,protocol,source_id,horizon):
    source=protocol['sources'][source_id];model,weights,item,ctx,allow=load_source(source)
    out=experimental_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',horizon,torch.Generator().manual_seed(2))
    state,raw=out['state'],out['raw_material'];p=state[:,model.config['ch_structure']]
    forward=next(r for r in records(run,'growth_case') if r['source_id']==source_id and r['firing_seed']==2 and r['steps']==horizon)
    with np.load(STORE.path(run)/forward['fields']['path'],allow_pickle=False) as f:
        assert np.array_equal(p.detach().numpy(),f['material']) and np.array_equal(raw.detach().numpy(),f['raw'])
    old,new,details=objective_pair(state,raw,ctx,model.config,allow,LossSpec());c=protocol['recipes'][source['recipe']]
    terms={'access_v1':old['terms']['access'][0],'access_v2':new['terms']['access'][0],
        'coverage':old['terms']['coverage'][0],'sparsity':old['terms']['sparsity'][0],
        'total_v1':weighted_total(old,c['family_weights'],c['regularizer_weights']),
        'total_v2':weighted_total(new,c['family_weights'],c['regularizer_weights'])}
    params=list(model.named_parameters());vectors={};raw_vectors={}
    for name,term in terms.items():
        grads=torch.autograd.grad(term,[p for _,p in params],retain_graph=True,allow_unused=True)
        vector=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,(_,p) in zip(grads,params)])
        raw_grad,=torch.autograd.grad(term,raw,retain_graph=True,allow_unused=True)
        raw_grad=torch.zeros_like(raw) if raw_grad is None else raw_grad.detach()
        assert torch.isfinite(vector).all() and torch.isfinite(raw_grad).all()
        vectors[name]=vector.numpy();raw_vectors['raw_gradient_'+name]=raw_grad.numpy()
    norms,cosines=vector_summary(vectors)
    summaries={k:{'value':float(v.detach()),'parameter_l2':norms[k],
        'last_raw_l2':float(np.linalg.norm(raw_vectors['raw_gradient_'+k].astype(float)))} for k,v in terms.items()}
    assert all(torch.equal(v,weights[n]) for n,v in model.state_dict().items())
    name=f'g-{source_id}-h{horizon}';field=arrays(run,name,{'material':p.detach().numpy(),'raw':raw.detach().numpy(),**vectors,**raw_vectors})
    record(run,name,{'source_id':source_id,'steps':horizon,'firing_seed':2,'scene':source['scene'],
        'fields':field,'norms':summaries,'cosines':cosines,'candidate':details[0],
        'parameter_layout':[{'name':n,'shape':list(p.shape),'elements':p.numel()} for n,p in params],
        'frozen_weights_unchanged':True,'saved_forward_exact':True},'growth_gradient')
    print(name+' saved',flush=True)


def launch(run,label,command,cap,kind):
    path=STORE.path(run)/(label+'.log');tick=time.perf_counter();timed_out=False
    with path.open('x',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*command],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=cap)
        except subprocess.TimeoutExpired:process.kill();process.wait();code=process.returncode;timed_out=True
        except BaseException:process.kill();process.wait();raise
    STORE.attach(run,path,'worker_log');elapsed=time.perf_counter()-tick
    record(run,'process-'+label,{'label':label,'kind':kind,'seconds':elapsed,'cap_seconds':cap,
        'returncode':code,'timed_out':timed_out,'elapsed_cap_exceeded':elapsed>cap},'process_record')
    print(f'{label}: exit={code}, elapsed={elapsed:.2f}',flush=True)
    if timed_out or elapsed>cap:raise TimeoutError(label+' exceeded elapsed cap; all evidence retained')
    if code:raise RuntimeError(label+' failed; inspect retained log')


def pilot_gate(run):
    assert not STORE.verify(run) and read_json(STORE.path(run)/'result.json')['status']=='completed'
    p=records(run,'protocol')[0];assert p['mode']=='pilot'
    rows=records(run,'process_record')
    assert all(r['returncode']==0 and not r['elapsed_cap_exceeded'] and not r['timed_out'] for r in rows)
    assert len(records(run,'growth_case'))==12 and len(records(run,'growth_gradient'))==2
    return cost_gate([r['seconds'] for r in rows if r['kind']=='growth'],[r['seconds'] for r in rows if r['kind']=='gradient'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['pilot','study'])
    p.add_argument('--worker',choices=['growth','gradient']);p.add_argument('--run-id');p.add_argument('--source-id');p.add_argument('--horizon',type=int)
    p.add_argument('--pilot-run');p.add_argument('--parent-run');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.worker:
        protocol=records(args.run_id,'protocol')[0]
        assert all(digest(REPO/k)==v for k,v in protocol['code_sha256'].items()),'Diagnostic source drift'
        if args.worker=='growth':growth(args.run_id,protocol,args.source_id)
        else:gradient(args.run_id,protocol,args.source_id,args.horizon)
        return 0
    if not args.mode:p.error('--mode required')
    config=read_json(REPO/CONFIG);sources,recipes=registry(config)
    audit=config['access_audit_run'];assert not STORE.verify(audit) and read_json(STORE.path(audit)/'result.json')['status']=='completed'
    a2=records(audit,'protocol')[0]
    for name in ('nca/access.py','nca/objective.py','nca/losses.py','nca/interventions.py','nca/facade.py','nca/regularizers.py','deploy/model_utils.py'):
        assert digest(REPO/name)==a2['code_sha256'][name], 'Reused gradient formula changed: '+name
    reused=records(audit,'access_gradient');assert len(reused)==12
    code=list((REPO/'nca').glob('*.py'))+[REPO/n for n in (CONFIG,'scripts/run_growth_audit.py','scripts/growth_common.py',
        'scripts/run_sensitivity.py','scripts/diagnostic_inputs.py','scripts/report_corridor_comparison.py','deploy/model_utils.py','deploy/checkpoints.py')]
    protocol={'protocol':'H1_v1','mode':args.mode,'config':config,'sources':sources,'recipes':recipes,
        'code_sha256':{f.relative_to(REPO).as_posix():digest(f) for f in code},'reused_gradient_records':reused,
        'pilot_run':args.pilot_run,'optimizer_updates':0}
    admission=None
    if args.mode=='study':
        if not args.pilot_run:p.error('--pilot-run required')
        admission=pilot_gate(args.pilot_run);assert admission['admitted'],admission
        old=records(args.pilot_run,'protocol')[0]
        for key in ('config','sources','recipes','code_sha256','reused_gradient_records'):assert protocol[key]==old[key],key
    run=STORE.create('H1 '+args.mode,'growth_diagnostic',protocol,2,provenance(REPO),parent_run=args.parent_run)
    print('RUN_ID='+run,flush=True);d=STORE.path(run);started=time.perf_counter();status,error='completed',None
    limit=config['pilot_total_cap_seconds'] if args.mode=='pilot' else config['study_total_cap_seconds']
    try:
        record(run,'protocol',protocol,'protocol');snapshot_source(REPO,d/'source.zip');STORE.attach(run,d/'source.zip','source_snapshot')
        selected=config['pilot_sources'] if args.mode=='pilot' else list(sources)
        for source in selected:
            cap=min(config['growth_worker_cap_seconds'],limit-(time.perf_counter()-started))
            if cap<=0:raise TimeoutError('Overall elapsed cap exhausted')
            launch(run,source,['--worker','growth','--run-id',run,'--source-id',source],cap,'growth')
        selected=config['pilot_gradient_sources'] if args.mode=='pilot' else [s for s in sources if sources[s]['arm']=='F2']
        for source in selected:
            for horizon in config['gradient_horizons']:
                cap=min(config['gradient_worker_cap_seconds'],limit-(time.perf_counter()-started))
                if cap<=0:raise TimeoutError('Overall elapsed cap exhausted')
                launch(run,f'g-{source}-h{horizon}',['--worker','gradient','--run-id',run,'--source-id',source,'--horizon',str(horizon)],cap,'gradient')
        assert len(records(run,'growth_case'))==(12 if args.mode=='pilot' else 180)
        assert len(records(run,'growth_gradient'))==(2 if args.mode=='pilot' else 8)
        if args.mode=='pilot':
            rows=records(run,'process_record');admission=cost_gate([r['seconds'] for r in rows if r['kind']=='growth'],[r['seconds'] for r in rows if r['kind']=='gradient'])
        if time.perf_counter()-started>limit:raise TimeoutError('Overall elapsed cap exceeded at finalization')
    except (KeyboardInterrupt,TimeoutError):status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'mode':args.mode,'growth_cases':len(records(run,'growth_case')),
        'new_gradient_cases':len(records(run,'growth_gradient')),'reused_gradient_cases':12,'optimizer_updates':0,
        'seconds':time.perf_counter()-started,'cap_seconds':limit,'admission':admission,'provenance':read_json(d/'run.json')['provenance']}
    record(run,'summary',summary,'summary')
    if error:STORE.event(run,'error','Growth diagnostic stopped; evidence retained',traceback=error)
    STORE.finish(run,status,summary,config['scope']);write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True);return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
