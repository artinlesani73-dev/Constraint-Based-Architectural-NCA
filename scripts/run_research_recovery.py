"""R2_v1: separate-process CPU optimizer recovery; gated by completed K1 evidence."""
import argparse
from dataclasses import asdict
from pathlib import Path
import random,subprocess,sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from nca.experiments import RunStore,provenance,snapshot_source,read_json,write_once,digest
from nca.losses import LossSpec,FAMILIES,context_from_scenes,material_envelope,mean_terms
from nca.interventions import experimental_rollout,objective_terms
from nca.recovery import save_checkpoint,restore_checkpoint,tree_equal
from nca.objective import research_terms,weighted_total
from nca.facade import endpoint_allowance
from scripts.diagnostic_inputs import load_inputs

CODE_FILES=('nca/interventions.py','nca/losses.py','nca/recovery.py','nca/contract.py',
            'deploy/model_utils.py','scripts/diagnostic_inputs.py','scripts/run_research_recovery.py',
            'nca/objective.py','nca/facade.py','nca/regularizers.py','experiments/configs/K2-sensitivity.json')


def worker(args):
    store=RunStore(REPO/'.local-artifacts/runs');directory=store.path(args.run_id)
    protocol=read_json(directory/'protocol.json');metadata=protocol['checkpoint_metadata']
    for name,expected in metadata['code_sha256'].items():
        if digest(REPO/name)!=expected:raise ValueError('Recovery source code differs: '+name)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if metadata['torch_version']!=str(torch.__version__) or metadata['numpy_version']!=str(np.__version__) or metadata['python_version']!=sys.version:
        raise ValueError('Recovery runtime differs from recorded metadata')
    cfg,weights,checkpoint=load_model_c()
    if digest(checkpoint)!=metadata['checkpoint_sha256']:raise ValueError('Original checkpoint differs')
    _,inputs=load_inputs(REPO)
    contexts={};allowances={}
    for name in metadata['scenes']:
        item=inputs[name]
        if item['scene_hash']!=metadata['scene_hashes'][name]:raise ValueError('Scene metadata differs')
        envelope=material_envelope(item['guide'],item['permitted'],6)
        contexts[name]=context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],envelope,item['feasible'])
        allowances[name]=endpoint_allowance(item['scene'],item['permitted'])[0]
    random.seed(123);np.random.seed(123);torch.manual_seed(123)
    model=UrbanPavilionNCA(dict(cfg));model.load_state_dict(weights);model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.9)
    generator=torch.Generator().manual_seed(123)
    completed=restore_checkpoint(args.resume,model,optimizer,scheduler,generator,metadata) if args.resume else 0
    if not 0 <= completed < args.stop_after <= metadata['updates']:raise ValueError('Invalid continuation boundary')
    branch=directory/args.worker;branch.mkdir(exist_ok=False)
    trace=[];checkpoints=[]
    for update in range(completed+1,args.stop_after+1):
        scene_draw=int(np.random.randint(len(metadata['scenes'])))
        name=metadata['scenes'][(update-1)%len(metadata['scenes'])];steps=random.randint(2,4)
        global_draw=float(torch.rand(()));item=inputs[name];ctx=contexts[name]
        optimizer.zero_grad(set_to_none=True)
        result=experimental_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',steps,generator)
        state,raw=result['state'],result['raw_material']
        values=research_terms(state,raw,ctx,cfg,allowances[name],LossSpec(**metadata['loss_spec']))
        terms=mean_terms(values)
        loss=weighted_total(values,metadata['loss_weights'],metadata['regularizer_weights'])
        terms.update({k:v.mean() for k,v in values['regularizers'].items()})
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        if (state[:,cfg['ch_structure']][~ctx.permitted]!=0).any():raise ValueError('Illegal material')
        if not torch.equal(state[:,:cfg['n_frozen']],item['seed'][:,:cfg['n_frozen']]):raise ValueError('Frozen context changed')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite weight')
        fields=branch/f'u{update:02d}.npz'
        with fields.open('xb') as stream:np.savez_compressed(stream,state=state.detach().numpy(),raw=raw.detach().numpy())
        field_ref=store.attach(args.run_id,fields,'recovery_fields')
        ckpt=branch/f'u{update:02d}.pt'
        save_checkpoint(ckpt,model,optimizer,scheduler,generator,metadata,update)
        ckpt_ref=store.attach(args.run_id,ckpt,'recovery_checkpoint')
        checkpoints.append({'update':update,'fields':field_ref,'checkpoint':ckpt_ref})
        row={'update':update,'scene':name,'steps':steps,'global_torch_draw':global_draw,'numpy_scene_draw':scene_draw,
             'total_loss':float(loss.detach()),'terms':{name:float(value.detach()) for name,value in terms.items()},
             'gradient_norm_before_clip':float(norm),'learning_rate_after_step':optimizer.param_groups[0]['lr']}
        trace.append(row)
        path=branch/f'u{update:02d}.json';write_once(path,row);store.attach(args.run_id,path,'recovery_update')
        print(f'{args.worker} update={update} scene={name} steps={steps} loss={row["total_loss"]:.8f}',flush=True)
        del state,raw,result,values,terms,loss
    result={'branch':args.worker,'restored_updates':completed,'completed_updates':args.stop_after,'trace':trace,'checkpoints':checkpoints}
    write_once(branch/'result.json',result);store.attach(args.run_id,branch/'result.json','recovery_branch')
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-run');parser.add_argument('--parent-run')
    parser.add_argument('--recipe',choices=['mapped_30','mass_3'],default='mass_3')
    parser.add_argument('--worker',choices=['uninterrupted','prefix','resumed','repeat']);parser.add_argument('--run-id')
    parser.add_argument('--resume');parser.add_argument('--stop-after',type=int,default=4)
    args=parser.parse_args()
    if args.worker:return worker(args)
    if not args.gate_run:parser.error('--gate-run is required')
    store=RunStore(REPO/'.local-artifacts/runs')
    if store.verify(args.gate_run):raise ValueError('K1 evidence corrupt')
    gate_dir=store.path(args.gate_run)
    if read_json(gate_dir/'result.json')['status']!='completed':raise ValueError('K1 incomplete')
    summaries=[read_json(gate_dir/e['details']['path']) for path in (gate_dir/'events').glob('*.json')
               for e in [read_json(path)] if e['kind']=='artifact' and e['details']['role']=='summary']
    if len(summaries)!=1 or summaries[0]['model_cases']!=71 or summaries[0]['probe_cases']!=51:raise ValueError('K1 matrix incomplete')
    proposal=read_json(REPO/'experiments/configs/K2-sensitivity.json')
    if proposal['source_calibration_run']!=args.gate_run:raise ValueError('Recipe source differs')
    recipe=proposal['recipes'][args.recipe]
    cfg,_,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
    scenes=['legacy-easy-seed-000','ref-01-ground-pair','ref-06-minimal-smoke']
    metadata={'protocol':'R2_v1','source_gate_run':args.gate_run,'config':cfg,'checkpoint_sha256':digest(checkpoint),
        'scenes':scenes,'scene_hashes':{n:inputs[n]['scene_hash'] for n in scenes},
        'code_sha256':{p:digest(REPO/p) for p in CODE_FILES},'loss_spec':asdict(LossSpec()),
        'loss_weights':recipe['family_weights'],'regularizer_weights':recipe['regularizer_weights'],'recipe':args.recipe,'scene_sampling':'cyclic; NumPy draw recorded independently','arm':'hard_preclamp','envelope_radius':6,'budget_contract':'envelope',
        'optimizer':'Adam','lr':1e-4,'scheduler':'StepLR(2,0.9)','clip_grad_norm':1.,'seed':123,
        'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),'python_version':sys.version,
        'device':'cpu','threads':2,'updates':4,'rollout_steps':[2,4],
        'scope':'Composed-objective CPU recovery only; candidate coefficients are not validated trained-model weights.'}
    protocol={'protocol':'R2_v1','gate_run':args.gate_run,'checkpoint_metadata':metadata,'paid_compute':False}
    origin=provenance(REPO);run=store.create('R2 composed-objective recovery smoke','optimizer_recovery',protocol,123,origin,parent_run=args.parent_run)
    directory=store.path(run);print('RUN_ID='+run,flush=True)
    status,error='completed',None;results={};checks={};started=time.perf_counter()
    try:
        source=directory/'source.zip';snapshot_source(REPO,source);store.attach(run,source,'source_snapshot');source.unlink()
        write_once(directory/'protocol.json',protocol);store.attach(run,directory/'protocol.json','protocol')
        for branch,stop in [('uninterrupted',4),('prefix',2),('resumed',4),('repeat',4)]:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',branch,'--run-id',run,'--stop-after',str(stop)]
            if branch in ('resumed','repeat'):
                command+=['--resume',str(directory/results['prefix']['checkpoints'][-1]['checkpoint']['path'])]
            process=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
            log=directory/(branch+'.log')
            with log.open('x',encoding='utf-8') as stream:stream.write(process.stdout+process.stderr)
            store.attach(run,log,'worker_log')
            print(process.stdout,flush=True)
            if process.returncode:raise RuntimeError(f'{branch} worker failed: {process.stderr}')
            results[branch]=read_json(directory/branch/'result.json')
        final=lambda name:torch.load(directory/results[name]['checkpoints'][-1]['checkpoint']['path'],weights_only=True,map_location='cpu')
        reference=final('uninterrupted')
        for name in ('resumed','repeat'):
            checks[name+'_full_checkpoint_equal']=tree_equal(reference,final(name))
            checks[name+'_trace_equal']=results['uninterrupted']['trace']==results['prefix']['trace']+results[name]['trace']
            initial=results['uninterrupted']['checkpoints'][:2]
            tail=results[name]['checkpoints']
            checks[name+'_field_arrays_equal']=True
            for left,right in zip(results['uninterrupted']['checkpoints'],results['prefix']['checkpoints']+tail):
                with np.load(directory/left['fields']['path'],allow_pickle=False) as a,np.load(directory/right['fields']['path'],allow_pickle=False) as b:
                    checks[name+'_field_arrays_equal'] &= all(np.array_equal(a[key],b[key]) for key in a.files)
        prefix_payload=torch.load(directory/results['prefix']['checkpoints'][-1]['checkpoint']['path'],weights_only=True)
        reference_u2=torch.load(directory/results['uninterrupted']['checkpoints'][1]['checkpoint']['path'],weights_only=True)
        checks['prefix_checkpoint_equal']=tree_equal(prefix_payload,reference_u2)
        if not all(checks.values()):raise ValueError('Recovery comparison failed')
    except KeyboardInterrupt:status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'checks':checks,'completed_branches':list(results),
             'logical_optimizer_updates':4,'executed_updates':sum(len(r['trace']) for r in results.values()),
             'seconds':time.perf_counter()-started,'gate_run':args.gate_run,'provenance':origin,
             'limits':metadata['scope'],'artifact_location':f'.local-artifacts/runs/{run}'}
    write_once(directory/'summary.json',summary);store.attach(run,directory/'summary.json','summary')
    if error:store.event(run,'error','Recovery attempt stopped; checkpoints and worker logs retained',traceback=error)
    store.finish(run,status,checks,metadata['scope']);write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
