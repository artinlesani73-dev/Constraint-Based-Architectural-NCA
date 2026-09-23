"""L2_v1 budget and material-gradient comparisons; no optimization."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,provenance,snapshot_source,write_once,digest
from nca.losses import LossSpec,context_from_scenes,material_envelope,soft_reach
from nca.interventions import ARMS,experimental_rollout,guidance_loss,budget_bounds
from nca.e0 import evaluate
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from scripts.diagnostic_inputs import load_inputs,C1_RUN


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    cfg,weights,checkpoint=load_model_c();spec=LossSpec()
    protocol={'protocol':'L2_v1','input_run':C1_RUN,'config':cfg,'checkpoint_sha256':digest(checkpoint),
        'loss_spec':asdict(spec),'arms':list(ARMS),'gradient_scenes':['legacy-easy-seed-000','ref-01-ground-pair','ref-06-minimal-smoke'],
        'seeds':[0,1,2],'horizons':[4,16],'beta':20.,'expected_budget_cases':108,'expected_gradient_cases':54,
        'expected_zero_controls':9,'optimizer_updates':0,'device':'cpu','threads':2,
        'note':'Envelope denominator changes absolute budgets explicitly; no research trainer is selected.'}
    store=RunStore(REPO/'.local-artifacts/runs');origin=provenance(REPO)
    run=store.create('L2 material intervention and budget contracts','intervention_diagnostic',protocol,0,origin,parent_run=args.parent_run)
    directory=store.path(run);print('RUN_ID='+run,flush=True)
    budget_rows=[];cases=[];status='completed';error=None;start=time.perf_counter()
    def record(name,data,role):
        path=directory/(name+'.json');write_once(path,data);return store.attach(run,path,role)
    def arrays(name,**data):
        path=directory/(name+'.npz')
        with path.open('xb') as stream:np.savez_compressed(stream,**{k:v.detach().cpu().numpy() for k,v in data.items()})
        ref=store.attach(run,path,'fields');path.unlink();return ref
    try:
        path=directory/'source.zip';snapshot_source(REPO,path);store.attach(run,path,'source_snapshot');path.unlink()
        record('protocol',protocol,'protocol')
        c1,inputs=load_inputs(REPO)
        if cfg!=c1['effective_model_config'] or protocol['checkpoint_sha256']!=c1['checkpoint_sha256']:raise ValueError('Checkpoint/config drift')
        for sid,item in sorted(inputs.items()):
            envelopes={'scaffold':item['scaffold']>.5,'radius3':material_envelope(item['guide'],item['permitted'],3),
                       'radius6':material_envelope(item['guide'],item['permitted'],6)}
            input_ref=arrays('input-'+str(len(budget_rows)),seed=item['seed'],guide=item['guide'],scaffold=item['scaffold'],**{'envelope_'+name:value for name,value in envelopes.items()})
            for ename,envelope in envelopes.items():
                ctx=context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],envelope,item['feasible'])
                for contract in ('site','envelope'):
                    b=budget_bounds(ctx,contract,spec)
                    row={'scene_id':sid,'scene_set':item['scene_set'],'scene_hash':item['scene_hash'],'envelope':ename,
                         'contract':contract,'route_feasible':bool(item['feasible'][0]),'source_fields':item['source_fields'],'fields':input_ref,
                         **{k:v.tolist() for k,v in b.items() if isinstance(v,torch.Tensor)}}
                    row['valid_context']=row['route_feasible'] and row['budget_compatible'][0]
                    row['minimum_volume_m3']=row['minimum_mass'][0]*item['scene']['voxel_size_m']**3
                    row['maximum_volume_m3']=row['maximum_mass'][0]*item['scene']['voxel_size_m']**3
                    record(f'budget-{len(budget_rows):03d}',row,'budget_record');budget_rows.append(row)
            item['context']=context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],envelopes['radius6'],item['feasible'])
        model=UrbanPavilionNCA(dict(cfg));model.load_state_dict(weights)
        for zero in (False,True):
            for sid in protocol['gradient_scenes']:
                item=inputs[sid];ctx=item['context']
                for seed in ([0] if zero else protocol['seeds']):
                    for steps in ([4] if zero else protocol['horizons']):
                        baseline=None
                        for arm in ARMS:
                            generator=torch.Generator().manual_seed(seed)
                            scaffold=torch.zeros_like(item['scaffold']) if zero else item['scaffold']
                            tick=time.perf_counter()
                            rollout=experimental_rollout(model,item['seed'],scaffold,arm,steps,generator,beta=20.)
                            final,raw=rollout['state'],rollout['raw_material']
                            coverage=guidance_loss(final,raw,item['guide'],cfg,arm).mean()
                            reached=soft_reach(final[:,cfg['ch_structure']]*ctx.permitted,ctx.source,64)
                            access=1-torch.stack([reached[0][region].max() for name,region in sorted(ctx.endpoints[0].items()) if name!=ctx.source_ids[0]]).mean()
                            gradients={};norms={}
                            for name,term in [('coverage',coverage),('access',access)]:
                                gradient=torch.autograd.grad(term,(raw,*model.parameters()),retain_graph=True,allow_unused=True)
                                rg=gradient[0] if gradient[0] is not None else torch.zeros_like(raw)
                                pg=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(gradient[1:],model.parameters())])
                                if not torch.isfinite(rg).all() or not torch.isfinite(pg).all():raise ValueError('Nonfinite derivative')
                                gradients[name+'_raw']=rg;gradients[name+'_parameters']=pg
                                norms[name]={'raw_l2':float(rg.double().norm()),'parameter_l2':float(pg.double().norm()),'value':float(term.detach())}
                            equal=None
                            if arm=='hard_projected':baseline=final.detach().clone()
                            elif arm=='hard_preclamp':
                                equal=torch.equal(final,baseline)
                                if not equal:raise ValueError('Hard-arm forward mismatch')
                            metrics=evaluate(final,cfg,item['scene'])
                            if metrics['legality']['illegal_voxels'] or not torch.equal(final[:,:cfg['n_frozen']],item['seed'][:,:cfg['n_frozen']]):raise ValueError('Legality/frozen context changed')
                            ref=arrays(f'case-{len(cases):03d}',seed=item['seed'],guide=item['guide'],scaffold=scaffold,state=final,raw=raw,**gradients)
                            row={'scene_id':sid,'seed':seed,'steps':steps,'arm':arm,'zero_scaffold':zero,'fields':ref,
                                 'hard_forward_equal':equal,'gradients':norms,'metrics':metrics,
                                 'soft_mass':float(final[:,cfg['ch_structure']].detach().sum()),'seconds':time.perf_counter()-tick}
                            if sid=='ref-01-ground-pair':
                                row['known_dead_cells']=[{'zyx':[z,15,9],'raw':float(raw[0,z,15,9].detach()),
                                    'material':float(final[0,cfg['ch_structure'],z,15,9].detach()),
                                    'coverage_raw_derivative':float(gradients['coverage_raw'][0,z,15,9]),
                                    'access_raw_derivative':float(gradients['access_raw'][0,z,15,9])} for z in (0,1)]
                            record(f'case-{len(cases):03d}',row,'gradient_record');cases.append(row)
                            print(f'GRADIENTS {len(cases)}/63 {sid} {seed} {steps} {arm} zero={zero}',flush=True)
                            del final,raw,rollout,gradients,gradient,rg,pg,coverage,access,reached,term
        if not all(torch.equal(v,weights[k]) for k,v in model.state_dict().items()):raise ValueError('Checkpoint weights changed')
        if len(budget_rows)!=108 or len(cases)!=63:raise ValueError('Incomplete matrix')
    except KeyboardInterrupt:status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    budget_summary={f'{ename}/{contract}':{'cases':len(rows),'valid':sum(r['valid_context'] for r in rows),
                      'compatible':sum(r['budget_compatible'][0] for r in rows)}
                    for ename in ('scaffold','radius3','radius6') for contract in ('site','envelope')
                    for rows in [[r for r in budget_rows if r['envelope']==ename and r['contract']==contract]]}
    known=[r for r in cases if r['scene_id']=='ref-01-ground-pair' and r['seed']==0 and r['steps']==4 and r['arm']=='hard_preclamp' and not r['zero_scaffold']]
    gate=(status=='completed' and budget_summary['radius6/envelope']['valid']==17 and len(known)==1
          and known[0]['gradients']['coverage']['parameter_l2']>0 and all(c['coverage_raw_derivative']!=0 for c in known[0]['known_dead_cells']))
    summary={'run_id':run,'status':status,'error':error,'budget_cases':len(budget_rows),'gradient_cases':sum(not c['zero_scaffold'] for c in cases),
             'zero_controls':sum(c['zero_scaffold'] for c in cases),'budget_summary':budget_summary,'recovery_mechanics_gate':gate,
             'optimizer_updates':0,'seconds':time.perf_counter()-start,'provenance':origin,'artifact_location':f'.local-artifacts/runs/{run}'}
    record('summary',summary,'summary')
    if error:store.event(run,'error','Run stopped; preceding evidence retained',traceback=error)
    store.finish(run,status,{'budget_cases':len(budget_rows),'gradient_cases':len(cases),'recovery_mechanics_gate':gate},'Local intervention diagnosis, no optimizer')
    write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
