"""K1_v1 actual parameter gradients and retained-regularizer calibration evidence."""
import argparse,ast,json,sys,time,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,provenance,snapshot_source,write_once,digest
from nca.losses import LossSpec,FAMILIES,context_from_scenes,material_envelope,mean_terms,loss_terms
from nca.interventions import experimental_rollout,objective_terms
from nca.facade import endpoint_allowance,facade_term
from nca.regularizers import regularizer_terms,NAMES
from nca.e0 import evaluate
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from scripts.diagnostic_inputs import load_inputs
from scripts.run_target_audit import with_budget
from scripts.report_witnesses import load as verify_w1,RUN as W1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);verify_w1()
    cfg,weights,checkpoint=load_model_c();_,inputs=load_inputs(REPO);spec=LossSpec()
    excluded=[sid for sid,item in inputs.items() if not bool(item['feasible'][0])]
    included=sorted(set(inputs)-set(excluded))
    model=UrbanPavilionNCA(dict(cfg));model.load_state_dict(weights);model.train()
    parameters=list(model.named_parameters());layout=[{'name':n,'shape':list(p.shape),'elements':p.numel()} for n,p in parameters]
    names=list(FAMILIES)+list(NAMES)
    notebook=REPO/'notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb';cells=json.loads(notebook.read_text())['cells']
    source=''.join(cells[19]['source']);tree=ast.parse(source)
    definitions={name:ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)) for name in ('DensityPenalty','TotalVariation3D','CantileverLoss')}
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu')['weights']
    trainer=ast.parse(''.join(cells[24]['source']))
    notebook_weights=next(ast.literal_eval(n.value) for n in ast.walk(trainer) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='weights' for t in n.targets))
    assert saved==notebook_weights
    protocol={'protocol':'K1_v1','input_run':W1,'config':cfg,'spec':asdict(spec),'checkpoint_sha256':digest(checkpoint),'notebook_sha256':digest(notebook),
              'included_scenes':included,'excluded_infeasible_scenes':excluded,'seeds':[0,1],'horizons':[4,16],
              'extra_horizon50_scenes':['legacy-easy-seed-000','ref-01-ground-pair','ref-06-minimal-smoke'],
              'names':names,'parameter_layout':layout,'expected_model_cases':71,'expected_probe_cases':51,'optimizer_updates':0,
              'objective':'hard_preclamp/radius6/envelope/facade_endpoint_v1','source_regularizer_definitions':definitions,'checkpoint_weights':saved}
    store=RunStore(REPO/'.local-artifacts/runs');origin=provenance(REPO)
    run=store.create('K1 parameter-gradient calibration','calibration_diagnostic',protocol,0,origin,parent_run=args.parent_run)
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter();cases=[];probes=[];status,error='completed',None
    def record(name,obj,role):
        p=d/(name+'.json');write_once(p,obj);return store.attach(run,p,role)
    def arrays(name,values):
        p=d/(name+'.npz')
        with p.open('xb') as f:np.savez_compressed(f,**{k:v.detach().cpu().numpy() for k,v in values.items()})
        ref=store.attach(run,p,'fields');p.unlink();return ref
    try:
        p=d/'source.zip';snapshot_source(REPO,p);store.attach(run,p,'source_snapshot');p.unlink();record('protocol',protocol,'protocol')
        for sid in included:
            item=inputs[sid];ctx=context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],material_envelope(item['guide'],item['permitted'],6),item['feasible'])
            allowance,annotation=endpoint_allowance(item['scene'],item['permitted'])
            inputref=arrays(f'i{len(probes):02d}',{'seed':item['seed'],'guide':item['guide'],'scaffold':item['scaffold'],'envelope':ctx.envelope,'allowance':allowance})
            common={'scene_id':sid,'scene_hash':item['scene_hash'],'input_fields':inputref}
            for ratio in (.015,.075,.20):
                p=(ctx.envelope.float()*ratio).requires_grad_();terms=with_budget(loss_terms(p,ctx,spec),p,ctx,'envelope',spec)
                terms['facade']=facade_term(p,ctx,allowance,spec);terms.update(regularizer_terms(p,ctx.support))
                grad,=torch.autograd.grad(terms['sparsity'].sum(),p)
                ref=arrays(f'p{len(probes):02d}',{'occupancy':p,'sparsity_gradient':grad})
                row={**common,'ratio':ratio,'terms':{k:float(v[0].detach()) for k,v in terms.items()},'fields':ref,
                     'sparsity_gradient_l2':float(grad.double().norm()),'sparsity_gradient_sum':float(grad.sum()),'mass_ratio':float((p*ctx.budget).sum()/ctx.envelope.sum())}
                probes.append(row);record(f'p{len(probes):02d}',row,'probe_record')
                del terms,p,grad
            configs=[(seed,h) for seed in (0,1) for h in (4,16)]
            if sid in protocol['extra_horizon50_scenes']:configs.append((0,50))
            for seed,steps in configs:
                tick=time.perf_counter();roll=experimental_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',steps,torch.Generator().manual_seed(seed))
                state,raw=roll['state'],roll['raw_material'];p=state[:,cfg['ch_structure']]
                result=objective_terms(state,raw,ctx,cfg,'hard_preclamp','envelope',spec)
                result['terms']['facade']=facade_term(p,ctx,allowance,spec);terms=mean_terms(result)
                terms.update({k:v.mean() for k,v in regularizer_terms(p,ctx.support).items()})
                vectors={};norms={}
                for name,term in terms.items():
                    grads=torch.autograd.grad(term,tuple(p for _,p in parameters),retain_graph=True,allow_unused=True)
                    vector=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,(_,p) in zip(grads,parameters)])
                    if not torch.isfinite(vector).all() or not torch.isfinite(term):raise ValueError('Nonfinite diagnostic')
                    vectors[name]=vector;norms[name]={'value':float(term.detach()),'parameter_l2':float(vector.double().norm())}
                coverage_raw,=torch.autograd.grad(terms['coverage'],raw,retain_graph=True)
                guide_raw=raw.detach()[ctx.coverage]
                saturation={'below_zero_fraction':float((guide_raw<0).float().mean()),'at_or_above_one_fraction':float((guide_raw>=1).float().mean()),
                            'maximum_raw':float(guide_raw.max()),'raw_coverage_gradient_l2':float(coverage_raw.double().norm())}
                cosines={a:{b:float(torch.dot(vectors[a].double(),vectors[b].double())/(vectors[a].double().norm()*vectors[b].double().norm()))
                           if norms[a]['parameter_l2']*norms[b]['parameter_l2'] else None for b in names} for a in names}
                if not torch.equal(state[:,:cfg['n_frozen']],item['seed'][:,:cfg['n_frozen']]) or (p[~ctx.permitted]!=0).any():raise ValueError('Context/legality drift')
                ref=arrays(f'm{len(cases):02d}',{'material':p,'raw':raw,'coverage_raw_gradient':coverage_raw,**vectors})
                row={**common,'seed':seed,'steps':steps,'fields':ref,'norms':norms,'cosines':cosines,'coverage_saturation':saturation,
                     'metrics':evaluate(state,cfg,item['scene']),'mass_ratio':float(result['mass_ratio'][0].detach()),'seconds':time.perf_counter()-tick}
                cases.append(row);record(f'm{len(cases):02d}',row,'model_record')
                print(f'MODEL {len(cases)}/71 {sid} seed={seed} steps={steps}',flush=True)
                del state,raw,p,roll,result,terms,vectors,grads,vector,coverage_raw
        assert len(cases)==71 and len(probes)==51
        assert all(torch.equal(value,weights[name]) for name,value in model.state_dict().items())
    except KeyboardInterrupt:status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'model_cases':len(cases),'probe_cases':len(probes),'excluded_scenes':excluded,
             'optimizer_updates':0,'seconds':time.perf_counter()-started,'provenance':origin,'artifact_location':f'.local-artifacts/runs/{run}'}
    record('summary',summary,'summary')
    if error:store.event(run,'error','Diagnostic stopped; evidence preserved',traceback=error)
    store.finish(run,status,{'model_cases':len(cases),'probe_cases':len(probes)},'Actual derivatives, no optimizer or weight selection')
    write_once(REPO/'experiments/records'/(run+'.json'),summary);print(summary,flush=True)
    return 0 if status=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
