"""T1_v1: static target compatibility and direct occupancy gradients, no training."""
import argparse,time,traceback,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,provenance,snapshot_source,write_once,digest
from nca.losses import LossSpec,FAMILIES,context_from_scenes,material_envelope,loss_terms
from nca.interventions import budget_bounds
from nca.target_audit import target_candidates,joint_bounds
from nca.e0 import evaluate
from scripts.diagnostic_inputs import load_inputs,C1_RUN


def with_budget(values,p,ctx,contract,spec):
    terms=dict(values['terms']);bounds=budget_bounds(ctx,contract,spec)
    ratio=(p*ctx.budget).flatten(1).sum(1)/bounds['denominator_voxels'].clamp_min(1)
    terms['sparsity']=150*torch.relu(ratio-spec.max_mass_ratio).square()+torch.relu(spec.min_mass_ratio-ratio)
    return terms


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    c1,inputs=load_inputs(REPO);cfg=c1['effective_model_config'];spec=LossSpec()
    protocol={'protocol':'T1_v1','input_run':C1_RUN,'config':cfg,'checkpoint_sha256':c1['checkpoint_sha256'],
              'scene_hashes':{k:v['scene_hash'] for k,v in inputs.items()},'spec':asdict(spec),
              'candidates':['empty','guide','scaffold','radius1','radius3','radius6'],
              'envelopes':[3,6],'budgets':['site','envelope'],'expected_targets':432,'expected_bounds':72,
              'expected_gradients':36,'gradient_seed':'1000 + sorted scene index','optimizer_updates':0,
              'note':'Static geometry/direct occupancy gradients only. No new objective or production choice.'}
    store=RunStore(REPO/'.local-artifacts/runs');origin=provenance(REPO)
    run=store.create('T1 target compatibility audit','target_audit',protocol,1000,origin,parent_run=args.parent_run)
    directory=store.path(run);print('RUN_ID='+run,flush=True)
    def record(name,data,role):
        path=directory/(name+'.json');write_once(path,data);return store.attach(run,path,role)
    def arrays(name,values):
        path=directory/(name+'.npz')
        with path.open('xb') as f:np.savez_compressed(f,**{k:v.detach().cpu().numpy() for k,v in values.items()})
        ref=store.attach(run,path,'fields');path.unlink();return ref
    rows=[];bounds_rows=[];gradient_rows=[];status,error='completed',None;started=time.perf_counter()
    try:
        path=directory/'source.zip';snapshot_source(REPO,path);store.attach(run,path,'source_snapshot');path.unlink()
        record('protocol',protocol,'protocol')
        for index,(sid,item) in enumerate(sorted(inputs.items())):
            targets=target_candidates(item['guide'],item['scaffold'],item['permitted'])
            envelopes={radius:material_envelope(item['guide'],item['permitted'],radius) for radius in (3,6)}
            contexts={r:context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],envelopes[r],item['feasible']) for r in envelopes}
            ctx=contexts[6]
            fields={'seed':item['seed'],**targets,'envelope3':envelopes[3],'envelope6':envelopes[6],
                    **{name:getattr(ctx,name) for name in ('permitted','facade','support','source','budget','protected')}}
            input_ref=arrays(f'in{index:02d}',fields)
            common={'scene_id':sid,'scene_set':item['scene_set'],'scene_hash':item['scene_hash'],
                    'route_feasible':bool(item['feasible'][0]),'voxel_size_m':item['scene']['voxel_size_m'],'input_fields':input_ref}
            for radius,ctx in contexts.items():
                for contract in ('site','envelope'):
                    bound=joint_bounds(ctx,contract,spec)
                    row={**common,'envelope_radius':radius,'budget':contract,
                         **{k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in bound.items()}}
                    bounds_rows.append(row);record(f'b{len(bounds_rows):03d}',row,'bound_record')
            for name,mask in targets.items():
                p=mask.float();state=item['seed'].clone();state[:,cfg['ch_structure']]=p
                metrics=evaluate(state,cfg,item['scene'])
                for radius,ctx in contexts.items():
                    with torch.no_grad():values=loss_terms(p,ctx,spec)
                    for contract in ('site','envelope'):
                        terms=with_budget(values,p,ctx,contract,spec)
                        numeric={k:float(v[0]) for k,v in terms.items()}
                        if not all(np.isfinite(v) for v in numeric.values()):raise ValueError('Nonfinite target loss')
                        bound=joint_bounds(ctx,contract,spec)
                        row={**common,'candidate':name,'envelope_radius':radius,'budget':contract,'terms':numeric,
                             'context_valid':bool((values['guide_and_envelope_valid'] & values['source_valid'] & bound['budget_compatible'])[0]),
                             'joint_necessary_compatible':bool(bound['joint_necessary_compatible'][0]),
                             'mass_voxels':int(mask.sum()),'volume_m3':int(mask.sum())*item['scene']['voxel_size_m']**3,
                             'facade_voxels':int((mask&ctx.facade).sum()),'metrics':metrics,
                             'all_terms_zero':all(abs(v)<=1e-7 for v in numeric.values()) and bool(mask.any())}
                        rows.append(row);record(f't{len(rows):03d}',row,'target_record')
            ctx=contexts[6];generator=torch.Generator().manual_seed(1000+index)
            draw=torch.rand(ctx.permitted.shape,generator=generator)
            p=torch.where(ctx.coverage,.7+.2*draw,torch.where(ctx.envelope,.01+.04*draw,0.)).requires_grad_()
            values=loss_terms(p,ctx,spec)
            for contract in ('site','envelope'):
                terms=with_budget(values,p,ctx,contract,spec);gradients={};norms={};vectors={}
                for name,term in terms.items():
                    grad,=torch.autograd.grad(term.sum(),p,retain_graph=True)
                    if not torch.isfinite(grad).all():raise ValueError('Nonfinite gradient')
                    gradients[name]=grad;legal=(grad*ctx.permitted).double().flatten();vectors[name]=legal
                    norms[name]={'full_l2':float(grad.double().norm()),'legal_l2':float(legal.norm()),'value':float(term.detach()[0])}
                cosine={a:{b:float(torch.dot(vectors[a],vectors[b])/(vectors[a].norm()*vectors[b].norm()))
                            if vectors[a].norm()*vectors[b].norm()>0 else None for b in FAMILIES} for a in FAMILIES}
                ref=arrays(f'g{len(gradient_rows):02d}',{'occupancy':p,**gradients})
                bound=joint_bounds(ctx,contract,spec)
                row={**common,'budget':contract,'envelope_radius':6,'fields':ref,'gradient_norms':norms,'cosines':cosine,
                     'context_valid':bool((values['guide_and_envelope_valid'] & values['source_valid'] & bound['budget_compatible'])[0]),
                     'scope':'direct occupancy gradients; no NCA parameter or learning claim'}
                gradient_rows.append(row);record(f'g{len(gradient_rows):02d}',row,'gradient_record')
            print(f'SCENE {index+1}/18 {sid}: targets={len(rows)} gradients={len(gradient_rows)}',flush=True)
        if (len(rows),len(bounds_rows),len(gradient_rows))!=(432,72,36):raise ValueError('Incomplete matrix')
    except KeyboardInterrupt:status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'targets':len(rows),'bounds':len(bounds_rows),
             'gradients':len(gradient_rows),'optimizer_updates':0,'seconds':time.perf_counter()-started,'provenance':origin,
             'zero_term_candidates':[{k:r[k] for k in ('scene_id','candidate','envelope_radius','budget')} for r in rows if r['all_terms_zero'] and r['context_valid']],
             'artifact_location':f'.local-artifacts/runs/{run}'}
    record('summary',summary,'summary')
    if error:store.event(run,'error','Audit stopped; evidence retained',traceback=error)
    store.finish(run,status,{'targets':len(rows),'bounds':len(bounds_rows),'gradients':len(gradient_rows)},protocol['note'])
    write_once(REPO/'experiments/records'/(run+'.json'),summary);print(summary,flush=True)
    return 0 if status=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
