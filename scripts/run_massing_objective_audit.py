"""MO1 retained endpoint agreement and gradient evidence; no optimization."""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,time,traceback
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
from nca.experiments import RunStore,write_once,provenance,snapshot_source
from nca.massing_objective import make_context,massing_residuals
from nca.massing_targets import MassingTargetSpec


def grid(coords,size):
    field=np.zeros((size,)*3,bool)
    if coords:field[tuple(np.array(coords).T)]=True
    return field


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/MO1-objective.json').read_bytes())
    torch.set_num_threads(recipe['threads']);torch.manual_seed(recipe['seed'])
    store=RunStore(REPO/'.local-artifacts/runs');metadata=provenance(REPO)
    run=store.create('MO1 massing objective admission','objective_audit',recipe,recipe['seed'],metadata,args.parent_run or recipe['parent_run'])
    d=store.path(run);print('RUN_ID='+run,flush=True);start=time.perf_counter()
    records=[];gradients=[];status='completed';error=None
    def save(name,value,role):write_once(d/name,value);store.attach(run,d/name,role)
    def clock_check():
        if time.perf_counter()-start>=recipe['wall_seconds_cap']:raise TimeoutError('MO1 wall cap at case boundary')
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','exact_source')
        for parent in recipe['sources']:
            assert not store.verify(parent)
            source=store.path(parent)/'study.json';data=json.loads(source.read_bytes())
            save(parent+'__study.json',data,'audited_input_study')
            for context in data['contexts']:
                clock_check();scene=context['scene'];n=scene['grid_size']
                domain=grid(context['domain_zyx'],n);fields={k:grid(v,n) for k,v in context['masks'].items()}
                ctx=make_context(scene,fields,domain,MassingTargetSpec(**data['recipe']['spec']))
                for original in [x for x in data['cases'] if x['scene_case']==context['case']]:
                    clock_check();field=grid(original['occupied_zyx'],n);p=torch.tensor(field,dtype=torch.double)
                    terms,bulk=massing_residuals(p,ctx)
                    values={k:float(v.detach()) for k,v in terms.items()}
                    agreement={k:(values[k]<=recipe['binary_zero_tolerance'])==original['targets']['family_pass'][k] for k in values}
                    bulk_match=np.array_equal(bulk.detach().numpy(),grid(original['bulk_zyx'],n))
                    record={'source_run':parent,'source_study_sha256':sha256(source.read_bytes()).hexdigest(),
                            'case':original['case'],'scene_case':context['case'],'residuals':values,'family_agreement':agreement,'bulk_exact':bulk_match,
                            'binary_family_pass':original['targets']['family_pass'],'field_sha256':sha256(field.tobytes()).hexdigest()}
                    records.append(record);save(f'{parent}__{context["case"]}__{original["case"]}.json',record,'endpoint_comparison')
                    if not all(agreement.values()) or not bulk_match:print('MISMATCH',record,flush=True)
                    if parent==recipe['sources'][-1] and original['case'] in recipe['gradient_cases']:
                        soft=(recipe['soft_background']+recipe['soft_scale']*p).requires_grad_()
                        soft_terms,_=massing_residuals(soft,ctx);norms={};finite={};facade_grad=None
                        for key,value in soft_terms.items():
                            grad=torch.autograd.grad(value,soft,retain_graph=True)[0]
                            norms[key]=float(grad.norm());finite[key]=bool(torch.isfinite(grad).all())
                            if key=='facade':facade_grad=grad
                        direction=-facade_grad/facade_grad.abs().max().clamp_min(1e-30)
                        eps=recipe['difference_epsilon']
                        plus=massing_residuals((soft.detach()+eps*direction),ctx)[0]['facade']
                        minus=massing_residuals((soft.detach()-eps*direction),ctx)[0]['facade']
                        numerical=float((plus-minus)/(2*eps));analytic=float((facade_grad*direction).sum())
                        agree=bool(np.isclose(analytic,numerical,atol=recipe['derivative_atol'],rtol=recipe['derivative_rtol']))
                        item={'case':original['case'],'soft_residuals':{k:float(v.detach()) for k,v in soft_terms.items()},'gradient_norms':norms,
                              'gradient_finite':finite,'facade_directional_analytic':analytic,'facade_directional_numerical':numerical,
                              'derivative_agrees':agree,'facade_gradient_nonzero':norms['facade']>0,
                              'interpretation':'Fixed-input derivative probe; no optimizer update or geometry improvement.'}
                        gradients.append(item);save(original['case']+'__gradients.json',item,'gradient_probe')
            print('SOURCE_COMPLETE',parent,len(records),flush=True)
        clock_check()
        if len(records)!=97 or len(gradients)!=2:raise AssertionError('Incomplete frozen audit matrix')
        if not all(all(r['family_agreement'].values()) and r['bulk_exact'] for r in records):status='failed'
        if not all(g['derivative_agrees'] and all(g['gradient_finite'].values()) for g in gradients):status='failed'
        if not next(g for g in gradients if g['case']=='partial_obstruction__v24__s0')['facade_gradient_nonzero']:status='failed'
    except (Exception,KeyboardInterrupt) as exc:
        status='interrupted' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'failed'
        error=traceback.format_exc();store.event(run,'error',error)
    study={'version':recipe['version'],'run_id':run,'recipe':recipe,'records':records,'gradients':gradients,'status':status,'error':error}
    save('study.json',study,'complete_or_partial_audit')
    metrics={'fields':len(records),'family_comparisons':sum(len(r['family_agreement']) for r in records),
             'family_mismatches':sum(not v for r in records for v in r['family_agreement'].values()),'bulk_mismatches':sum(not r['bulk_exact'] for r in records),
             'gradient_probes':len(gradients),'wall_seconds':time.perf_counter()-start,'optimizer_updates':0}
    store.finish(run,status,metrics,'CPU endpoint/gradient admission only; no weights, optimizer or NCA model selected.')
    write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':status,'config':recipe,'metrics':metrics,'provenance':metadata,'artifact_location':d.relative_to(REPO).as_posix(),'drive_backup':'pending'})
    print(json.dumps({'run_id':run,'status':status,**metrics},indent=2))
    if error:print(error)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
