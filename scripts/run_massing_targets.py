"""MT1: freeze pilot target semantics and retain every counterexample/sensitivity."""
from pathlib import Path
from dataclasses import replace
from hashlib import sha256
import argparse,json,sys,time,traceback
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from nca.experiments import RunStore,provenance,snapshot_source,write_once
from nca.massing_targets import MassingTargetSpec,evaluate_targets
from nca.massing_cases import target_scenes,target_context,target_controls
from nca.spatial import legacy_diagnostics
from deploy.checkpoints import load_model_c


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/MT1-targets.json').read_bytes())
    spec=MassingTargetSpec(**recipe['spec']);config,_,checkpoint=load_model_c(device='cpu')
    torch.set_num_threads(2);torch.manual_seed(recipe['seed']);np.random.seed(recipe['seed'])
    metadata={**provenance(REPO),'effective_config':config,'checkpoint_weights_used':False,
              'checkpoint_sha256':sha256(checkpoint.read_bytes()).hexdigest()}
    store=RunStore(REPO/'.local-artifacts/runs');run=store.create('MT1 massing target contract','binary_contract_audit',recipe,recipe['seed'],metadata,args.parent_run or recipe['parent_run'])
    d=store.path(run);print('RUN_ID='+run,flush=True);start=time.perf_counter()
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','exact_source')
        records=[];contexts=[];checks={};sensitivity=[]
        for name,scene in target_scenes():
            fields,domain,region=target_context(scene,config,recipe['padding_m_zyx'])
            context={'case':name,'scene':scene,'domain':region,'domain_zyx':np.argwhere(domain).tolist(),
                     'masks':{k:np.argwhere(fields[k]).tolist() for k in ('permitted','protected','existing','support_boundary')}}
            contexts.append(context);write_once(d/(name+'__scene.json'),context);store.attach(run,d/(name+'__scene.json'),'fixed_context')
            for case,field in target_controls(scene,domain).items():
                report,masks=evaluate_targets(field,scene,fields,domain,spec)
                old=legacy_diagnostics(scene,field,config)
                expected=case in recipe['positive_controls'] and name not in recipe['infeasible_contexts']
                record={'scene_case':name,'case':case,'occupied_zyx':np.argwhere(field).tolist(),
                        'bulk_zyx':np.argwhere(masks['bulk']).tolist(),'targets':report,'legacy':old,
                        'expected_contract_pass':expected,'learned':False}
                records.append(record);write_once(d/(name+'__'+case+'.json'),record);store.attach(run,d/(name+'__'+case+'.json'),'individual_control')
                checks[name+'/'+case+'/expected']=report['contract_pass']==expected
                checks[name+'/'+case+'/facade_parity']=abs(report['facade_excess']-old['families']['facade'])<=1e-6
                for cube in recipe['sensitivity']['min_cube_m']:
                    for cap in recipe['sensitivity']['max_volume_fraction']:
                        variant=replace(spec,min_cube_m=cube,max_volume_fraction=cap)
                        target=report if variant==spec else evaluate_targets(field,scene,fields,domain,variant)[0]
                        sensitivity.append({'scene_case':name,'case':case,'cube_m':cube,'max_volume_fraction':cap,'targets':target})
                print(name,case,report['contract_pass'],[k for k,v in report['family_pass'].items() if not v],flush=True)
        study={'version':recipe['version'],'run_id':run,'training':False,'recipe':recipe,'contexts':contexts,'cases':records,'checks':checks}
        write_once(d/'study.json',study);store.attach(run,d/'study.json','gallery_data')
        write_once(d/'sensitivity.json',sensitivity);store.attach(run,d/'sensitivity.json','all_sensitivity_reports')
        metrics={'scenes':len(contexts),'cases':len(records),'sensitivity_cases':len(sensitivity),
                 'contract_passes':sum(r['targets']['contract_pass'] for r in records),'checks':checks,
                 'all_expected_outcomes_match':all(checks.values()),'wall_seconds':time.perf_counter()-start}
        status='completed' if all(checks.values()) else 'failed';store.finish(run,status,metrics,'Pilot binary target audit; no training, architectural quality or generalization claim.')
        write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':status,'config':recipe,'metrics':metrics,'provenance':metadata,'artifact_location':d.relative_to(REPO).as_posix(),'drive_backup':'pending'})
        print(json.dumps({'run_id':run,'status':status,**{k:v for k,v in metrics.items() if k!='checks'}},indent=2))
        return 0 if status=='completed' else 1
    except Exception:
        error=traceback.format_exc();store.event(run,'error',error)
        if not (d/'result.json').exists():store.finish(run,'failed',{},error)
        raise


if __name__=='__main__':raise SystemExit(main())
