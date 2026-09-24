"""MG1: bounded procedural mass alternatives with unchanged independent scoring."""
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
import argparse, json, sys, time, traceback
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from nca.experiments import RunStore, provenance, snapshot_source, write_once
from nca.mass_generator import generate_mass, MassGeneratorSpec, pairwise_diversity
from nca.mass_generation_cases import generation_scenes, challenge_fields
from nca.massing_cases import target_context
from nca.massing_targets import MassingTargetSpec, evaluate_targets


def as_field(record, shape):
    a=np.zeros(shape,bool)
    if record['occupied_zyx']:a[tuple(np.array(record['occupied_zyx']).T)]=True
    return a


def summarize(records, contexts):
    groups=[]
    for context in contexts:
        scene_records=[r for r in records if r['scene_case']==context['case'] and r['kind']=='generated']
        for request in [None]+sorted({r['request_fraction'] for r in scene_records}):
            selected=[r for r in scene_records if request is None or r['request_fraction']==request]
            valid=[r for r in selected if r['targets']['contract_pass']]
            diversity=pairwise_diversity([as_field(r,(context['scene']['grid_size'],)*3) for r in valid])
            groups.append({'scene_case':context['case'],'request_fraction':request,
                           'candidate_count':len(selected),'invalid_fraction':1-len(valid)/len(selected) if selected else None,
                           'valid_candidate_ids':[r['case'] for r in valid],**diversity})
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    recipe_path=REPO/'experiments/configs/MG1-procedural.json'
    recipe=json.loads(recipe_path.read_bytes());spec=MassingTargetSpec(**recipe['spec'])
    config,_,checkpoint=load_model_c(device='cpu');torch.set_num_threads(2)
    metadata={**provenance(REPO),'effective_config':config,'checkpoint_weights_used':False,
              'checkpoint_sha256':sha256(checkpoint.read_bytes()).hexdigest(),
              'recipe_sha256':sha256(recipe_path.read_bytes()).hexdigest()}
    store=RunStore(REPO/'.local-artifacts/runs')
    run=store.create('MG1 procedural mass alternatives','procedural_massing',recipe,0,metadata,args.parent_run or recipe['parent_run'])
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter()
    contexts=[];records=[];status='completed';failure=None
    tasks=[f'{name}__v{int(round(request*100))}__s{seed}' for name in recipe['contexts'] for request in recipe['volume_requests'] for seed in recipe['seeds']]
    tasks += ['challenge__'+name for name in recipe['challenges']]
    def save(name,value,role):
        write_once(d/name,value);store.attach(run,d/name,role)
    def check_time():
        if time.perf_counter()-started >= recipe['study_seconds_cap']:raise TimeoutError('MG1 study wall cap reached at case boundary')
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','exact_source')
        scenes=dict(generation_scenes())
        for name in recipe['contexts']:
            check_time();scene=scenes[name]
            fields,domain,region=target_context(scene,config,recipe['padding_m_zyx'])
            context={'case':name,'scene':scene,'domain':region,'domain_zyx':np.argwhere(domain).tolist(),
                     'masks':{k:np.argwhere(fields[k]).tolist() for k in ('permitted','protected','existing','support_boundary')}}
            contexts.append(context);save(name+'__context.json',context,'fixed_context')
            for request in recipe['volume_requests']:
                for seed in recipe['seeds']:
                    check_time();case=f'{name}__v{int(round(request*100))}__s{seed}'
                    gen_spec=MassGeneratorSpec(recipe['cube_m'],request,recipe['candidate_seconds_cap'])
                    field,route,generation=generate_mass(scene,fields,domain,seed,gen_spec)
                    t=time.perf_counter();report,masks=evaluate_targets(field,scene,fields,domain,spec)
                    evaluation_seconds=time.perf_counter()-t
                    record={'case':case,'kind':'generated','scene_case':name,'request_fraction':request,'seed':seed,
                            'occupied_zyx':np.argwhere(field).tolist(),'route_zyx':np.argwhere(route).tolist(),
                            'bulk_zyx':np.argwhere(masks['bulk']).tolist(),'field_sha256':sha256(field.tobytes()).hexdigest(),
                            'generation':generation,'targets':report,'evaluation_seconds':evaluation_seconds,'learned':False}
                    records.append(record);save(case+'.json',record,'generated_candidate')
                    print(case,generation['status'],report['contract_pass'],[k for k,v in report['family_pass'].items() if not v],flush=True)
        scene=scenes['aligned'];fields,domain,_=target_context(scene,config,recipe['padding_m_zyx'])
        challenges=challenge_fields(scene,domain)
        for name in recipe['challenges']:
            check_time();field=challenges[name];t=time.perf_counter()
            report,masks=evaluate_targets(field,scene,fields,domain,spec)
            record={'case':'challenge__'+name,'kind':'analytical_challenge','scene_case':'aligned','occupied_zyx':np.argwhere(field).tolist(),
                    'bulk_zyx':np.argwhere(masks['bulk']).tolist(),'targets':report,'evaluation_seconds':time.perf_counter()-t,'learned':False}
            records.append(record);save(record['case']+'.json',record,'analytical_challenge')
            print(record['case'],report['contract_pass'],[k for k,v in report['family_pass'].items() if not v],flush=True)
        check_time()
    except (Exception,KeyboardInterrupt) as exc:
        status='interrupted' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'failed'
        failure=traceback.format_exc();store.event(run,'error',failure)
    executed={r['case'] for r in records};unexecuted=[t for t in tasks if t not in executed]
    if status=='completed' and unexecuted:status='failed';failure='Incomplete planned matrix'
    summary=summarize(records,contexts)
    study={'version':recipe['version'],'run_id':run,'training':False,'recipe':recipe,'contexts':contexts,'cases':records,'groups':summary,
           'unexecuted':unexecuted,'failure':failure,'status':status}
    save('study.json',study,'complete_or_partial_study')
    generated=[r for r in records if r['kind']=='generated']
    metrics={'generated':len(generated),'challenges':len(records)-len(generated),'unexecuted':unexecuted,
             'valid_generated':sum(r['targets']['contract_pass'] for r in generated),'groups':summary,
             'generation_seconds':sum(r['generation']['wall_seconds'] for r in generated),
             'evaluation_seconds':sum(r['evaluation_seconds'] for r in records),'wall_seconds':time.perf_counter()-started}
    interpretation='Procedural development matrix, not learned generation or held-out generalization. Failures retained; MT1 unchanged.'
    store.finish(run,status,metrics,interpretation)
    write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':status,'config':recipe,'metrics':metrics,'provenance':metadata,'artifact_location':d.relative_to(REPO).as_posix(),'drive_backup':'pending','interpretation':interpretation})
    print(json.dumps({'run_id':run,'status':status,**{k:v for k,v in metrics.items() if k!='groups'}},indent=2))
    if failure:print(failure)
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
