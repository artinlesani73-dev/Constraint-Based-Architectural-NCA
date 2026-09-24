"""MG5 conditional four-case/225-case coverage-prioritized comparison."""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,write_once,provenance,snapshot_source,digest
from nca.coverage_mass_generator import generate_coverage_mass,CoverageGeneratorSpec
from nca.massing_targets import evaluate_targets,MassingTargetSpec
from run_site_generalization import MemorySampler,memory_bytes,diversity


def grid(coords):
    a=np.zeros((32,32,32),bool)
    if coords:a[tuple(np.array(coords).T)]=True
    return a


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['diagnostic','matrix'],required=True)
    parser.add_argument('--diagnostic-run');parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/MG5-coverage.json').read_bytes())
    for name,expected in recipe['source_sha256'].items():
        if digest(REPO/name)!=expected:raise ValueError('Frozen source changed: '+name)
    torch.set_num_threads(recipe['threads']);store=RunStore(REPO/'.local-artifacts/runs')
    for run in (recipe['mg3_run'],recipe['mg4_run']):
        if store.verify(run):raise ValueError('Baseline evidence corrupt')
    if args.mode=='matrix':
        if not args.diagnostic_run or store.verify(args.diagnostic_run):raise ValueError('Verified diagnostic required')
        diagnostic=json.loads((store.path(args.diagnostic_run)/'study.json').read_bytes())
        if diagnostic['mode']!='diagnostic' or not diagnostic['metrics']['admission_gate'] or diagnostic['recipe']!=recipe:raise ValueError('Diagnostic gate/config mismatch')
    studies={key:json.loads((store.path(recipe[key+'_run'])/'study.json').read_bytes()) for key in ('mg3','mg4')}
    tasks=recipe['diagnostic'] if args.mode=='diagnostic' else [{'source':source,'case':c['case']} for source in ('mg3','mg4') for c in studies[source]['cases']]
    run=store.create('MG5 coverage growth '+args.mode,'procedural_massing',{'recipe':recipe,'mode':args.mode,'diagnostic_run':args.diagnostic_run},0,
        provenance(REPO),args.parent_run or args.diagnostic_run or recipe['mg4_run'])
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter();rows=[];contexts={};valid_groups={};before_groups={}
    status='completed';failure=None;cap=recipe['diagnostic_seconds_cap'] if args.mode=='diagnostic' else recipe['matrix_seconds_cap']
    def save(name,data,role):write_once(d/name,data);store.attach(run,d/name,role)
    def check():
        if time.perf_counter()-started>cap:raise TimeoutError('Study cap at case boundary')
        if memory_bytes()['rss_bytes']>recipe['rss_stop_bytes']:raise MemoryError('RSS stopping threshold')
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','exact_source')
        for name in ('docs/next-phase/COVERAGE_GROWTH_PROTOCOL.md','experiments/configs/MG5-coverage.json'):store.attach(run,REPO/name,'frozen_protocol_config')
        for task in tasks:
            check();source=task['source'];stub=next(c for c in studies[source]['cases'] if c['case']==task['case'])
            root=store.path(recipe[source+'_run']);previous=stub if source=='mg3' else json.loads((root/stub['path']).read_bytes())
            context_key=source+'__'+previous['scene_case'];case=source+'__'+previous['case']
            if context_key not in contexts:
                context=next(c for c in studies['mg3']['contexts'] if c['case']==previous['scene_case']) if source=='mg3' else json.loads((root/previous['context_path']).read_bytes())
                context_path='contexts/'+context_key+'.json';save(context_path,context,'unchanged_context')
                contexts[context_key]={'path':context_path,'sha256':digest(d/context_path)}
            context=json.loads((d/contexts[context_key]['path']).read_bytes());scene=context['scene'];domain=grid(context['domain_zyx']);fields={k:grid(v) for k,v in context['masks'].items()}
            before,_=evaluate_targets(grid(previous['occupied_zyx']),scene,fields,domain)
            if before!=previous['targets']:raise ValueError('Baseline score changed')
            fraction=previous['request_fraction'];seed=previous['seed'];partition=previous['scene_case']=='blocked_gap' if source=='mg3' else previous['partition_control']
            base={'case':case,'source':source,'baseline_case':previous['case'],'context_key':context_key,'request_fraction':fraction,'seed':seed,'partition_control':partition}
            with MemorySampler(recipe['rss_sample_seconds']) as sampler:
                cpu=time.process_time();t=time.perf_counter()
                field,route,generation=generate_coverage_mass(scene,fields,domain,seed,CoverageGeneratorSpec(recipe['cube_m'],fraction,recipe['candidate_seconds_cap'],recipe['contact_weight']))
                eval_start=time.perf_counter();targets,masks=evaluate_targets(field,scene,fields,domain,MassingTargetSpec());evaluation=time.perf_counter()-eval_start
                wall=time.perf_counter()-t;cpu=time.process_time()-cpu
            resources={**sampler.report(),'cpu_seconds':cpu,'generation_evaluation_wall_seconds':wall}
            old_route=previous['route_zyx'];same_route=np.array_equal(route,grid(old_route));request_met=0<=generation['target_error_voxels']<27
            record={**base,'occupied_zyx':np.argwhere(field).tolist(),'route_zyx':np.argwhere(route).tolist(),'bulk_zyx':np.argwhere(masks['bulk']).tolist(),
                'field_sha256':sha256(field.tobytes()).hexdigest(),'generation':generation,'targets':targets,'previous':previous,
                'evaluation_seconds':evaluation,'resources':resources}
            path='cases/'+case+'.json';save(path,record,'paired_candidate')
            row={**base,'path':path,'sha256':digest(d/path),'pass':targets['contract_pass'],'previous_pass':before['contract_pass'],
                'same_route':same_route,'request_met':request_met,'request_error_voxels':generation['target_error_voxels'],
                'status':generation['status'],'failures':[k for k,v in targets['family_pass'].items() if not v],
                'generation_seconds':generation['wall_seconds'],'evaluation_seconds':evaluation,'resources':resources,
                'field_changed':record['occupied_zyx']!=previous['occupied_zyx']}
            rows.append(row);key=(context_key,fraction)
            valid_groups.setdefault(key,[]);before_groups.setdefault(key,[])
            if targets['contract_pass']:valid_groups[key].append(set(map(tuple,record['occupied_zyx'])))
            if before['contract_pass']:before_groups[key].append(set(map(tuple,previous['occupied_zyx'])))
            store.event(run,'case_completed','Paired candidate retained',case=case,passed=row['pass'])
            print(f"{len(rows)}/{len(tasks)} {case}: {'pass' if row['pass'] else ','.join(row['failures'])}; {generation['wall_seconds']:.3f}s",flush=True)
            if resources['sampled_peak_rss_bytes']>recipe['rss_stop_bytes']:raise MemoryError('Sampled RSS cap exceeded after retained case')
            del record
        check()
    except Exception:
        status='interrupted';failure=traceback.format_exc();save('failure.json',{'traceback':failure},'interruption')
    non=[x for x in rows if not x['partition_control']];blocked=[x for x in rows if x['partition_control']]
    gate=status=='completed' and len(rows)==len(tasks) and all(x['pass'] and x['request_met'] and x['same_route'] and x['status']!='time_limit' for x in non) and all(not x['pass'] and x['same_route'] and x['status']!='time_limit' for x in blocked)
    metrics={'planned':len(tasks),'executed':len(rows),'nonpartition':len(non),'nonpartition_pass':sum(x['pass'] for x in non),
        'partition':len(blocked),'partition_pass':sum(x['pass'] for x in blocked),'admission_gate':gate,
        'regressions':[x['case'] for x in rows if x['previous_pass'] and not x['pass']],
        'repairs':[x['case'] for x in rows if not x['previous_pass'] and x['pass']],
        'route_matches':sum(x['same_route'] for x in rows),'changed_fields':sum(x['field_changed'] for x in rows),
        'timeouts':sum(x['status']=='time_limit' for x in rows),'study_seconds':time.perf_counter()-started}
    groups=[{'context_key':key[0],'request_fraction':key[1],'current':diversity(items),'previous':diversity(before_groups[key])} for key,items in valid_groups.items()]
    save('study.json',{'version':'MG5_v1','run_id':run,'mode':args.mode,'recipe':recipe,'status':status,'metrics':metrics,'cases':rows,
        'contexts':contexts,'diversity':groups,'failure':failure,'pending':[t for t in tasks if t['source']+'__'+t['case'] not in {x['case'] for x in rows}]},'study_summary')
    store.finish(run,status,metrics,'Paired development comparison; independent MT1, no trained-model or held-out claim.')
    write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':status,'mode':args.mode,'metrics':metrics,'artifact_location':'.local-artifacts/runs/'+run,'drive_backup':'pending'})
    print(json.dumps(metrics,indent=2));return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
