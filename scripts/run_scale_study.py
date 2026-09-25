"""MG6 bounded 48-grid / conditional 64-grid physical-environment study."""
from pathlib import Path
from unittest.mock import patch
import argparse,json,sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,write_once,provenance,snapshot_source,digest
from nca.scale_study import scale_context,embedded_scene,embed_field,decode_grid
from nca.massing_targets import evaluate_targets
from nca import coverage_mass_generator as generator
from deploy.checkpoints import load_model_c
from run_site_generalization import MemorySampler,memory_bytes,diversity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,choices=(48,64),required=True)
    parser.add_argument('--admission-run');parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/MG6-scale.json').read_bytes())
    for name,expected in recipe['source_sha256'].items():
        if digest(REPO/name)!=expected:raise ValueError('Frozen source changed: '+name)
    if digest(REPO/recipe['scene_file'])!=recipe['scene_sha256']:raise ValueError('Frozen scenes changed')
    store=RunStore(REPO/'.local-artifacts/runs')
    if store.verify(recipe['baseline_run']):raise ValueError('Baseline corrupt')
    if args.size==64:
        if not args.admission_run or store.verify(args.admission_run):raise ValueError('48 admission evidence required')
        admission=json.loads((store.path(args.admission_run)/'study.json').read_bytes())
        if admission['size']!=48 or not admission['metrics']['admission_gate'] or admission['recipe']!=recipe:raise ValueError('48 gate/config mismatch')
    sites=[x for x in json.loads((REPO/recipe['scene_file']).read_bytes())['sites'] if x['scene']['grid_size']==args.size]
    tasks=[(site,seed) for site in sites for seed in recipe['seeds']]
    torch.set_num_threads(recipe['threads']);config,_,checkpoint=load_model_c(device='cpu')
    run=store.create('MG6 physical scale '+str(args.size),'procedural_massing',{'recipe':recipe,'size':args.size},recipe['seeds'][0],
        {**provenance(REPO),'checkpoint_sha256':digest(checkpoint),'checkpoint_weights_used':False},args.parent_run or args.admission_run or recipe['baseline_run'])
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter();rows=[];contexts={};groups={}
    status='completed';failure=None;embedding=None;limits=recipe['limits'][str(args.size)]
    def save(name,obj,role):write_once(d/name,obj);store.attach(run,d/name,role)
    def arrays(name,role,**values):
        p=d/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:np.savez_compressed(f,**values)
        store.attach(run,p,role)
    def check():
        if time.perf_counter()-started>limits['study_seconds']:raise TimeoutError('Study cap at boundary')
        if memory_bytes()['rss_bytes']>recipe['rss_stop_bytes']:raise MemoryError('RSS threshold')
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','exact_source')
        for p in ('docs/next-phase/SCALE_STUDY_PROTOCOL.md','experiments/configs/MG6-scale.json',recipe['scene_file']):store.attach(run,REPO/p,'frozen_protocol_config')
        # A saved baseline field is translated, not regenerated: RNG draw shape
        # changes at larger grids, so equal-seed generation is not equivariant.
        root=store.path(recipe['baseline_run']);base=json.loads((root/'study.json').read_bytes())
        row=next(x for x in base['cases'] if x['case']==recipe['embedding_case'])
        previous=json.loads((root/row['path']).read_bytes());oldcontext=json.loads((root/base['contexts'][row['context_key']]['path']).read_bytes())
        oldscene=oldcontext['scene'];oldfield=decode_grid(previous['occupied_zyx'],32);olddomain=decode_grid(oldcontext['domain_zyx'],32)
        scene=embedded_scene(oldscene,args.size);fields,domain,audit=scale_context(scene,config)
        field=embed_field(oldfield,args.size);score,bulk=evaluate_targets(field,scene,fields,domain)
        embedding={'size':args.size,'scene':scene,'baseline_case':row['case'],'audit':audit,
            'same_domain':np.array_equal(domain,embed_field(olddomain,args.size)),
            'same_scores':score==previous['targets'],'scores':score,'scope':'Translated saved field; no new generation, no larger physical domain.'}
        save('embedding.json',embedding,'embedding_audit');arrays('embedding.npz','embedding_masks',field=field,domain=domain,bulk=bulk['bulk'],**fields)
        if not embedding['same_domain'] or not embedding['same_scores']:raise ValueError('Physical embedding audit failed')
        for site,seed in tasks:
            check();key=site['case'];case=key+'__s'+str(seed)
            if key not in contexts:
                t=time.perf_counter();fields,domain,audit=scale_context(site['scene'],config);context_seconds=time.perf_counter()-t
                save('contexts/'+key+'.json',{'scene':site['scene'],'audit':audit,'seconds':context_seconds},'physical_context')
                arrays('contexts/'+key+'.npz','context_masks',domain=domain,**fields)
                contexts[key]={'json':'contexts/'+key+'.json','arrays':'contexts/'+key+'.npz','domain_voxels':int(domain.sum()),'domain_volume_m3':audit['domain']['domain_volume_m3']}
            with np.load(d/contexts[key]['arrays'],allow_pickle=False) as data:
                domain=data['domain'];fields={k:data[k] for k in data.files if k!='domain'}
            stage_times={'growth_seconds':0.0,'growth_calls':0};original=generator.grow_coverage
            def timed_growth(*a,**kw):
                t=time.perf_counter()
                try:return original(*a,**kw)
                finally:stage_times['growth_seconds']+=time.perf_counter()-t;stage_times['growth_calls']+=1
            with MemorySampler(recipe['rss_sample_seconds']) as sampler:
                cpu=time.process_time();t=time.perf_counter()
                with patch.object(generator,'grow_coverage',timed_growth):
                    field,route,gen=generator.generate_coverage_mass(site['scene'],fields,domain,seed,
                        generator.CoverageGeneratorSpec(recipe['cube_m'],recipe['request_fraction'],limits['candidate_seconds'],recipe['contact_weight']))
                gen_outer=time.perf_counter()-t;t=time.perf_counter()
                target,masks=evaluate_targets(field,site['scene'],fields,domain);evaluation=time.perf_counter()-t
                cpu=time.process_time()-cpu
            resources={**sampler.report(),'cpu_seconds':cpu}
            times={**stage_times,'generation_outer_seconds':gen_outer,'setup_routing_residual_seconds':gen['wall_seconds']-stage_times['growth_seconds'],'evaluation_seconds':evaluation,
                'scope':'One timed growth wrapper; residual includes setup, Dijkstra and radial initialization; not isolated route-search time.'}
            record={'case':case,'scene_case':key,'size':args.size,'seed':seed,'request_fraction':recipe['request_fraction'],
                'partition_control':site['partition_control'],'generation':gen,'targets':target,'timing':times,'resources':resources}
            t=time.perf_counter();arrays('cases/'+case+'.npz','candidate_arrays',field=field,route=route,bulk=masks['bulk'])
            save('cases/'+case+'.json',record,'candidate_record');serialization=time.perf_counter()-t
            row={'case':case,'scene_case':key,'json':'cases/'+case+'.json','arrays':'cases/'+case+'.npz','pass':target['contract_pass'],
                'partition_control':site['partition_control'],'status':gen['status'],'request_met':0<=gen['target_error_voxels']<27,
                'request_error_voxels':gen['target_error_voxels'],'failures':[k for k,v in target['family_pass'].items() if not v],
                'timing':times,'resources':resources,'serialization_and_attachment_seconds':serialization}
            rows.append(row);store.event(run,'case_completed','Scale candidate retained',case=case,passed=row['pass'])
            groups.setdefault(key,[])
            if row['pass']:groups[key].append(set(map(tuple,np.argwhere(field))))
            print(f"{len(rows)}/{len(tasks)} {case}: {row['pass']} {gen['status']} {gen['wall_seconds']:.3f}s",flush=True)
            if resources['sampled_peak_rss_bytes']>recipe['rss_stop_bytes']:raise MemoryError('Observed RSS cap after retained case')
        check()
    except Exception:
        status='interrupted';failure=traceback.format_exc();save('failure.json',{'traceback':failure},'interruption')
    non=[x for x in rows if not x['partition_control']];blocked=[x for x in rows if x['partition_control']]
    gate=status=='completed' and len(rows)==len(tasks) and bool(embedding and embedding['same_domain'] and embedding['same_scores']) and all(x['pass'] and x['request_met'] and x['status']!='time_limit' for x in non) and all(not x['pass'] and x['status']=='no_cube_route' for x in blocked)
    metrics={'planned':len(tasks),'executed':len(rows),'nonpartition_pass':sum(x['pass'] for x in non),'nonpartition':len(non),
        'partition_pass':sum(x['pass'] for x in blocked),'partition':len(blocked),'admission_gate':gate,
        'timeouts':sum(x['status']=='time_limit' for x in rows),'study_seconds':time.perf_counter()-started}
    save('study.json',{'version':'MG6_v1','run_id':run,'size':args.size,'recipe':recipe,'status':status,'metrics':metrics,'cases':rows,'contexts':contexts,
        'embedding':embedding,'diversity':{k:diversity(v) for k,v in groups.items()},'failure':failure,
        'pending':[site['case']+'__s'+str(seed) for site,seed in tasks if site['case']+'__s'+str(seed) not in {x['case'] for x in rows}]},'study_summary')
    store.finish(run,status,metrics,'Designed physical-scale development study; no training, finer-resolution or generalization claim.')
    write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':status,'size':args.size,'metrics':metrics,'artifact_location':'.local-artifacts/runs/'+run,'drive_backup':'pending'})
    print(json.dumps(metrics,indent=2));return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
