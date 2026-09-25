"""MG7 gated exact-choice comparison and balanced paired performance study."""
from pathlib import Path
from unittest.mock import patch
import argparse,json,sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,write_once,provenance,snapshot_source,digest
from nca.scale_study import decode_grid
from nca.massing_targets import evaluate_targets
from nca import coverage_mass_generator as reference
from nca import incremental_mass_generator as optimized
from run_site_generalization import MemorySampler,memory_bytes


class TimedSampler(MemorySampler):
    def __init__(self,interval=.01):super().__init__(interval);self.timeline=[]
    def sample(self):
        wall=time.perf_counter();cpu=time.process_time();data=super().sample()
        self.timeline.append([wall,cpu,data['rss_bytes']]);return data
    def report(self):
        result=super().report();samples=np.asarray(self.timeline)
        result['max_sample_gap_seconds']=float(np.diff(samples[:,0]).max()) if len(samples)>1 else 0.
        result['sampled_wall_span_seconds']=float(samples[-1,0]-samples[0,0])
        return result


def load_case(store,recipe,source,case):
    root=store.path(recipe['baselines'][source]);study=json.loads((root/'study.json').read_bytes())
    row=next(x for x in study['cases'] if x['case']==case)
    if source=='mg5':
        previous=json.loads((root/row['path']).read_bytes());context=json.loads((root/study['contexts'][row['context_key']]['path']).read_bytes())
        scene=context['scene'];n=scene['grid_size'];domain=decode_grid(context['domain_zyx'],n)
        fields={k:decode_grid(v,n) for k,v in context['masks'].items()}
        field=decode_grid(previous['occupied_zyx'],n);route=decode_grid(previous['route_zyx'],n)
    else:
        previous=json.loads((root/row['json']).read_bytes());entry=study['contexts'][row['scene_case']]
        scene=json.loads((root/entry['json']).read_bytes())['scene']
        with np.load(root/entry['arrays'],allow_pickle=False) as a:domain=a['domain'];fields={k:a[k] for k in a.files if k!='domain'}
        with np.load(root/row['arrays'],allow_pickle=False) as a:field=a['field'];route=a['route']
    return scene,fields,domain,previous,field,route


def equivalent(gen,field,route,previous,old_field,old_route):
    old=previous['generation'];same_route=np.array_equal(route,old_route)
    if old['status']!='time_limit':
        clean=lambda x:{k:v for k,v in x.items() if k not in ('wall_seconds','version')}
        return {'mode':'full','equal':bool(same_route and np.array_equal(field,old_field) and clean(gen)==clean(old)),
                'trace_steps':len(old.get('growth',{}).get('trace',[]))}
    changing={'version','wall_seconds','status','occupied_voxels','target_error_voxels','target_reached','selected_origins_zyx','growth'}
    same_meta=all(gen.get(k)==v for k,v in old.items() if k not in changing)
    steps=old.get('growth',{}).get('trace',[]);new=gen.get('growth',{}).get('trace',[])
    origins=old['selected_origins_zyx'];prefix=gen['selected_origins_zyx'][:len(origins)]==origins
    rebuilt=old_route.copy();width=old['cube_width_cells']
    for step in new[:len(steps)]:
        if step['origin_zyx'] is not None:
            z,y,x=step['origin_zyx'];rebuilt[z:z+width,y:y+width,x:x+width]=True
    return {'mode':'recorded_prefix','equal':bool(same_route and same_meta and prefix and new[:len(steps)]==steps and np.array_equal(rebuilt,old_field)),
            'trace_steps':len(steps),'new_trace_steps':len(new),'full_reference_output_available':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('diagnostic','matrix','timing'),required=True)
    parser.add_argument('--admission-run');parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/MG7-incremental.json').read_bytes());store=RunStore(REPO/'.local-artifacts/runs')
    for path,expected in recipe['source_sha256'].items():
        if digest(REPO/path)!=expected:raise ValueError('Frozen source changed: '+path)
    for run in recipe['baselines'].values():
        if store.verify(run):raise ValueError('Baseline corrupt')
    if args.mode!='diagnostic':
        if not args.admission_run or store.verify(args.admission_run):raise ValueError('Verified admission required')
        admission=json.loads((store.path(args.admission_run)/'study.json').read_bytes())
        expected='diagnostic' if args.mode=='matrix' else 'matrix'
        if admission['mode']!=expected or not admission['metrics']['admission_gate'] or admission['recipe']!=recipe:raise ValueError('Admission mismatch')
    if args.mode=='diagnostic':tasks=[{**x,'method':'optimized','trial':0,'warmup':False} for x in recipe['diagnostic']]
    elif args.mode=='matrix':
        tasks=[{'source':source,'case':x['case'],'method':'optimized','trial':0,'warmup':False}
            for source,run in recipe['baselines'].items() for x in json.loads((store.path(run)/'study.json').read_bytes())['cases']]
    else:
        tasks=[{**recipe['timing_cases'][0],'method':m,'trial':-1,'warmup':True} for m in ('reference','optimized')]
        for trial in range(recipe['trials']):
            order=('reference','optimized') if trial%2==0 else ('optimized','reference')
            tasks.extend({**case,'method':m,'trial':trial,'warmup':False} for case in recipe['timing_cases'] for m in order)
    torch.set_num_threads(2)
    run=store.create('MG7 incremental '+args.mode,'procedural_massing',{'recipe':recipe,'mode':args.mode},0,provenance(REPO),args.parent_run or args.admission_run or recipe['baselines']['mg64'])
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter();rows=[];contexts={};status='completed';failure=None
    def save(name,obj,role):write_once(d/name,obj);store.attach(run,d/name,role)
    def arrays(name,**values):
        p=d/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:np.savez_compressed(f,**values)
        store.attach(run,p,'lossless_arrays')
    def check():
        if time.perf_counter()-started>recipe['study_caps'][args.mode]:raise TimeoutError('Study boundary time cap')
        if memory_bytes()['rss_bytes']>recipe['rss_stop_bytes']:raise MemoryError('RSS threshold')
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','exact_source')
        for path in ('docs/next-phase/INCREMENTAL_GROWTH_PROTOCOL.md','experiments/configs/MG7-incremental.json'):store.attach(run,REPO/path,'frozen_protocol_config')
        for i,task in enumerate(tasks):
            check();source,case=task['source'],task['case'];key=source+'__'+case
            scene,fields,domain,previous,old_field,old_route=load_case(store,recipe,source,case)
            old_score,_=evaluate_targets(old_field,scene,fields,domain)
            if old_score!=previous['targets']:raise ValueError('Baseline score mismatch')
            if key not in contexts:
                save('inputs/'+key+'.json',{'scene':scene,'previous':previous},'exact_input_record')
                arrays('inputs/'+key+'.npz',domain=domain,old_field=old_field,old_route=old_route,**fields)
                contexts[key]={'json':'inputs/'+key+'.json','arrays':'inputs/'+key+'.npz'}
            module=reference if task['method']=='reference' else optimized
            fn=module.generate_coverage_mass if task['method']=='reference' else module.generate_incremental_mass
            times={'growth_wall_seconds':0.,'growth_cpu_seconds':0.};grow=module.grow_coverage
            def timed(*a,**kw):
                t=time.perf_counter();cpu=time.process_time()
                try:return grow(*a,**kw)
                finally:times['growth_wall_seconds']+=time.perf_counter()-t;times['growth_cpu_seconds']+=time.process_time()-cpu
            with TimedSampler(recipe['rss_sample_seconds']) as sampler:
                t=time.perf_counter();cpu=time.process_time()
                with patch.object(module,'grow_coverage',timed):
                    field,route,gen=fn(scene,fields,domain,previous['seed'],module.CoverageGeneratorSpec(**previous['generation']['spec']))
                times.update(generation_wall_seconds=time.perf_counter()-t,generation_cpu_seconds=time.process_time()-cpu)
                t=time.perf_counter();cpu=time.process_time();target,masks=evaluate_targets(field,scene,fields,domain)
                times.update(evaluation_wall_seconds=time.perf_counter()-t,evaluation_cpu_seconds=time.process_time()-cpu)
            resources=sampler.report();eq=equivalent(gen,field,route,previous,old_field,old_route)
            partition=previous['partition_control'];quality=(not target['contract_pass'] and gen['status']=='no_cube_route') if partition else (target['contract_pass'] and 0<=gen['target_error_voxels']<27 and gen['status']!='time_limit')
            name=f'{i:03d}__'+key+'__'+task['method'];t=time.perf_counter()
            arrays('cases/'+name+'.npz',field=field,route=route,bulk=masks['bulk'],samples=np.asarray(sampler.timeline))
            save('cases/'+name+'.json',{'task':task,'input_key':key,'generation':gen,'targets':target,'timing':times,'resources':resources,'equivalence':eq},'candidate_record')
            row={'task':task,'input_key':key,'json':'cases/'+name+'.json','arrays':'cases/'+name+'.npz','equivalence':eq,
                'quality':bool(quality),'pass':target['contract_pass'],'partition_control':partition,'status':gen['status'],
                'request_error_voxels':gen['target_error_voxels'],'timing':times,'resources':resources,'save_seconds':time.perf_counter()-t}
            rows.append(row);store.event(run,'case_completed','Incremental comparison retained',case=name,equal=eq['equal'],quality=bool(quality))
            print(f"{i+1}/{len(tasks)} {key} {task['method']}: equal={eq['equal']} quality={quality} {times['generation_wall_seconds']:.3f}s",flush=True)
            if resources['sampled_peak_rss_bytes']>recipe['rss_stop_bytes']:raise MemoryError('Sampled RSS cap after saved case')
        check()
    except Exception:
        status='interrupted';failure=traceback.format_exc();save('failure.json',{'traceback':failure},'interruption')
    gate=status=='completed' and len(rows)==len(tasks) and all(x['equivalence']['equal'] and x['quality'] for x in rows)
    performance=[]
    if args.mode=='timing':
        measured=[x for x in rows if not x['task']['warmup']]
        for case in recipe['timing_cases']:
            items=[x for x in measured if x['task']['source']==case['source'] and x['task']['case']==case['case']]
            by={m:[x for x in items if x['task']['method']==m] for m in ('reference','optimized')}
            if any(len(v)!=recipe['trials'] for v in by.values()):gate=False;continue
            ratios={name:float(np.median([x['timing'][name] for x in by['optimized']])/np.median([x['timing'][name] for x in by['reference']])) for name in ('generation_wall_seconds','generation_cpu_seconds')}
            rss_ratio=max(x['resources']['sampled_peak_rss_bytes'] for x in by['optimized'])/max(x['resources']['sampled_peak_rss_bytes'] for x in by['reference'])
            clean=all(x['resources']['max_sample_gap_seconds']<=recipe['max_sample_gap_seconds'] for x in items)
            passes=all(v<=recipe['max_time_ratio'] for v in ratios.values()) and rss_ratio<=recipe['max_rss_ratio'] and clean
            performance.append({**case,'ratios':ratios,'peak_rss_ratio':rss_ratio,'sample_gaps_admitted':clean,'pass':passes})
            gate=gate and passes
        gate=gate and len(performance)==len(recipe['timing_cases'])
    metrics={'planned':len(tasks),'executed':len(rows),'equivalent':sum(x['equivalence']['equal'] for x in rows),
        'full_comparisons':sum(x['equivalence']['mode']=='full' for x in rows),'prefix_comparisons':sum(x['equivalence']['mode']=='recorded_prefix' for x in rows),
        'quality_pass':sum(x['quality'] for x in rows),'admission_gate':gate,'study_seconds':time.perf_counter()-started}
    save('study.json',{'version':'MG7_v1','run_id':run,'mode':args.mode,'recipe':recipe,'status':status,'metrics':metrics,
        'cases':rows,'contexts':contexts,'performance':performance,'failure':failure,'pending':tasks[len(rows):]},'study_summary')
    store.finish(run,status,metrics,'Exact full/prefix comparisons; paired timing only where completed reference exists; no trained model.')
    write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'mode':args.mode,'status':status,'metrics':metrics,'artifact_location':'.local-artifacts/runs/'+run,'drive_backup':'pending'})
    print(json.dumps(metrics,indent=2));return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
