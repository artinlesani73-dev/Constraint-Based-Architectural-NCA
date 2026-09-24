"""MG4 frozen new-site evaluation. No generator changes or outcome-based retuning."""
from pathlib import Path
from hashlib import sha256
from itertools import combinations
from threading import Event, Thread
import argparse
import ctypes
import json
import os
import sys
import time
import traceback
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,write_once,provenance,snapshot_source,digest
from nca.contract import validate_scene
from nca.mass_generation_cases import generation_scenes
from nca.massing_cases import target_context
from nca.budget_mass_generator import generate_budget_mass,BudgetGeneratorSpec
from nca.massing_targets import evaluate_targets,MassingTargetSpec
from deploy.checkpoints import load_model_c


def geometry_key(scene):
    return json.dumps({k:scene[k] for k in ('grid_size','voxel_size_m','street_levels','ceiling_z','buildings','entrances')},sort_keys=True)


def load_design(recipe):
    path=REPO/recipe['scene_file']
    if digest(path)!=recipe['scene_sha256']:raise ValueError('Frozen scene bytes changed')
    for name,expected in recipe['source_sha256'].items():
        if digest(REPO/name)!=expected:raise ValueError('Frozen core source changed: '+name)
    sites=json.loads(path.read_bytes())['sites']
    old={geometry_key(s) for _,s in generation_scenes()};seen=set()
    for site in sites:
        scene=validate_scene(site['scene']);key=geometry_key(scene)
        if key in old or key in seen:raise ValueError('Scene duplicates a previous geometry')
        seen.add(key)
        if scene['grid_size']!=32 or scene['voxel_size_m']!=.8:raise ValueError('Unexpected physical scale')
    if len(sites)!=20 or sum(s['partition_control'] for s in sites)!=4:raise ValueError('Frozen population changed')
    if len(sites)*len(recipe['seeds'])*len(recipe['volume_requests'])!=recipe['expected_cases']:raise ValueError('Case count mismatch')
    return sites


def memory_bytes():
    """Actual resident working set and process lifetime high water; no tensor estimate."""
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise ctypes.WinError(ctypes.get_last_error())
        return {'rss_bytes':int(counters.WorkingSetSize),'lifetime_peak_rss_bytes':int(counters.PeakWorkingSetSize)}
    if sys.platform.startswith('linux'):
        import resource
        return {'rss_bytes':int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                'lifetime_peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024}
    raise RuntimeError('Resident memory measurement unsupported on this host')


class MemorySampler:
    def __init__(self,interval=.01):self.interval=interval;self.stop=Event();self.error=None
    def sample(self):
        value=memory_bytes();self.peak=max(self.peak,value['rss_bytes']);self.count+=1;return value
    def loop(self):
        while not self.stop.wait(self.interval):
            try:self.sample()
            except Exception as e:self.error=e;return
    def __enter__(self):
        self.peak=0;self.count=0;self.initial=self.sample();self.thread=Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.final=self.sample()
        if self.error:raise self.error
    def report(self):
        return {'initial_rss_bytes':self.initial['rss_bytes'],'final_rss_bytes':self.final['rss_bytes'],
                'sampled_peak_rss_bytes':self.peak,'sampled_increase_bytes':max(0,self.peak-self.initial['rss_bytes']),
                'lifetime_peak_rss_bytes':self.final['lifetime_peak_rss_bytes'],'samples':self.count,
                'sample_interval_seconds':self.interval,'scope':'Entire process resident set; sampling may miss short peaks; lifetime peak is cumulative.'}


def diversity(items):
    pairs=[]
    for a,b in combinations(items,2):
        union=a|b;pairs.append(1-len(a&b)/len(union) if union else 0.0)
    return {'valid_count':len(items),'unique_fields':len({frozenset(s) for s in items}),
            'mean_jaccard_distance':float(np.mean(pairs)) if pairs else None,'pairs':len(pairs)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/MG4-sites.json').read_bytes());sites=load_design(recipe)
    torch.set_num_threads(recipe['threads']);config,_,checkpoint=load_model_c(device='cpu')
    meta={**provenance(REPO),'effective_config':config,'checkpoint_weights_used':False,'checkpoint_sha256':digest(checkpoint)}
    store=RunStore(REPO/'.local-artifacts/runs');run=store.create('MG4 new-site stress evaluation','procedural_massing',recipe,3,meta,args.parent_run or recipe['parent_run'])
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter();rows=[];contexts=[];groups=[]
    tasks=[f"{s['case']}__v{round(v*100)}__s{seed}" for s in sites for v in recipe['volume_requests'] for seed in recipe['seeds']]
    status='completed';failure=None
    def save(name,data,role):write_once(d/name,data);store.attach(run,d/name,role)
    def limits():
        if time.perf_counter()-started>recipe['study_seconds_cap']:raise TimeoutError('Study time cap at case boundary')
        if memory_bytes()['rss_bytes']>recipe['rss_stop_bytes']:raise MemoryError('Resident set stopping threshold at case boundary')
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','exact_source')
        for name,role in [('docs/next-phase/SITE_GENERALIZATION_PROTOCOL.md','frozen_protocol'),('experiments/configs/MG4-sites.json','frozen_config'),(recipe['scene_file'],'frozen_scenes')]:store.attach(run,REPO/name,role)
        for site in sites:
            limits();t=time.perf_counter();fields,domain,region=target_context(site['scene'],config,recipe['padding_m_zyx'])
            context={**site,'domain':region,'domain_zyx':np.argwhere(domain).tolist(),
                     'masks':{k:np.argwhere(fields[k]).tolist() for k in ('permitted','protected','existing','support_boundary')},
                     'setup_seconds':time.perf_counter()-t}
            context_file='contexts/'+site['case']+'.json';save(context_file,context,'new_site_context')
            contexts.append({k:context[k] for k in ('case','group','partition_control','setup_seconds')}|{'path':context_file,'sha256':digest(d/context_file)})
            for fraction in recipe['volume_requests']:
                valid_fields=[]
                for seed in recipe['seeds']:
                    limits();case=f"{site['case']}__v{round(fraction*100)}__s{seed}";error=None
                    base={'case':case,'scene_case':site['case'],'group':site['group'],'partition_control':site['partition_control'],
                          'seed':seed,'request_fraction':fraction,'context_path':context_file,'context_sha256':digest(d/context_file)}
                    cpu=time.process_time();wall=time.perf_counter()
                    try:
                        with MemorySampler(recipe['rss_sample_seconds']) as sample:
                            field,route,generation=generate_budget_mass(site['scene'],fields,domain,seed,
                                BudgetGeneratorSpec(recipe['cube_m'],fraction,recipe['candidate_seconds_cap'],recipe['contact_weight']))
                            t=time.perf_counter();targets,masks=evaluate_targets(field,site['scene'],fields,domain,MassingTargetSpec());evaluation=time.perf_counter()-t
                        resources={'generation_evaluation_wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu,**sample.report()}
                        record={**base,'execution':'completed','occupied_zyx':np.argwhere(field).tolist(),'route_zyx':np.argwhere(route).tolist(),
                                'field_sha256':sha256(field.tobytes()).hexdigest(),'bulk_zyx':np.argwhere(masks['bulk']).tolist(),
                                'generation':generation,'targets':targets,'evaluation_seconds':evaluation,'resources':resources}
                        row={**base,'execution':'completed','contract_pass':targets['contract_pass'],
                             'family_failures':[k for k,v in targets['family_pass'].items() if not v],
                             'context_necessary_checks_pass':targets['context_necessary_checks_pass'],
                             'status':generation['status'],'requested_voxels':generation['requested_voxels'],
                             'occupied_voxels':generation['occupied_voxels'],'request_error_voxels':generation['target_error_voxels'],
                             'request_met':0<=generation['target_error_voxels']<generation['cube_width_cells']**3,
                             'facade_contact_fraction':targets['facade_contact_fraction'],'bulk_fraction':targets['bulk_fraction'],
                             'generation_seconds':generation['wall_seconds'],'evaluation_seconds':evaluation,'resources':resources}
                        if targets['contract_pass']:valid_fields.append(set(map(tuple,record['occupied_zyx'])))
                    except Exception:
                        error=traceback.format_exc();record={**base,'execution':'error','error':error}
                        row={**base,'execution':'error','contract_pass':False,'request_met':False,'status':'execution_error'}
                    path='cases/'+case+'.json';save(path,record,'candidate_result' if error is None else 'candidate_error')
                    rows.append({**row,'path':path,'sha256':digest(d/path)})
                    store.event(run,'case_completed','Candidate retained',case=case,execution=record['execution'])
                    del record
                groups.append({'scene_case':site['case'],'request_fraction':fraction,**diversity(valid_fields)})
            print(f"{site['case']}: {sum(x['contract_pass'] for x in rows if x['scene_case']==site['case'])}/9 pass",flush=True)
        limits()
    except Exception:
        status='interrupted';failure=traceback.format_exc();save('study-failure.json',{'traceback':failure},'study_interruption')
    nonpartition=[x for x in rows if not x['partition_control']];blocked=[x for x in rows if x['partition_control']]
    errors=sum(x['execution']!='completed' for x in rows);timeouts=sum(x['status']=='time_limit' for x in rows)
    gate=status=='completed' and len(rows)==180 and not errors and not timeouts and all(x['contract_pass'] and x['request_met'] for x in nonpartition) and all(not x['contract_pass'] for x in blocked)
    metrics={'planned':180,'executed':len(rows),'nonpartition_count':len(nonpartition),'nonpartition_pass':sum(x['contract_pass'] for x in nonpartition),
             'nonpartition_request_met':sum(x['request_met'] for x in nonpartition),'partition_count':len(blocked),
             'partition_pass':sum(x['contract_pass'] for x in blocked),'all_pass':sum(x['contract_pass'] for x in rows),
             'execution_errors':errors,'timeouts':timeouts,'scale_admission_gate':gate,'study_seconds':time.perf_counter()-started}
    study={'version':'MG4_v1','run_id':run,'status':status,'metrics':metrics,'contexts':contexts,'cases':rows,'diversity':groups,
           'pending':[x for x in tasks if x not in {r['case'] for r in rows}],'failure':failure}
    save('study.json',study,'study_summary')
    store.finish(run,'failed' if errors else status,metrics,'Designed previously untested sites; no population generalization or learned-model claim. Full failures retained.')
    write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':'failed' if errors else status,'metrics':metrics,'config':recipe,
        'artifact_location':'.local-artifacts/runs/'+run,'drive_backup':'pending'})
    print(json.dumps(metrics,indent=2),flush=True)
    return 0 if status=='completed' and not errors else 1


if __name__=='__main__':raise SystemExit(main())
