"""A2 frozen-field semantics replay and actual parameter-gradient audit. No optimizer."""
import argparse
from collections import deque
from pathlib import Path
import subprocess,sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.access import component_access,component_connectivity,component_strength,VERSION
from nca.experiments import provenance,snapshot_source,read_json,write_once,digest
from nca.losses import LossSpec,soft_reach
from nca.sensitivity import contexts
from nca.interventions import experimental_rollout
from nca.objective import research_terms,weighted_total
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from scripts.diagnostic_inputs import load_inputs
from scripts.run_sensitivity import STORE,records,record,arrays

CONFIG='experiments/configs/A2-access.json'


def sources(config):
    result=[]
    for run,key in ((config['fitting_run'],'fitting'),(config['sensitivity_run'],'sensitivity')):
        assert not STORE.verify(run) and read_json(STORE.path(run)/'result.json')['status']=='completed'
        for row in records(run,'evaluation_record'):
            t=row['trace'] if key=='fitting' else row
            result.append({'source_run':run,'branch':row['branch'],'update':t.get('update'),
                'scene':t['scene'],'steps':t['steps'],'fields':row['fields'],
                'metrics':t['metrics'],'terms':t['terms'],'regularizers':t['regularizers'],
                'totals':t['totals_under_both_recipes'],'kind':key})
    run=config['direct_run'];assert not STORE.verify(run) and read_json(STORE.path(run)/'result.json')['status']=='completed'
    for row in records(run,'direct_case'):
        t=row['final'];result.append({'source_run':run,'branch':row['recipe'],'update':32,
            'scene':row['scene'],'steps':None,'fields':t['fields'],'metrics':t['metrics'],
            'terms':t['terms'],'regularizers':t['regularizers'],'totals':t['totals'],'kind':'direct'})
    assert len(result)==277
    return result


def fixed_distances(material,ctx):
    """Independent unrestricted BFS distances from the legacy fixed point."""
    occupied=material>.5;origin=tuple(np.argwhere(ctx.source[0].numpy())[0])
    distance=np.full(occupied.shape,-1,dtype=int);queue=deque()
    if occupied[origin]:distance[origin]=0;queue.append(origin)
    while queue:
        cell=queue.popleft()
        for axis in range(3):
            for delta in (-1,1):
                q=list(cell);q[axis]+=delta;q=tuple(q)
                if all(0<=q[i]<occupied.shape[i] for i in range(3)) and occupied[q] and distance[q]<0:
                    distance[q]=distance[cell]+1;queue.append(q)
    result={}
    for name,region in ctx.endpoints[0].items():
        if name==ctx.source_ids[0]:continue
        reached=distance[region.numpy()];reached=reached[reached>=0]
        result[name]=int(reached.min()) if len(reached) else None
    return result


def replay(run,protocol):
    cfg,_,_=load_model_c();_,inputs=load_inputs(REPO)
    ctxs=contexts(inputs,cfg,[n for n,i in inputs.items() if bool(i['feasible'][0])])
    with torch.no_grad():
        for index,source in enumerate(protocol['sources']):
            ctx,_=ctxs[source['scene']]
            with np.load(STORE.path(source['source_run'])/source['fields']['path'],allow_pickle=False) as f:
                p=torch.from_numpy(f['material'].copy())
            reached=soft_reach(p*ctx.permitted,ctx.source,64)
            scores=[reached[0][region].max() for name,region in sorted(ctx.endpoints[0].items()) if name!=ctx.source_ids[0]]
            old=float(1-torch.stack(scores).mean());assert old==source['terms']['access']
            loss,details=component_access(p,ctx.permitted,ctx.endpoints);new=float(loss[0])
            fixed_regions=dict(ctx.endpoints[0]);fixed_regions[ctx.source_ids[0]]=ctx.source[0]
            fixed_strength,_=component_strength(p[0],ctx.permitted[0],fixed_regions)
            binary=component_connectivity(p[0].numpy(),ctx.permitted[0].numpy(),ctx.endpoints[0])
            assert binary['all_connected']==(1-new>.5)
            distances=fixed_distances(p[0].numpy(),ctx)
            row={'index':index,'source':source,'scene_hash':inputs[source['scene']]['scene_hash'],
                'access_v1':old,'access_v2':new,'candidate':details[0],'binary_v2':binary,
                'fixed_worst_64':float(1-torch.stack(scores).min()),
                'fixed_worst_unbounded':float(1-fixed_strength),
                'fixed_source_material':float(p[ctx.source].item()),'fixed_target_distances':distances,
                'finite_hop_binary_miss':bool(distances) and all(v is not None for v in distances.values()) and max(distances.values())>64,
                'old_source_fragmented':source['metrics']['connectivity']['status']=='unscorable',
                'totals_v2':{r:t+15.*(new-old) for r,t in source['totals'].items()},
                'total_note':'Exact arithmetic delta in Python float; only access coefficient15 is replaced. Other terms unchanged.'}
            record(run,f'r{index:03d}',row,'access_replay')
            if (index+1)%25==0:print(f'replay {index+1}/277',flush=True)


def gradient(run,protocol,index):
    case=protocol['gradient_cases'][index];cfg,weights,_=load_model_c();_,inputs=load_inputs(REPO)
    if case['checkpoint']:
        path=STORE.path(protocol['config']['fitting_run'])/case['checkpoint']['path']
        assert digest(path)==case['checkpoint']['sha256']
        weights=torch.load(path,weights_only=True,map_location='cpu')['model']
    model=UrbanPavilionNCA(dict(cfg));model.load_state_dict(weights);model.train()
    item=inputs[case['scene']];ctx,allow=contexts(inputs,cfg,[case['scene']])[case['scene']]
    out=experimental_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',case['steps'],torch.Generator().manual_seed(2))
    state,raw=out['state'],out['raw_material'];p=state[:,cfg['ch_structure']]
    with np.load(STORE.path(case['source_run'])/case['fields']['path'],allow_pickle=False) as f:
        assert np.array_equal(raw.detach().numpy(),f['raw']) and np.array_equal(p.detach().numpy(),f['material'])
    values=research_terms(state,raw,ctx,cfg,allow,LossSpec())
    reached=soft_reach(p*ctx.permitted,ctx.source,64)
    old=1-torch.stack([reached[0][r].max() for n,r in sorted(ctx.endpoints[0].items()) if n!=ctx.source_ids[0]]).mean()
    new,details=component_access(p,ctx.permitted,ctx.endpoints);new=new[0]
    assert float(old.detach())==float(values['terms']['access'][0].detach())
    recipe=protocol['recipes'][case['recipe']]
    total=weighted_total(values,recipe['family_weights'],recipe['regularizer_weights'])
    candidate_values={**values,'terms':{**values['terms'],'access':new[None]}}
    terms={'access_v1':old,'access_v2':new,'coverage':values['terms']['coverage'][0],
        'sparsity':values['terms']['sparsity'][0],'total_v1':total,
        'total_v2':weighted_total(candidate_values,recipe['family_weights'],recipe['regularizer_weights'])}
    params=list(model.named_parameters());vectors={};norms={};raw_grads={}
    for name,term in terms.items():
        grads=torch.autograd.grad(term,[v for _,v in params],retain_graph=True,allow_unused=True)
        vector=torch.cat([(g if g is not None else torch.zeros_like(v)).detach().flatten() for g,(_,v) in zip(grads,params)])
        raw_grad,=torch.autograd.grad(term,raw,retain_graph=True,allow_unused=True)
        raw_grad=torch.zeros_like(raw) if raw_grad is None else raw_grad.detach()
        assert torch.isfinite(vector).all() and torch.isfinite(raw_grad).all()
        vectors[name]=vector.numpy();raw_grads['raw_gradient_'+name]=raw_grad.numpy()
        norms[name]={'value':float(term.detach()),'parameter_l2':float(vector.double().norm()),'last_raw_l2':float(raw_grad.double().norm())}
    cosine={}
    for a in terms:
        cosine[a]={}
        for b in terms:
            va,vb=vectors[a].astype(float),vectors[b].astype(float);den=np.linalg.norm(va)*np.linalg.norm(vb)
            cosine[a][b]=float(np.dot(va,vb)/den) if den else None
    assert all(torch.equal(v,weights[n]) for n,v in model.state_dict().items())
    field=arrays(run,f'g{index:02d}',{'material':p.detach().numpy(),'raw':raw.detach().numpy(),**vectors,**raw_grads})
    record(run,f'g{index:02d}',{'index':index,'case':case,'candidate':details[0],
        'scene_hash':item['scene_hash'],'norms':norms,'cosines':cosine,'fields':field,
        'parameter_layout':[{'name':n,'shape':list(v.shape),'elements':v.numel()} for n,v in params],
        'frozen_weights_unchanged':True,'saved_forward_exact':True},'access_gradient')
    print(f'gradient {index+1}/12 {case["label"]} h{case["steps"]}',flush=True)


def launch(run,label,command,cap):
    path=STORE.path(run)/(label+'.log');start=time.perf_counter();timed_out=False
    with path.open('x',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*command],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();code=process.returncode;timed_out=True
        except BaseException:process.kill();process.wait();raise
    STORE.attach(run,path,'worker_log')
    record(run,'process-'+label,{'seconds':time.perf_counter()-start,'exit_code':code,'timed_out':timed_out,'cap':cap},'process_record')
    print(f'{label}: exit={code}, seconds={time.perf_counter()-start:.2f}',flush=True)
    if timed_out:raise TimeoutError(label+' exceeded cap')
    if code:raise RuntimeError(label+' failed; inspect retained log')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=['replay','gradient'])
    parser.add_argument('--run-id');parser.add_argument('--index',type=int);parser.add_argument('--parent-run');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.worker:
        protocol=records(args.run_id,'protocol')[0]
        if args.worker=='replay':replay(args.run_id,protocol)
        else:gradient(args.run_id,protocol,args.index)
        return 0
    config=read_json(REPO/CONFIG);saved=sources(config)
    fitting=records(config['fitting_run'],'protocol')[0];cases=[]
    for row in records(config['sensitivity_run'],'evaluation_record'):
        if row['branch']=='original_checkpoint' and row['scene'] in config['gradient_scenes']:
            cases.append({'label':'original','scene':row['scene'],'steps':row['steps'],'recipe':'mapped_30',
                'checkpoint':None,'fields':row['fields'],'source_run':config['sensitivity_run']})
    for row in records(config['fitting_run'],'evaluation_record'):
        if row['trace']['update']==64:
            cases.append({'label':row['branch'],'scene':row['trace']['scene'],'steps':row['trace']['steps'],
                'recipe':fitting['members'][row['branch']]['recipe'],'checkpoint':row['checkpoint'],
                'fields':row['fields'],'source_run':config['fitting_run']})
    assert len(cases)==12
    code=list((REPO/'nca').glob('*.py'))+[REPO/p for p in (CONFIG,'scripts/run_access_audit.py','scripts/diagnostic_inputs.py','deploy/model_utils.py','deploy/checkpoints.py')]
    protocol={'protocol':'A2_v1','config':config,'sources':saved,'gradient_cases':cases,'recipes':fitting['proposal']['recipes'],
        'code_sha256':{p.relative_to(REPO).as_posix():digest(p) for p in code},
        'firing_seed':2,'optimizer_updates':0,'candidate':VERSION,
        'scope':'Frozen-field access-contract and actual-gradient diagnostic; no architecture/training or promotion'}
    origin=provenance(REPO);run=STORE.create('A2 access alignment','access_diagnostic',protocol,2,origin,parent_run=args.parent_run)
    print('RUN_ID='+run,flush=True);d=STORE.path(run);start=time.perf_counter();status,error='completed',None
    try:
        record(run,'protocol',protocol,'protocol');p=d/'source.zip';snapshot_source(REPO,p);STORE.attach(run,p,'source_snapshot')
        launch(run,'replay',['--worker','replay','--run-id',run],config['replay_cap_seconds'])
        for i in range(12):
            cap=min(config['gradient_case_cap_seconds'],config['total_cap_seconds']-(time.perf_counter()-start))
            if cap<=0:raise TimeoutError('Total audit cap exhausted')
            launch(run,f'g{i:02d}',['--worker','gradient','--run-id',run,'--index',str(i)],cap)
        assert len(records(run,'access_replay'))==277 and len(records(run,'access_gradient'))==12
    except (KeyboardInterrupt,TimeoutError):status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'replay_cases':len(records(run,'access_replay')),
        'gradient_cases':len(records(run,'access_gradient')),'optimizer_updates':0,'seconds':time.perf_counter()-start,
        'provenance':origin,'artifact_location':f'.local-artifacts/runs/{run}'}
    record(run,'summary',summary,'summary')
    if error:STORE.event(run,'error','Audit stopped; all partial evidence retained',traceback=error)
    STORE.finish(run,status,summary,protocol['scope']);write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True);return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
