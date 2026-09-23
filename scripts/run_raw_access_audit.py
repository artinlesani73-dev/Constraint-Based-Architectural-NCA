"""A3: frozen-field raw maximin semantics, actual gradients, bounded probes."""
import argparse
from pathlib import Path
import subprocess,sys,time,traceback
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_sensitivity import STORE,records,record,arrays
from scripts.growth_common import load_source,vector_summary
from scripts.diagnostic_inputs import load_inputs
from deploy.checkpoints import load_model_c
from nca.experiments import read_json,write_once,provenance,snapshot_source,digest
from nca.access import component_strength,component_connectivity
from nca.raw_access import raw_component_strength,raw_component_access
from nca.access_trace import traced_rollout
from nca.interventions import experimental_rollout
from nca.access_training import objective_pair
from nca.objective import weighted_total
from nca.losses import LossSpec
from nca.sensitivity import contexts

CONFIG='experiments/configs/A3-raw-access.json'


def registry(c):
    for key in ('F3_run','H1_run','W1_run','D1_run'):
        run=c[key];assert not STORE.verify(run)
        assert read_json(STORE.path(run)/'result.json')['status']=='completed'
    run=c['F3_run'];p=records(run,'protocol')[0];sources={}
    for name,meta in p['members'].items():
        rows=[r for r in records(run,'evaluation_record') if r['branch']==name and r['trace']['update']==64]
        sources[name]={'arm':'F3','branch':name,'scene':meta['scene'],'recipe':meta['recipe'],
            'source_run':run,'metadata':meta,'update':64,'checkpoint':rows[0]['checkpoint'],
            'anchors':{str(r['trace']['steps']):r for r in rows}}
    fields=[]
    def add(group,run,row,scene,kind):
        fields.append({'group':group,'source_run':run,'scene':scene,'fields':row['fields'],'raw_kind':kind})
    for r in records(c['H1_run'],'growth_case'):
        if r['source_id'].startswith('original-'):add('original',c['H1_run'],r,r['scene'],'network_preclamp')
        elif r['source_id'].startswith('F2-'):add('F2',c['H1_run'],r,r['scene'],'network_preclamp')
    for r in records(run,'horizon_evaluation'):add('F3',run,r,r['trace']['scene'],'network_preclamp')
    for r in records(c['W1_run'],'witness_record'):
        if r['route_feasible']:add('W1',c['W1_run'],r,r['scene_id'],'material_as_raw_control')
    for r in records(c['D1_run'],'direct_case'):add('D1',c['D1_run'],r['final'],r['scene'],'direct_parameter_raw')
    assert {g:sum(r['group']==g for r in fields) for g in ('original','F2','F3','W1','D1')}=={'original':36,'F2':72,'F3':72,'W1':17,'D1':34}
    return sources,p['proposal']['recipes'],fields


def selected_indices(p):
    if p['mode']=='study':return list(range(len(p['fields'])))
    counts={};result=[]
    for i,r in enumerate(p['fields']):
        g=r['group'];counts[g]=counts.get(g,0)+1
        if counts[g]<=p['config']['pilot_per_group']:result.append(i)
    return result


def field_score(source,ctx):
    path=STORE.path(source['source_run'])/source['fields']['path'];assert digest(path)==source['fields']['sha256']
    with np.load(path,allow_pickle=False) as f:
        material=torch.from_numpy(f['material'].copy()).float()
        raw=material.clone() if source['raw_kind']=='material_as_raw_control' else torch.from_numpy(f['raw'].copy()).float()
    assert torch.equal(material,raw.clamp(0,1)*ctx.permitted),'Projection mismatch'
    b,details=raw_component_strength(raw[0],ctx.permitted[0],ctx.endpoints[0])
    old,od=component_strength(material[0],ctx.permitted[0],ctx.endpoints[0])
    assert details['legal_route_exists']==od['legal_route_exists']
    if details['legal_route_exists']:assert torch.equal(b.clamp(0,1),old)
    binary=component_connectivity(material[0].numpy(),ctx.permitted[0].numpy(),ctx.endpoints[0])
    # Separate BFS on the raw threshold indicator checks the unchanged binary rule.
    rb=component_connectivity((raw[0].numpy()>.5).astype(float),ctx.permitted[0].numpy(),ctx.endpoints[0])
    assert rb==binary
    assert binary['all_connected']==(details['legal_route_exists'] and float(b)>.5)
    return {'raw_bottleneck':float(b),'projected_bottleneck':float(old),'loss_v2':float(1-old),
        'loss_v3':float(torch.relu(1-b)),'raw_details':details,'projected_details':od,'binary':binary,
        'clamp_identity':True,'binary_unchanged':True,
        'critical_route_changed':details['critical_zyx']!=od['critical_zyx']}


def replay(run,p):
    cfg,_,_=load_model_c();_,inputs=load_inputs(REPO)
    ctxs=contexts(inputs,cfg,sorted({s['scene'] for s in p['fields']}))
    with torch.no_grad():
        for i in selected_indices(p):
            s=p['fields'][i];ctx,_=ctxs[s['scene']]
            row={'index':i,'source':s,'scene_hash':inputs[s['scene']]['scene_hash'],'score':field_score(s,ctx)}
            record(run,f'r{i:03d}',row,'raw_replay')
    print('replayed '+str(len(selected_indices(p))),flush=True)


def gradient(run,p,sid,h):
    source=p['sources'][sid];model,weights,item,ctx,allow=load_source(source)
    seed=p['config']['firing_seed'];cfg=model.config
    out=traced_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',h,torch.Generator().manual_seed(seed))
    state,raw=out['state'],out['raw_material'];material=state[:,cfg['ch_structure']]
    anchor=source['anchors'][str(h)]
    with np.load(STORE.path(source['source_run'])/anchor['fields']['path'],allow_pickle=False) as f:
        assert np.array_equal(f['material'],material.detach().numpy()) and np.array_equal(f['raw'],raw.detach().numpy())
    _,v2,d2=objective_pair(state,raw,ctx,cfg,allow,LossSpec());a3,d3=raw_component_access(raw,ctx.permitted,ctx.endpoints)
    v3={**v2,'terms':{**v2['terms'],'access':a3}}
    recipe=p['recipes'][source['recipe']];fw,rw=recipe['family_weights'],recipe['regularizer_weights']
    terms={'access_v2':v2['terms']['access'][0],'access_v3':a3[0],
        'coverage':v2['terms']['coverage'][0],'sparsity':v2['terms']['sparsity'][0],
        'total_v2':weighted_total(v2,fw,rw),'total_v3':weighted_total(v3,fw,rw)}
    params=list(model.named_parameters());ps=[v for _,v in params];vectors={};extra={};traces={}
    history=out['trajectory'];raws=[t['raw'] for t in history];deltas=[t['delta'] for t in history]
    for name,term in terms.items():
        targets=ps+raws+deltas if name.startswith('access_') else ps+[raw]
        gs=torch.autograd.grad(term,targets,retain_graph=True,allow_unused=True)
        gs=[torch.zeros_like(t) if g is None else g.detach() for g,t in zip(gs,targets)]
        vectors[name]=torch.cat([g.flatten() for g in gs[:len(ps)]]).numpy()
        if name.startswith('access_'):
            rg=gs[len(ps):len(ps)+h];dg=gs[len(ps)+h:]
            extra['trace_raw_gradient_'+name]=torch.stack(rg).numpy()
            traces[name]={'raw_l2':[float(g.double().norm()) for g in rg],
                'delta_l2':[float(g.double().norm()) for g in dg]}
            extra['raw_gradient_'+name]=rg[-1].numpy()
        else:extra['raw_gradient_'+name]=gs[-1].numpy()
    assert all(np.isfinite(v).all() for v in [*vectors.values(),*extra.values()])
    norms,cosines=vector_summary(vectors)
    summaries={k:{'value':float(t.detach()),'parameter_l2':norms[k],
        'last_raw_l2':float(np.linalg.norm(extra['raw_gradient_'+k].astype(float)))} for k,t in terms.items()}
    # A bounded, reversible directional probe, not an optimizer/training step.
    # Ties/clamps can make observed secants differ from the selected branch derivative.
    probes=[];length=p['config']['parameter_probe_l2'];v=vectors['access_v3'];norm=norms['access_v3']
    if norm:
        direction=torch.from_numpy(v/norm)
        try:
            for sign in (-1,1):
                with torch.no_grad():
                    offset=0
                    for name,param in params:
                        size=param.numel();param.copy_(weights[name]+sign*length*direction[offset:offset+size].reshape(param.shape));offset+=size
                    q=experimental_rollout(model,item['seed'],item['scaffold'],'hard_preclamp',h,torch.Generator().manual_seed(seed))
                    _,vv,_=objective_pair(q['state'],q['raw_material'],ctx,cfg,allow,LossSpec())
                    aa,_=raw_component_access(q['raw_material'],ctx.permitted,ctx.endpoints)
                    extra['probe_raw_'+str(sign)]=q['raw_material'].numpy()
                    extra['probe_material_'+str(sign)]=q['state'][:,cfg['ch_structure']].numpy()
                    probes.append({'sign':sign,'requested_parameter_l2':length,
                        'actual_parameter_l2':float(torch.cat([(param-weights[name]).flatten() for name,param in params]).double().norm()),
                        'access_v3':float(aa[0]),'access_v2':float(vv['terms']['access'][0]),
                        'mass_ratio':float(vv['mass_ratio'][0]),
                        'total_v3':float(weighted_total({**vv,'terms':{**vv['terms'],'access':aa}},fw,rw))})
        finally:model.load_state_dict(weights)
    assert all(torch.equal(v,weights[n]) for n,v in model.state_dict().items())
    for label,details in [('v2',d2[0]),('v3',d3[0])]:
        cell=details['critical_zyx']
        if cell is not None:
            at=(0,*cell)
            traces[label+'_critical']={'zyx':cell,'raw':[float(t['raw'][at].detach()) for t in history],
                'fired':[bool(t['mask'][(0,0,*cell)]) for t in history]}
    weighted={k:fw[k.split('_')[0]]*vectors[k] for k in ('access_v2','access_v3','coverage','sparsity')}
    weighted['other_terms']=vectors['total_v3']-weighted['access_v3']
    wn,wc=vector_summary(weighted)
    name=f'g-{sid}-h{h}'
    field=arrays(run,name,{'material':material.detach().numpy(),'raw':raw.detach().numpy(),
        'trajectory_raw':torch.stack([t['raw'].detach() for t in history]).numpy(),
        'trajectory_fired':torch.stack([t['mask'].bool() for t in history]).numpy(),**vectors,**extra})
    record(run,name,{'source_id':sid,'scene':source['scene'],'steps':h,'firing_seed':seed,
        'fields':field,'norms':summaries,'cosines':cosines,'weighted_norms':wn,'weighted_cosines':wc,
        'projected_details':d2[0],'raw_details':d3[0],'traces':traces,'probes':probes,
        'parameter_layout':[{'name':n,'shape':list(v.shape),'elements':v.numel()} for n,v in params],
        'saved_forward_exact':True,'frozen_weights_unchanged':True,'optimizer_updates':0},'raw_gradient')
    print(name+' saved',flush=True)


def cost_gate(run):
    p=records(run,'protocol')[0];assert p['mode']=='pilot'
    rows=records(run,'process_record');assert len(rows)==3
    assert all(r['returncode']==0 and not r['elapsed_cap_exceeded'] for r in rows)
    replay_time=next(r['seconds'] for r in rows if r['kind']=='replay')
    g=max(r['seconds'] for r in rows if r['kind']=='gradient')
    est=p['config']['safety_factor']*(len(p['fields'])/len(selected_indices(p))*replay_time+8*g)
    return {'estimated_seconds':est,'admitted':est<=p['config']['study_total_cap_seconds'],
        'cap_seconds':p['config']['study_total_cap_seconds'],'pilot_replay_seconds':replay_time,'pilot_max_gradient_seconds':g}


def launch(run,label,command,cap,kind):
    log=STORE.path(run)/(label+'.log');tick=time.perf_counter();timed_out=False
    with log.open('x',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*command],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=cap)
        except subprocess.TimeoutExpired:process.kill();process.wait();code=process.returncode;timed_out=True
        except BaseException:process.kill();process.wait();raise
    STORE.attach(run,log,'worker_log');elapsed=time.perf_counter()-tick
    record(run,'process-'+label,{'kind':kind,'label':label,'seconds':elapsed,'cap_seconds':cap,
        'returncode':code,'timed_out':timed_out,'elapsed_cap_exceeded':elapsed>cap},'process_record')
    print(f'{label}: exit={code}, elapsed={elapsed:.2f}',flush=True)
    if timed_out or elapsed>cap:raise TimeoutError(label+' elapsed cap exceeded; evidence retained')
    if code:raise RuntimeError(label+' failed; inspect retained log')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['pilot','study'])
    ap.add_argument('--worker',choices=['replay','gradient']);ap.add_argument('--run-id');ap.add_argument('--source-id');ap.add_argument('--horizon',type=int)
    ap.add_argument('--pilot-run');ap.add_argument('--parent-run');a=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if a.worker:
        p=records(a.run_id,'protocol')[0];assert all(digest(REPO/n)==v for n,v in p['code_sha256'].items())
        if a.worker=='replay':replay(a.run_id,p)
        else:gradient(a.run_id,p,a.source_id,a.horizon)
        return 0
    if not a.mode:ap.error('--mode required')
    c=read_json(REPO/CONFIG);sources,recipes,fields=registry(c)
    code=list((REPO/'nca').glob('*.py'))+[REPO/n for n in (CONFIG,'scripts/run_raw_access_audit.py',
        'scripts/report_raw_access_audit.py','scripts/growth_common.py','scripts/run_sensitivity.py','scripts/diagnostic_inputs.py',
        'scripts/report_corridor_comparison.py','deploy/model_utils.py','deploy/checkpoints.py')]
    p={'protocol':c['protocol'],'mode':a.mode,'config':c,'sources':sources,'recipes':recipes,'fields':fields,
        'code_sha256':{f.relative_to(REPO).as_posix():digest(f) for f in code},'pilot_run':a.pilot_run,'optimizer_updates':0}
    admission=None
    if a.mode=='study':
        if not a.pilot_run:ap.error('--pilot-run required')
        from scripts.report_raw_access_audit import verify
        verify(a.pilot_run)
        admission=cost_gate(a.pilot_run);assert admission['admitted'],admission
        old=records(a.pilot_run,'protocol')[0]
        for key in ('config','sources','recipes','fields','code_sha256'):assert old[key]==p[key],key
    run=STORE.create('A3 '+a.mode,'raw_access_diagnostic',p,2,provenance(REPO),parent_run=a.parent_run)
    print('RUN_ID='+run,flush=True);d=STORE.path(run);tick=time.perf_counter();status,error='completed',None
    limit=c['pilot_total_cap_seconds'] if a.mode=='pilot' else c['study_total_cap_seconds']
    try:
        record(run,'protocol',p,'protocol');snapshot_source(REPO,d/'source.zip');STORE.attach(run,d/'source.zip','source_snapshot')
        launch(run,'replay',['--worker','replay','--run-id',run],min(c['replay_worker_cap_seconds'],limit-(time.perf_counter()-tick)),'replay')
        selected=[c['pilot_branch']] if a.mode=='pilot' else list(sources)
        for sid in selected:
            for h in c['gradient_horizons']:
                cap=min(c['gradient_worker_cap_seconds'],limit-(time.perf_counter()-tick))
                if cap<=0:raise TimeoutError('Overall cap exhausted')
                launch(run,f'g-{sid}-h{h}',['--worker','gradient','--run-id',run,'--source-id',sid,'--horizon',str(h)],cap,'gradient')
        assert len(records(run,'raw_replay'))==len(selected_indices(p))
        assert len(records(run,'raw_gradient'))==len(selected)*2
        if a.mode=='pilot':admission=cost_gate(run)
        if time.perf_counter()-tick>limit:raise TimeoutError('Overall elapsed cap exceeded')
    except (KeyboardInterrupt,TimeoutError):status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'mode':a.mode,'replay_cases':len(records(run,'raw_replay')),
        'gradient_cases':len(records(run,'raw_gradient')),'optimizer_updates':0,'seconds':time.perf_counter()-tick,
        'cap_seconds':limit,'admission':admission,'provenance':read_json(d/'run.json')['provenance']}
    record(run,'summary',summary,'summary')
    if error:STORE.event(run,'error','A3 stopped; all evidence retained',traceback=error)
    STORE.finish(run,status,summary,c['scope']);write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True);return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
