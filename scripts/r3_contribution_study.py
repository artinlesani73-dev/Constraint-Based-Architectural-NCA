"""Frozen development ablation; no training and no studio mutations."""
from pathlib import Path
import argparse,json,hashlib,shutil,sys,subprocess,time,traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.local-artifacts/r3-contribution-2026-10-08'
def save(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    import numpy as np
    base=ROOT/'.local-artifacts/r3-site-studio-2026-10-08'
    identity=json.loads((base/'identity.json').read_text())
    for name,digest in identity['files'].items():
        if sha(base/name)!=digest:raise RuntimeError('Studio identity mismatch: '+name)
    OUT.mkdir(exist_ok=False)
    for name in ['source','model']:shutil.copytree(base/name,OUT/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(base/'config.json',OUT/'config.json')
    shutil.copyfile(__file__,OUT/'frozen-study.py')
    ids=['scale-32','site-staggered-court-40','site-offset-obstacle-40','site-blocked-gap-40']
    scenes={s['scene_id']:s for s in json.loads((base/'scenes.json').read_text())}
    references=json.loads((base/'integration-v2/result.json').read_text())['runs']
    parity={}
    for row in references:
        fields={}
        for p in (base/'runs'/row['id']).glob('*.npz'):
            if p.stem in ['G10-64','G10-128','R3-64','R3-128']:
                a=np.load(p)['field'];fields[p.stem]=hashlib.sha256(a.astype(bool).tobytes()).hexdigest()
        parity[row['site']]={'run':row['id'],'fields_sha256_bool_bytes':fields}
    protocol=dict(version='r3-neutral-selection-v1',date='2026-10-08',sites=[scenes[i] for i in ids],seeds=[2102,2103],request=.24,steps=128,horizons=[64,128],route='original',methods=['G10','R3','neutral'],
        neutral_definition='Unchanged R3 planner and admission. Replace learned network output with constant occupancy logit +1 and seven zero hidden updates; q=sigmoid(1)>0.5. Equal scores tie-break by flattened origin index. Same per-case firing seed and RNG draws. Not MG7, not witness-only, not a score-ranking-only intervention.',
        measurements=['nine families at64/128','absolute requested-volume error','late growth fraction','paired method Jaccard at128','seed-pair Jaccard at128 per site/method','occupied bounding extent zyx','rollout wall seconds (single samples, no performance claim)'],
        gates={'absolute_volume_error_max':.04,'late_growth_max':.05,'all_nine_both_horizons':True},
        interpretation='Exposed development geometries; two firing seeds are correlated repeats, not independent sites. Blocked requests count as no output, separately from feasible-site validity. No parameter tuning, best-of-N selection, extra retries or automatic adoption.',
        decision_rule='If neutral matches R3 validity on all certified requests, neural selection is not demonstrated necessary for validity here. Occupancy differences alone establish influence, not architectural superiority. Need later broader comparisons and human form review before adoption/training.',
        limits={'worker_seconds':180,'total_seconds':900,'device':'cpu','threads':2},references=parity)
    save(OUT/'protocol.json',protocol)
    inputs={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'input-manifest.json',inputs)
    save(ROOT/'experiments/reports/R3-contribution-protocol-2026-10-08.json',dict(protocol=protocol,input_manifest_sha256=sha(OUT/'input-manifest.json'),artifact_root=str(OUT)))
    print('Frozen',OUT)

def worker(index):
    import numpy as np,torch,platform
    sys.dont_write_bytecode=True;sys.path.insert(0,str(OUT/'source'))
    from context_route import route_from_context
    from g11_reservation import witness,rollout
    from nca.generation_data import seed_inputs
    from nca.repair_training import perceive
    from nca.repair_benchmark import condition
    from nca.paced_generation import PacedNCA
    from nca.repair_portable import read_portable
    from nca.massing_cases import target_context
    from nca.massing_targets import evaluate_targets,neighbors
    from nca.contract import entrance_masks
    from nca.budget_reference import budget
    protocol=json.loads((OUT/'protocol.json').read_text());s=protocol['sites'][index]
    dest=OUT/f'case-{index}';dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    save(dest/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=2))
    cfg=json.loads((OUT/'config.json').read_text());cfg['grid_size']=s['grid_size'];save(dest/'config.json',cfg);save(dest/'scene.json',s)
    fields,domain,_=target_context(s,cfg);c=condition(s,fields,domain,protocol['request']);x=seed_inputs(c);_,C=budget(int(domain.sum()),protocol['request'],3)
    np.savez_compressed(dest/'context.npz',condition=c)
    route=route_from_context(c);W=None;cert={'certified':False,'reason':'No legal thick route'}
    if route is not None:
        W,plan=witness(route,x['allowed'],s,fields,domain,C);score,_=evaluate_targets(W,s,fields,domain)
        cert=dict(certified=bool(score['contract_pass'] and W.sum()<=C),score=score,plan=plan)
        np.savez_compressed(dest/'witness.npz',field=W,route=route)
    save(dest/'certificate.json',cert)
    model=PacedNCA();model.load_state_dict(read_portable(OUT/'model/checkpoint-0427.pt',json.loads((OUT/'model/identity.json').read_text()))['model']);model.eval()
    class Neutral:
        def first(self,inputs):return inputs.new_zeros((1,8,*inputs.shape[2:]))
        def last(self,hidden):
            output=hidden.clone();output[:,0]=1;return output
    # Guard the baseline semantics before any baseline rollout.
    stub=Neutral();test=stub.last(stub.first(torch.zeros(1,61,4,4,4)));assert (test[:,0]==1).all() and (test[:,1:]==0).all()
    occ=torch.from_numpy(x['occupancy'])[None,None];static=perceive(torch.from_numpy(c)[None]);allowed=torch.from_numpy(x['allowed'])[None,None]
    contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool);face=neighbors(fields['existing'].astype(bool));ends=entrance_masks(s)
    for e in s['entrances']:
        if e['kind']=='facade':contact &= ~(ends[e['id']]&face&x['allowed'])
    results=[]
    for seed in protocol['seeds']:
        for method in protocol['methods']:
            if method!='G10' and not cert['certified']:
                results.append(dict(seed=seed,method=method,status='no_certified_output',accepted=False));continue
            t=time.perf_counter()
            with torch.no_grad():
                if method=='G10':
                    value=model.rollout(occ,static,allowed,torch.Generator().manual_seed(seed),steps=128,capture=True)
                    births=value['births'].numpy()[:,0,0];states={step:(x['occupancy'].astype(bool)|births[:step].any(0)) for step in [64,128]}
                    payload={'births':births,'state128':value['state'].numpy(),'counts':value['admission_counts'].numpy()}
                else:
                    value=rollout(model if method=='R3' else stub,occ,static,allowed,W,contact,firing_seed=seed)
                    states={step:value['states'][step][0,0].astype(bool) for step in [64,128]}
                    payload={'births':value['births'],'admission_kind':value['provenance'],'state64':value['states'][64],'state128':value['states'][128]}
                    save(dest/f'{method}-{seed}-trace.json',value['trace'])
            elapsed=time.perf_counter()-t;assert all(np.isfinite(a).all() for a in payload.values())
            np.savez_compressed(dest/f'{method}-{seed}-trajectory.npz',**payload)
            scores={}
            for step,f in states.items():
                np.savez_compressed(dest/f'{method}-{seed}-{step}.npz',field=f)
                score,_=evaluate_targets(f,s,fields,domain);coords=np.argwhere(f)
                scores[str(step)]={'score':score,'volume_error':float(abs(f.sum()/domain.sum()-.24)),'extent_cells_zyx':(coords.max(0)-coords.min(0)+1).tolist()}
                if seed==2102 and method in ['R3','G10']:
                    expected=protocol['references'][s['scene_id']]['fields_sha256_bool_bytes'][f'{method}-{step}']
                    assert hashlib.sha256(f.tobytes()).hexdigest()==expected,('Parity failed',s['scene_id'],method,step)
            growth=float((states[128].sum()-states[64].sum())/states[64].sum())
            accepted=all(q['score']['contract_pass'] and q['volume_error']<=.04 for q in scores.values()) and growth<=.05
            results.append(dict(seed=seed,method=method,status='evaluated',accepted=bool(accepted),scores=scores,growth=growth,rollout_seconds=elapsed))
            save(dest/'results.json',results)
            print(s['scene_id'],seed,method,'accepted',accepted,flush=True)
    save(dest/'results.json',results)
    save(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'})

def run():
    for name,digest in json.loads((OUT/'input-manifest.json').read_text()).items():
        if sha(OUT/name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    if sha(Path(__file__))!=sha(OUT/'frozen-study.py'):raise RuntimeError('Study code changed after freeze')
    status=OUT/'controller.json'
    if status.exists():raise RuntimeError('Existing attempt: inspect controller; do not overwrite or silently retry')
    protocol=json.loads((OUT/'protocol.json').read_text());start=time.monotonic();records=[];save(status,{'state':'running','workers':records})
    for i in range(len(protocol['sites'])):
        remaining=protocol['limits']['total_seconds']-(time.monotonic()-start)
        if remaining<=0:break
        log=OUT/f'worker-{i}.log';t=time.monotonic()
        with log.open('w',encoding='utf-8') as stream:
            try:
                job=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--index',str(i)],stdout=stream,stderr=subprocess.STDOUT,timeout=min(180,remaining))
                state='completed' if job.returncode==0 else 'error'
            except subprocess.TimeoutExpired:state='timeout'
        records.append({'index':i,'state':state,'seconds':time.monotonic()-t});save(status,{'state':'running','workers':records})
        print(records[-1],flush=True)
        if state!='completed':break
    save(status,{'state':'completed' if len(records)==4 and all(v['state']=='completed' for v in records) else 'stopped','workers':records,'seconds':time.monotonic()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','worker']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:worker(a.index)
