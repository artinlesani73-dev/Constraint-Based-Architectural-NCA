"""NR3 local CPU scoring of three sealed final models; no training or cloud access."""
from pathlib import Path
import argparse,io,json,sys,time,traceback,zipfile
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from nca.experiments import read_json,write_once,digest,RunStore,snapshot_source
from nca.repair_quality import SEEDS,FIRING,HORIZONS,CHECKPOINTS,SETTINGS,primary_gate
from nca.repair_training import RepairNCA,perceive
from nca.repair_benchmark import load_example,repair_metrics
from nca.massing_targets import evaluate_targets


def read_models(archives,expected_manifest):
    """Validate all three final models before opening any heldout examples."""
    models={};sealed=[]
    for path in archives:
        path=Path(path);receipt=read_json(path.with_suffix('.receipt.json'))
        if digest(path)!=receipt['sha256']:raise ValueError('Returned archive hash differs')
        with zipfile.ZipFile(path) as z:
            m=json.loads(z.read('evidence-manifest.json'));names=z.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(m)|{'evidence-manifest.json'} or len(m)!=receipt['files']:raise ValueError('Archive members differ')
            if any(sha256(z.read(n)).hexdigest()!=h for n,h in m.items()):raise ValueError('Corrupt evidence')
            result=json.loads(z.read('result.json'));request=result['request'];seed=request['seed']
            if seed not in SEEDS or seed in models or result['status']!='completed' or request['cpu_rehearsal'] or request['updates']!=256:raise ValueError('Three distinct completed quality jobs required')
            if request['settings']!=SETTINGS or request['manifest_sha256']!=expected_manifest:raise ValueError('Frozen study/source identity differs')
            models[seed]={}
            for step in CHECKPOINTS:
                n=f'worker/checkpoint-{step:04d}.pt';raw=z.read(n);meta=json.loads(z.read(n[:-3]+'.json'))
                if sha256(raw).hexdigest()!=meta['sha256'] or len(raw)!=meta['bytes']:raise ValueError('Checkpoint bytes differ')
                p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                if p['completed']!=step or p['identity']['seed']!=seed or p['identity']['experiment']!={'study':SETTINGS,'manifest_sha256':expected_manifest}:raise ValueError('Checkpoint identity differs')
                if p['identity']['runtime']['device']!='cuda:0':raise ValueError('GPU-trained checkpoint required')
                models[seed][step]=p['model']
            sealed.append({'seed':seed,'archive':str(path.resolve()),'archive_sha256':digest(path),
                           'final_checkpoint_sha256':m['worker/checkpoint-0256.pt']})
    if set(models)!=set(SEEDS):raise ValueError('All three final seeds must be fixed before evaluation')
    return models,sealed


def predict(weights,inputs,steps,firing_seed):
    model=RepairNCA().float();model.load_state_dict(weights,strict=True);model.eval()
    occupancy=torch.from_numpy(inputs['occupancy'])[None,None]
    context=torch.from_numpy(inputs['context'])[None]
    allowed=(context[:,:1]>0)&(context[:,1:2]>0)
    g=torch.Generator(device='cpu').manual_seed(firing_seed)
    with torch.no_grad():
        state=model.rollout(occupancy,perceive(context),allowed,g,steps)
        probability=torch.sigmoid(state[:,:1])*allowed
    if not torch.isfinite(state).all():raise ValueError('Nonfinite evaluated state')
    return state.numpy()[0].copy(),probability.numpy()[0,0].copy()


def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    # Authoritative deployment-oriented inference remains on the verified local CPU stack.
    if str(torch.__version__)!='2.8.0+cpu' or np.__version__!='2.5.2':raise ValueError('Review a changed local evaluation stack before scoring')
    snapshot_source(ROOT,out/'source.zip')
    write_once(out/'request.json',{'mode':a.mode,'settings':SETTINGS,'manifest_sha256':a.manifest_sha256,'archives':a.archives,
                                 'torch':str(torch.__version__),'numpy':np.__version__,'seconds_between_examples_cap':a.seconds})
    rows=[];failure=None;status='completed';start=time.monotonic()
    try:
        models,sealed=read_models(a.archives,a.manifest_sha256);write_once(out/'sealed-finals.json',sealed)
        source=ROOT/'.local-artifacts/runs'/SETTINGS['dataset_run']
        if RunStore(source.parent).verify(source.name):raise ValueError('NL0 source evidence differs')
        study=read_json(source/'study.json');targets={x['case']:x for x in study['targets']}
        examples=[x for x in study['examples'] if x['split'] in (('validation',) if a.mode=='diagnostics' else ('validation','test'))]
        schedule=[(k,32,2101) for k in CHECKPOINTS] if a.mode=='diagnostics' else [(256,s,f) for s in HORIZONS for f in FIRING]
        for seed in SEEDS:
            for step,steps,firing in schedule:
                for index,row in enumerate(examples):
                    if time.monotonic()-start>=a.seconds:raise TimeoutError('Local evaluation cap reached between examples; retain partial results')
                    tick=time.monotonic();inputs,target=load_example(source,row,split=row['split']);loaded=time.monotonic()
                    raw,prob=predict(models[seed][step],inputs,steps,firing);inferred=time.monotonic();field=prob>.5
                    t=targets[row['case']];context=read_json(source/t['json'])
                    with np.load(source/t['arrays'],allow_pickle=False) as pack:
                        domain=pack['domain'];fields={k:pack[k] for k in ('permitted','existing','protected','support_boundary')}
                    report,_=evaluate_targets(field,context['scene'],fields,domain)
                    metrics=repair_metrics(field,target.astype(bool),inputs['occupancy'].astype(bool),domain,context['generation']['spec']['target_fraction'])
                    metrics['targets']=report
                    key=f'{seed}-{step}-{steps}-{firing}-{index}'
                    record={'seed':seed,'checkpoint':step,'steps':steps,'firing_seed':firing,'split':row['split'],'case':row['case'],
                            'damage':row['damage'],'site':row['site'],'metrics':metrics,
                            'seconds':{'loading':loaded-tick,'inference_including_model_load':inferred-loaded,'scoring':time.monotonic()-inferred}}
                    folder=out/'observations';folder.mkdir(exist_ok=True)
                    with (folder/(key+'.npz')).open('xb') as f:np.savez_compressed(f,state=raw,probability=prob,field=field)
                    record['arrays_sha256']=digest(folder/(key+'.npz'));write_once(folder/(key+'.json'),record);rows.append(record)
        if a.mode=='final':write_once(out/'primary-gate.json',primary_gate(rows,study['examples']))
    except BaseException:status='failed';failure=traceback.format_exc()
    result={'status':status,'failure':failure,'observations':len(rows),'wall_seconds':time.monotonic()-start,
            'note':'CPU-scored geometry; GPU training alone is not a quality claim. Failed/partial evaluation cannot pass the gate.'}
    write_once(out/'result.json',result);print(json.dumps(result,indent=2))
    return 0 if status=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archives',nargs=3,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['diagnostics','final'],required=True);p.add_argument('--seconds',type=float,default=10800)
    a=p.parse_args();raise SystemExit(main(a))
