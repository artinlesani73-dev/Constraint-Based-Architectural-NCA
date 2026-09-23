"""A1_v1 matched facade comparison; no optimizer or production change."""
import argparse,hashlib,time,traceback,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,provenance,snapshot_source,write_once,read_json,digest
from nca.losses import LossSpec,context_from_scenes,material_envelope,loss_terms
from nca.facade import VERSION,endpoint_allowance,facade_term,facade_bounds
from nca.target_audit import joint_bounds
from nca.e0 import evaluate
from scripts.report_target_audit import load as load_t1,RUN as T1
from scripts.run_target_audit import with_budget
from scripts.diagnostic_inputs import load_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    store=RunStore(REPO/'.local-artifacts/runs');old=load_t1();_,inputs=load_inputs(REPO)
    cfg=old['protocol'][0]['config'];spec=LossSpec();annroot=REPO/'experiments/annotations'/VERSION
    manifest=read_json(annroot/'manifest.json');annotations={};allowances={}
    assert len(manifest['entries'])==18
    for entry in manifest['entries']:
        sid=entry['scene_id'];path=annroot/(sid+'.json');assert entry['path']==path.name and digest(path)==entry['sha256']
        allowance,annotation=endpoint_allowance(inputs[sid]['scene'],inputs[sid]['permitted'])
        annotation['source_scene_hash']=inputs[sid]['scene_hash'];annotation['mask_sha256']=hashlib.sha256(allowance.numpy().tobytes()).hexdigest()
        assert annotation==read_json(path);annotations[sid]=annotation;allowances[sid]=allowance
    protocol={'protocol':'A1_v1','input_run':T1,'config':cfg,'spec':asdict(spec),'annotation_manifest_sha256':digest(annroot/'manifest.json'),
              'arms':['original','endpoint_allowance'],'target_arm_records':864,'control_arm_records':144,'bound_arm_records':144,'gradient_arm_records':72,
              'optimizer_updates':0,'device':'cpu','threads':2,'note':'Only facade contact accounting differs; no attachment guarantee or production switch.'}
    origin=provenance(REPO);run=store.create('A1 facade endpoint allowance','facade_comparison',protocol,0,origin,parent_run=args.parent_run)
    directory=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter()
    def record(name,value,role):
        p=directory/(name+'.json');write_once(p,value);return store.attach(run,p,role)
    def arrays(name,values):
        p=directory/(name+'.npz')
        with p.open('xb') as f:np.savez_compressed(f,**{k:v.detach().cpu().numpy() for k,v in values.items()})
        ref=store.attach(run,p,'fields');p.unlink();return ref
    rows=[];controls=[];bounds=[];gradients=[];status,error='completed',None
    try:
        p=directory/'source.zip';snapshot_source(REPO,p);store.attach(run,p,'source_snapshot');p.unlink()
        record('protocol',protocol,'protocol');record('annotations',annotations,'annotations')
        t1dir=store.path(T1)
        for index,(sid,item) in enumerate(sorted(inputs.items())):
            allowance=allowances[sid];contexts={radius:context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],material_envelope(item['guide'],item['permitted'],radius),item['feasible']) for radius in (3,6)}
            source=next(r for r in old['target_record'] if r['scene_id']==sid)
            with np.load(t1dir/source['input_fields']['path'],allow_pickle=False) as f:fields={k:torch.from_numpy(f[k].copy()) for k in f.files}
            ref=arrays(f'i{index:02d}',{**fields,'allowance':allowance})
            common={'scene_id':sid,'scene_hash':item['scene_hash'],'route_feasible':bool(item['feasible'][0]),'fields':ref}
            cache={}
            for previous in [r for r in old['target_record'] if r['scene_id']==sid]:
                radius=previous['envelope_radius'];contract=previous['budget'];name=previous['candidate'];ctx=contexts[radius];p=fields[name].float()
                key=(name,radius)
                if key not in cache:
                    with torch.no_grad():cache[key]=loss_terms(p,ctx,spec)
                terms=with_budget(cache[key],p,ctx,contract,spec);baseline={k:float(v[0]) for k,v in terms.items()}
                if baseline!=previous['terms']:raise ValueError('T1 baseline differs')
                for arm in ('original','endpoint_allowance'):
                    values=dict(baseline)
                    if arm=='endpoint_allowance':values['facade']=float(facade_term(p,ctx,allowance,spec)[0])
                    assert all(values[k]==baseline[k] for k in values if k!='facade')
                    row={**common,'arm':arm,'candidate':name,'envelope_radius':radius,'budget':contract,'terms':values,
                         'mass_voxels':previous['mass_voxels'],'volume_m3':previous['volume_m3'],'metrics':previous['metrics'],
                         'context_valid':previous['context_valid'],'all_terms_zero':bool(p.any()) and all(abs(v)<=1e-7 for v in values.values()),
                         'source_run':T1,'source_candidate':name}
                    rows.append(row);record(f't{len(rows):03d}',row,'target_record')
            for radius,ctx in contexts.items():
                for contract in ('site','envelope'):
                    for arm in ('original','endpoint_allowance'):
                        b=joint_bounds(ctx,contract,spec) if arm=='original' else facade_bounds(ctx,contract,allowance,spec)
                        row={**common,'arm':arm,'envelope_radius':radius,'budget':contract,**{k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in b.items()}}
                        bounds.append(row);record(f'b{len(bounds):03d}',row,'bound_record')
            ctx=contexts[6];blanket=ctx.facade&ctx.permitted
            candidates={'empty':torch.zeros_like(allowance),'allowance_only':allowance,'facade_blanket':blanket,'guide_and_blanket':item['guide']|blanket}
            for name,mask in candidates.items():
                p=mask.float();state=item['seed'].clone();state[:,cfg['ch_structure']]=p;metrics=evaluate(state,cfg,item['scene'])
                with torch.no_grad():terms=with_budget(loss_terms(p,ctx,spec),p,ctx,'envelope',spec)
                for arm in ('original','endpoint_allowance'):
                    values={k:float(v[0]) for k,v in terms.items()}
                    if arm=='endpoint_allowance':values['facade']=float(facade_term(p,ctx,allowance,spec)[0])
                    if name=='facade_blanket' and (blanket&~allowance).any() and values['facade']<=0:raise ValueError('Blanket control not penalized')
                    row={**common,'arm':arm,'control':name,'terms':values,'mass_voxels':int(mask.sum()),'metrics':metrics}
                    controls.append(row);record(f'c{len(controls):03d}',row,'control_record')
            for previous in [r for r in old['gradient_record'] if r['scene_id']==sid]:
                with np.load(t1dir/previous['fields']['path'],allow_pickle=False) as f:p=torch.from_numpy(f['occupancy'].copy()).requires_grad_()
                for arm in ('original','endpoint_allowance'):
                    allowed=allowance if arm=='endpoint_allowance' else torch.zeros_like(allowance)
                    value=facade_term(p,ctx,allowed,spec);grad,=torch.autograd.grad(value.sum(),p)
                    if not torch.isfinite(grad).all():raise ValueError('Nonfinite derivative')
                    fieldref=arrays(f'g{len(gradients):02d}',{'occupancy':p,'gradient':grad})
                    row={**common,'arm':arm,'budget':previous['budget'],'gradient_fields':fieldref,'value':float(value[0].detach()),
                         'full_l2':float(grad.double().norm()),'legal_l2':float((grad*ctx.permitted).double().norm())}
                    if arm=='original':
                        assert row['value']==previous['gradient_norms']['facade']['value']
                        assert row['full_l2']==previous['gradient_norms']['facade']['full_l2']
                    gradients.append(row);record(f'g{len(gradients):02d}',row,'gradient_record')
            print(f'SCENE {index+1}/18 {sid}: targets={len(rows)} controls={len(controls)}',flush=True)
        assert (len(rows),len(controls),len(bounds),len(gradients))==(864,144,144,72)
    except KeyboardInterrupt:status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'target_arm_records':len(rows),'control_arm_records':len(controls),
             'bound_arm_records':len(bounds),'gradient_arm_records':len(gradients),'seconds':time.perf_counter()-started,
             'optimizer_updates':0,'provenance':origin,'artifact_location':f'.local-artifacts/runs/{run}'}
    record('summary',summary,'summary')
    if error:store.event(run,'error','Stopped; preceding evidence preserved',traceback=error)
    store.finish(run,status,{k:summary[k] for k in ('target_arm_records','control_arm_records','bound_arm_records','gradient_arm_records')},protocol['note'])
    write_once(REPO/'experiments/records'/(run+'.json'),summary);print(summary,flush=True)
    return 0 if status=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
