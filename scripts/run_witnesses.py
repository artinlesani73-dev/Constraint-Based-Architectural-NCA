"""W1_v1 procedural witness follow-up; fixed A1 contract, no training."""
import argparse,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,provenance,snapshot_source,write_once
from nca.losses import LossSpec,context_from_scenes,material_envelope,loss_terms
from nca.facade import endpoint_allowance,facade_term
from nca.constructive import build_witness
from nca.e0 import evaluate
from scripts.diagnostic_inputs import load_inputs
from scripts.run_target_audit import with_budget
from scripts.report_facade_comparison import load as verified_a1,RUN as A1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);verified_a1()
    c1,inputs=load_inputs(REPO);cfg=c1['effective_model_config'];spec=LossSpec()
    protocol={'protocol':'W1_v1','input_run':A1,'envelope_radius':6,'budget':'envelope','facade':'facade_endpoint_v1',
              'expected_scenes':18,'optimizer_updates':0,'scope':'Procedural bound-satisfaction baseline, not architecture quality'}
    store=RunStore(REPO/'.local-artifacts/runs');origin=provenance(REPO)
    run=store.create('W1 constructive witnesses','constructive_baseline',protocol,0,origin,parent_run=args.parent_run)
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter();rows=[];status,error='completed',None
    def record(name,data,role):
        p=d/(name+'.json');write_once(p,data);return store.attach(run,p,role)
    try:
        p=d/'source.zip';snapshot_source(REPO,p);store.attach(run,p,'source_snapshot');p.unlink();record('protocol',protocol,'protocol')
        for index,(sid,item) in enumerate(sorted(inputs.items())):
            ctx=context_from_scenes(item['seed'],cfg,[item['scene']],item['guide'],material_envelope(item['guide'],item['permitted'],6),item['feasible'])
            allowance,annotation=endpoint_allowance(item['scene'],item['permitted'])
            result=build_witness(ctx,allowance,spec);material=result.pop('material');p=material.float()
            terms=with_budget(loss_terms(p,ctx,spec),p,ctx,'envelope',spec);terms['facade']=facade_term(p,ctx,allowance,spec)
            values={k:float(v[0]) for k,v in terms.items()}
            state=item['seed'].clone();state[:,cfg['ch_structure']]=p;metrics=evaluate(state,cfg,item['scene'])
            path=d/f'f{index:02d}.npz'
            with path.open('xb') as stream:np.savez_compressed(stream,guide=item['guide'].numpy(),material=material.numpy(),allowance=allowance.numpy(),envelope=ctx.envelope.numpy())
            ref=store.attach(run,path,'fields');path.unlink()
            witness=(result['status']=='constructed' and all(abs(v)<=1e-7 for v in values.values()) and metrics['connectivity']['all_connected'] is True and metrics['legality']['illegal_voxels']==0)
            row={'scene_id':sid,'scene_hash':item['scene_hash'],'route_feasible':bool(item['feasible'][0]),'fields':ref,**result,
                 'terms':values,'metrics':metrics,'witness':witness,'guide_voxels':int(item['guide'].sum()),'material_voxels':int(material.sum()),
                 'volume_m3':int(material.sum())*item['scene']['voxel_size_m']**3,'annotation':annotation}
            rows.append(row);record(f'w{index:02d}',row,'witness_record');print(sid,result['status'],'witness='+str(witness),flush=True)
        assert len(rows)==18
    except KeyboardInterrupt:status,error='interrupted',traceback.format_exc()
    except Exception:status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'cases':len(rows),'witnesses':sum(r['witness'] for r in rows),
             'incompatible_scenes':[r['scene_id'] for r in rows if r['status']=='incompatible'],'optimizer_updates':0,
             'seconds':time.perf_counter()-started,'provenance':origin,'artifact_location':f'.local-artifacts/runs/{run}'}
    record('summary',summary,'summary')
    if error:store.event(run,'error','Witness attempt stopped; evidence retained',traceback=error)
    store.finish(run,status,{'cases':len(rows),'witnesses':summary['witnesses']},protocol['scope'])
    write_once(REPO/'experiments/records'/(run+'.json'),summary);print(summary,flush=True)
    return 0 if status=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
