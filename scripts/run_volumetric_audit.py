"""VA1: archived equal-context material/void probes, no training or model promotion."""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,time,traceback
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from nca.experiments import RunStore,provenance,snapshot_source,write_once
from nca.contract import scene_hash,declared_existing
from nca.volumetric import measure_volume,fixed_probes
from nca.spatial import construct_platform,legacy_diagnostics
from scripts.run_spatial_prototype import cases
from deploy.studio import plan_scene
from deploy.checkpoints import load_model_c


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/VA1-volumetric.json').read_text())
    config,_,checkpoint=load_model_c(device='cpu')
    torch.set_num_threads(2);torch.manual_seed(recipe['random_seed']);np.random.seed(recipe['random_seed'])
    metadata={**provenance(REPO),'effective_config':config,'checkpoint_weights_used':False,
        'checkpoint_config_source':checkpoint.relative_to(REPO).as_posix(),
        'checkpoint_config_source_sha256':sha256(checkpoint.read_bytes()).hexdigest()}
    store=RunStore(REPO/'.local-artifacts/runs')
    run=store.create('VA1 volumetric material and void audit','geometry_objective_audit',recipe,recipe['random_seed'],metadata,args.parent_run)
    d=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter()
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','source_snapshot')
        scene=dict(cases())['aligned'];existing=declared_existing(scene)
        fields=fixed_probes();fields['sp1_platform']=construct_platform(scene)[0]
        w1=plan_scene(scene,config);fields['w1_scaffold']=np.zeros_like(existing)
        for z,y,x in w1['material_zyx']:fields['w1_scaffold'][z,y,x]=True
        region=np.zeros_like(existing);region[tuple(slice(*v) for v in recipe['analysis_region_zyx'])]=True
        write_once(d/'scene.json',scene);store.attach(run,d/'scene.json','shared_scene')
        results=[]
        for name in recipe['cases']:
            field=fields[name];volume,masks=measure_volume(field,existing,region,scene['voxel_size_m'])
            old=legacy_diagnostics(scene,field,config)
            record={'case':name,'scene_hash':scene_hash(scene),'material_zyx':np.argwhere(field).tolist(),
                'void_zyx':np.argwhere(masks['bracketed']).tolist(),
                'sealed_form_zyx':np.argwhere(masks['sealed_form']).tolist(),
                'sealed_context_zyx':np.argwhere(masks['sealed_context']).tolist(),
                'volume':volume,'legacy':old,'learned':False}
            write_once(d/(name+'.json'),record);store.attach(run,d/(name+'.json'),'individual_field_and_diagnostics')
            results.append(record)
            print(f"{name}: material={volume['material_voxels']}, bracketed={volume['bracketed_2_axes_voxels']}, sealed-form={volume['sealed_by_form_voxels']}, old-budget={old['in_budget']}",flush=True)
        checks={c['case']:c['volume']['material_voxels']==recipe['expected_material_counts'].get(c['case'],c['volume']['material_voxels']) for c in results}
        by={c['case']:c for c in results}
        checks.update({'closed_cavity_analytic':by['closed_shell_610']['volume']['sealed_by_form_voxels']==686,
            'open_ends_unsealed_by_form':by['open_ends_512']['volume']['sealed_by_form_voxels']==0,
            'aperture_unsealed_with_context':by['side_aperture_487']['volume']['sealed_with_context_voxels']==0,
            'solid_no_internal_brackets':by['solid_512']['volume']['bracketed_2_axes_voxels']==0,
            'decoy_no_internal_brackets':by['extent_decoy_8']['volume']['bracketed_2_axes_voxels']==0,
            'same_old_envelope':len({c['legacy']['envelope_voxels'] for c in results})==1,
            'w1_parity':all(by['w1_scaffold']['legacy'][k]==w1['diagnostics'][k] for k in ('families','material_ratio','connectivity','joint_budget_connectivity'))})
        study={'version':recipe['version'],'run_id':run,'training':False,'scene':scene,'recipe':recipe,
            'cases':results,'checks':checks,'note':'Fixed analytical probes. Descriptive voids, not a quality score or generated architectural solution.'}
        write_once(d/'study.json',study);store.attach(run,d/'study.json','gallery_data')
        metrics={'cases':len(results),'checks':checks,'all_checks_passed':all(checks.values()),'wall_seconds':time.perf_counter()-started}
        status='completed' if all(checks.values()) else 'failed'
        store.finish(run,status,metrics,study['note'])
        write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':status,'config':recipe,'metrics':metrics,
            'provenance':metadata,'artifact_location':d.relative_to(REPO).as_posix(),'drive_backup':'pending'})
        print(json.dumps({'run_id':run,'status':status,'metrics':metrics},indent=2));return 0 if status=='completed' else 1
    except Exception:
        message=traceback.format_exc();store.event(run,'error',message)
        if not (d/'result.json').exists():store.finish(run,'failed',{},message)
        raise


if __name__=='__main__':raise SystemExit(main())
