"""Preserve SP1 procedural examples, negatives, old-metric audit and exact source."""
from pathlib import Path
import argparse,json,sys,time,traceback
from dataclasses import asdict
from copy import deepcopy
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from nca.experiments import RunStore,provenance,snapshot_source,write_once
from nca.contract import validate_scene,scene_hash
from nca.spatial import PlatformSpec,construct_platform,evaluate_platform,legacy_diagnostics
from deploy.checkpoints import load_model_c
from deploy.studio import plan_scene


def cases():
    base=json.loads((REPO/'experiments/scenes/reference_v1/ref-01-ground-pair.json').read_text())
    base['entrances']=[{'id':'E_west','kind':'facade','x':8,'y':15,'z':8,'extent':2},
                       {'id':'E_east','kind':'facade','x':22,'y':15,'z':8,'extent':2}]
    base['notes']=['New single-level external approach scene; not a held-out test site.']
    descriptions={
        'aligned':'Level approaches, a 2.4 m deck and a 4 m shared landing.',
        'narrow-deck':'Negative control: only 0.8 m wide away from the landing.',
        'missing-floor':'Negative control: a complete one-cell break across the deck.',
        'low-headroom':'A context beam leaves only 1.6 m clear above part of the deck.',
        'blocked-span':'A solid context partition intersects the proposed span.',
        'split-levels':'Approaches at different heights; unsupported by this level-deck constructor.'}
    for name,description in descriptions.items():
        scene=deepcopy(base);scene['scene_id']='sp1-'+name;scene['description']=description
        if name in ('low-headroom','blocked-span'):
            scene['buildings'].append({'id':'B_obstruction','x':[14,17],'y':[0,32],
                'z':[10,11] if name=='low-headroom' else [0,32], 'side':None,'gap_facing_x':None})
        if name=='split-levels':scene['entrances'][1]['z']=10
        yield name,validate_scene(scene)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run');args=parser.parse_args()
    recipe=json.loads((REPO/'experiments/configs/SP1-platform.json').read_text())
    spec=PlatformSpec(**{k:recipe[k] for k in asdict(PlatformSpec())})
    torch.set_num_threads(2);torch.manual_seed(recipe['random_seed']);np.random.seed(recipe['random_seed'])
    config,_,checkpoint=load_model_c(device='cpu')
    store=RunStore(REPO/'.local-artifacts/runs')
    run=store.create('SP1 spatial platform prototype','geometric_prototype',recipe,recipe['random_seed'],provenance(REPO),args.parent_run)
    directory=store.path(run);print('RUN_ID='+run,flush=True);started=time.perf_counter()
    try:
        snapshot=directory/'source.zip';snapshot_source(REPO,snapshot);store.attach(run,snapshot,'exact_source')
        results=[]
        for name,scene in cases():
            material,construction=construct_platform(scene,spec)
            if name=='narrow-deck':
                material[:]=False
                material[7,15,8:24]=True;material[7,13:18,13:18]=True
                construction={'status':'negative_control','reason':'Deliberately narrow connecting strips'}
            if name=='missing-floor':
                material[:,:,11]=False
                construction={'status':'negative_control','reason':'Deliberately removed an entire cross section'}
            spatial,masks=evaluate_platform(scene,material,spec)
            legacy=legacy_diagnostics(scene,material,config)
            baseline=plan_scene(scene,config)
            old=np.zeros_like(material)
            for z,y,x in baseline['material_zyx']:old[z,y,x]=True
            baseline_spatial,baseline_masks=evaluate_platform(scene,old,spec)
            item={'case':name,'scene':scene,'scene_hash':scene_hash(scene),'construction':construction,
                'material_zyx':np.argwhere(material).tolist(),
                'surface_zyx':np.argwhere(masks['surface']).tolist(),
                'centers_zyx':np.argwhere(masks['centers']).tolist(),
                'clearance_zyx':np.argwhere(masks['clearance']).tolist(),
                'spatial':spatial,'legacy':legacy,
                'baseline':{'method':baseline['method'],'material_zyx':baseline['material_zyx'],
                    'legacy':baseline['diagnostics'],'spatial':baseline_spatial,
                    'surface_zyx':np.argwhere(baseline_masks['surface']).tolist(),
                    'clearance_zyx':np.argwhere(baseline_masks['clearance']).tolist()}}
            write_once(directory/(name+'.json'),item);store.attach(run,directory/(name+'.json'),'individual_case')
            results.append(item)
            print(name, 'spatial=',spatial['spatial_gate'],'legacy_joint=',legacy['joint_budget_connectivity'],flush=True)
        matched=all(r['spatial']['spatial_gate']==recipe['expected_spatial_gate'][r['case']] for r in results)
        study={'version':spec.version,'run_id':run,'training':False,'spec':asdict(spec),
            'checkpoint_weights_used':False,'config_source_checkpoint':checkpoint.name,
            'cases':results,'all_expected_outcomes_matched':matched,
            'interpretation':'Illustrative development cases and semantic audit only. New spatial gate is not comparable to old material-access scores or a complete architectural validity claim.'}
        write_once(directory/'study.json',study);store.attach(run,directory/'study.json','gallery_data')
        metrics={'cases':len(results),'expected_outcomes_matched':matched,
            'spatial_passes':sum(r['spatial']['spatial_gate'] for r in results),
            'baseline_spatial_passes':sum(r['baseline']['spatial']['spatial_gate'] for r in results),
            'wall_seconds':time.perf_counter()-started}
        store.finish(run,'completed' if matched else 'failed',metrics,study['interpretation'])
        write_once(REPO/'experiments/records'/f'{run}.json',{'run_id':run,'status':'completed' if matched else 'failed',
            'config':recipe,'metrics':metrics,'artifact_location':directory.relative_to(REPO).as_posix(),'drive_backup':'pending'})
        print(json.dumps({'run_id':run,'metrics':metrics},indent=2))
        return 0 if matched else 1
    except Exception:
        error=traceback.format_exc()
        store.event(run,'error',error)
        if not (directory/'result.json').exists():store.finish(run,'failed',{},error)
        raise


if __name__=='__main__':raise SystemExit(main())
