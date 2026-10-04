from pathlib import Path
import sys,json,hashlib,platform,zipfile,time
from copy import deepcopy
ROOT=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from nca.mass_generation_cases import generation_scenes
from nca.massing_cases import target_context
from nca.repair_benchmark import condition,context_hash
from nca.massing_targets import evaluate_targets
from nca.incremental_mass_generator import generate_incremental_mass
from anchored_teacher import generate_anchored_mass,CoverageGeneratorSpec
from generation_data import seed_inputs,teacher_distance

OUT=Path(__file__).resolve().parent
def save(path,data):
    with path.open('x',encoding='utf-8') as f: json.dump(data,f,indent=2)

def main():
    torch.set_num_threads(2)
    config,_,checkpoint=load_model_c(device='cpu')
    scenes=dict(generation_scenes());entries=[]
    families=[('aligned','train'),('wide_gap','train'),('partial_obstruction','train'),('offset_interfaces','development')]
    for family,split in families:
        for dy in (-2,0,2):
            scene=deepcopy(scenes[family]);scene['scene_id']=f'g1-{family}-y{dy+2}'
            for e in scene['entrances']: e['y']+=dy
            entries.append(dict(id=scene['scene_id'],family=family,split=split,scene=scene))
    # Reserved families: no teachers or evaluations produced in this preparation.
    for family in ('raised_pair','unequal_building_heights'):
        for variant in (0,1):
            scene=deepcopy(scenes['aligned']);scene['scene_id']=f'g1-{family}-{variant}'
            if family=='raised_pair':
                for e in scene['entrances']:e['z']+=6+variant*2
            else:
                scene['buildings'][0]['z'][1]=16+variant*2
                scene['buildings'][1]['z'][1]=28
                scene['entrances'][1]['z']=13+variant
            entries.append(dict(id=scene['scene_id'],family=family,split='reserved',scene=scene))
    contexts={};seen={}
    for e in entries:
        f,d,_=target_context(e['scene'],config)
        h=context_hash(e['scene'],f,d)
        assert h not in seen, 'Duplicate context'
        seen[h]=e['split'];e['context_sha256']=h
        contexts[e['id']]=(f,d)
    save(OUT/'split-manifest.json',dict(version='g1_context_split_v1',requests=[.16,.24,.32],teacher_seed=0,
        reserved_status='Geometry frozen; labels not built or scored. Synthetic relatives of historical scenes, not external architectural validation.',entries=entries))
    save(OUT/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,config=config,
        checkpoint_weights_used=False,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    (OUT/'examples').mkdir()
    rows=[];started=time.perf_counter()
    for e in entries:
        if e['split']=='reserved':continue
        f,d=contexts[e['id']]
        for request in (.16,.24,.32):
            name=e['id']+f'-v{round(request*100)}';c=condition(e['scene'],f,d,request);inputs=seed_inputs(c)
            spec=CoverageGeneratorSpec(target_fraction=request,max_seconds=15)
            target,route,report=generate_anchored_mass(e['scene'],f,d,0,inputs['anchor'],spec)
            score,_=evaluate_targets(target,e['scene'],f,d)
            baseline,_,base_report=generate_incremental_mass(e['scene'],f,d,0,spec)
            base_score,_=evaluate_targets(baseline,e['scene'],f,d)
            distance=np.full(d.shape,-1,np.int16);trajectory_error=None
            try:distance=teacher_distance(target,inputs['occupancy'])
            except ValueError as exc:trajectory_error=str(exc)
            path=OUT/'examples'/f'{name}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream,context=c,seed=inputs['occupancy'],target=target,distance=distance,baseline=baseline,route=route)
            row=dict(id=name,split=e['split'],family=e['family'],context_sha256=e['context_sha256'],request=request,
                arrays=str(path.relative_to(OUT)),arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                anchor=[int(v) for v in inputs['anchor']],teacher=report,score=score,baseline=base_report,baseline_score=base_score,
                trajectory_error=trajectory_error,max_teacher_distance=int(distance.max()),
                admissible=bool(score['contract_pass'] and trajectory_error is None))
            save(OUT/'examples'/f'{name}.json',row);rows.append(row)
            print(name,score['contract_pass'],base_score['contract_pass'],int(distance.max()),flush=True)
    summary={split:dict(count=sum(r['split']==split for r in rows),
        admitted=sum(r['split']==split and r['admissible'] for r in rows),
        baseline_valid=sum(r['split']==split and r['baseline_score']['contract_pass'] for r in rows)) for split in ('train','development')}
    save(OUT/'dataset.json',dict(version='g1_anchored_teacher_data_v1',rows=rows,summary=summary,
        wall_seconds=time.perf_counter()-started,reserved_labels_generated=0,
        readiness='All requested cases retained. Do not silently filter failures for training.'))
    print(json.dumps(summary))

if __name__=='__main__':main()
