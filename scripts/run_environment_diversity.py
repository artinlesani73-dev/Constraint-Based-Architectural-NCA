"""One bounded ED1 batch. Retains all outcomes; does not publish presets."""
from pathlib import Path
import sys,json,base64,time,traceback
from hashlib import sha256
from dataclasses import asdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.diversity_sites import diversity_sites
from nca.scale_study import scale_context
from nca.incremental_mass_generator import generate_incremental_mass,CoverageGeneratorSpec,VERSION as GENERATOR
from nca.massing_targets import evaluate_targets,MassingTargetSpec,VERSION as EVALUATOR
from nca.experiments import RunStore,write_once,provenance,snapshot_source,digest
from deploy.checkpoints import load_model_c


def main():
    recipe=json.loads((REPO/'experiments/configs/ED1-diversity.json').read_bytes())
    torch.set_num_threads(recipe['threads']);config,_,checkpoint=load_model_c(device='cpu')
    store=RunStore(REPO/'.local-artifacts/runs')
    run=store.create('ED1 environment diversity','procedural_massing',recipe,6,
        {**provenance(REPO),'config_checkpoint_sha256':digest(checkpoint),'weights_used':False})
    d=store.path(run);print('RUN '+run,flush=True)
    def save(name,value):
        p=d/name;write_once(p,value);store.attach(run,p,'evidence')
    rows=[];presets=[];started=time.perf_counter();state='completed'
    try:
        snapshot_source(REPO,d/'source.zip');store.attach(run,d/'source.zip','source_snapshot')
        for site in diversity_sites():
            key=site['case'];scene=site['scene'];fields,domain,audit=scale_context(scene,config)
            save(key+'.context.json',dict(scene=scene,audit=audit))
            packed=d/(key+'.context.npz');np.savez_compressed(packed,domain=domain,**fields);store.attach(run,packed,'context_arrays')
            raw=packed.read_bytes()
            presets.append(dict(case=key,label=site['label'],size=48,seeds=recipe['seeds'],requests=[.24],scene=scene,
                domain_voxels=int(domain.sum()),arrays_npz_b64=base64.b64encode(raw).decode(),arrays_sha256=sha256(raw).hexdigest(),source_run=run))
            for seed in recipe['seeds']:
                case=key+'__s'+str(seed);spec=CoverageGeneratorSpec(target_fraction=.24,max_seconds=45)
                t=time.perf_counter();field,route,generation=generate_incremental_mass(scene,fields,domain,seed,spec)
                targets,diagnostics=evaluate_targets(field,scene,fields,domain,MassingTargetSpec())
                item=dict(case=case,scene_case=key,seed=seed,request_fraction=.24,generator=GENERATOR,evaluator=EVALUATOR,
                    generator_spec=asdict(spec),evaluator_spec=asdict(MassingTargetSpec()),generation=generation,targets=targets,
                    field_sha256=sha256(field.tobytes()).hexdigest(),seconds=time.perf_counter()-t)
                save(case+'.json',item)
                packed=d/(case+'.npz');np.savez_compressed(packed,field=field,route=route,**diagnostics);store.attach(run,packed,'raw_candidate')
                rows.append(dict(case=case,passed=targets['contract_pass'],failures=[k for k,v in targets['family_pass'].items() if not v],
                    status=generation['status'],volume_m3=targets['gross_volume_m3'],seconds=item['seconds']))
                store.event(run,'case_completed','ED1 candidate retained',**rows[-1]);print(json.dumps(rows[-1]),flush=True)
        save('candidate-presets.json',dict(version='ED1',contexts=presets))
    except BaseException as error:
        state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        save('failure.json',dict(error=str(error),traceback=traceback.format_exc()))
    summary=dict(run_id=run,status=state,cases=rows,completed=len(rows),passed=sum(x['passed'] for x in rows),seconds=time.perf_counter()-started,
        scope='Designed development examples; no new constraints, training or generalization claim.')
    save('study.json',summary);store.finish(run,state,summary,'All attempted outcomes retained.')
    write_once(REPO/'experiments/records'/ (run+'.json'),dict(**summary,artifact_location='.local-artifacts/runs/'+run,drive_backup='pending'))
    print(json.dumps(summary),flush=True)
    return 0 if state=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
