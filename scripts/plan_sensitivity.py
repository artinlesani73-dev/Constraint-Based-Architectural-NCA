"""Post-K1 fixed coefficient sensitivity proposal, not automatic training."""
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,write_once
from nca.losses import FAMILIES
from scripts.report_calibration import load,RUN


def build(records):
    protocol=records['protocol'][0];old=protocol['checkpoint_weights']
    mapping={'access':'access_conn','support':'loadpath'}
    families={n:old[mapping.get(n,n)] for n in FAMILIES}
    regularizers={'density_binary':old['density'],'tv':old['tv'],'cantilever_boundary':old['cantilever']}
    recipes={'mapped_30':{'family_weights':families,'regularizer_weights':regularizers},
             'mass_3':{'family_weights':dict(families,sparsity=3.),'regularizer_weights':dict(regularizers)}}
    directory=RunStore(REPO/'.local-artifacts/runs').path(RUN);rows=[]
    for case in records['model_record']:
        with np.load(directory/case['fields']['path'],allow_pickle=False) as f:
            vectors={n:f[n].astype(np.float64) for n in list(FAMILIES)+list(regularizers)}
        for recipe,weights in recipes.items():
            coefficients={**weights['family_weights'],**weights['regularizer_weights']}
            combined=sum(vectors[n]*coefficients[n] for n in coefficients)
            norms={n:float(np.linalg.norm(vectors[n])*coefficients[n]) for n in coefficients}
            # Signs apply to an infinitesimal negative RAW-gradient step, not Adam.
            dots={n:float(np.dot(vectors[n],combined)) for n in coefficients}
            rows.append({'scene_id':case['scene_id'],'seed':case['seed'],'steps':case['steps'],'recipe':recipe,
                         'combined_gradient_l2':float(np.linalg.norm(combined)),'weighted_component_norms':norms,
                         'negative_gradient_direction_dot':dots})
    config={'protocol':'K2_proposal_v1','status':'prepared_not_run','source_calibration_run':RUN,'recipes':recipes,
            'purpose':'One-factor local mass-coefficient sensitivity on corrected terms; not a validated final recipe',
            'training_seeds':[0,1],'training_scenes':protocol['included_scenes'],'updates_per_run':17,'rollout_steps':16,
            'optimizer':{'type':'Adam','lr':1e-4,'clip_grad_norm':1.},'scheduler':'constant',
            'expected_training_runs':4,'expected_logical_updates':68,
            'evaluation':{'scene_set':'same17 development scenes','firing_seed':2,'horizons':[16,50],
                          'geometry_holdout':False,'baseline_models':['original_checkpoint','W1_procedural']},
            'cpu_time_cap_seconds_per_run':900,'checkpoint_every_updates':1,'paid_compute':False,
            'must_verify_before_run':['exact CPU restart for composed objective/metadata','explicit complete scene order','no overwrite/checkpoint provenance'],
            'notes':['Mapped historical numbers on corrected definitions are not original training parity.',
                     'Only sparsity coefficient differs between arms; both keep all nine families positive.',
                     'Historical cantilever is excluded from the research recipe; boundary-aware version is explicit.',
                     'These recipes probe sensitivity, not final weight selection or holdout generalization.',
                     'Raw-gradient direction estimates below are not Adam update predictions.']}
    return config,rows


def render(config,rows):
    lines=['# K2 local sensitivity proposal','', 'Prepared from K1; not executed. Two fixed recipes differ only in sparsity coefficient30 versus3. Both use the corrected objective and explicit boundary-aware cantilever. Their other numeric coefficients come from the historical checkpoint; that is a comparison control, not a claim of equivalent term scales.',
        '', '| Recipe | Steps | Cases | Median combined gradient norm | Coverage improving raw direction | Sparsity improving raw direction |',
        '|---|---:|---:|---:|---:|---:|']
    for recipe in config['recipes']:
        for steps in (4,16,50):
            selected=[r for r in rows if r['recipe']==recipe and r['steps']==steps]
            count=lambda n:sum(r['negative_gradient_direction_dot'][n]>1e-12 for r in selected)
            lines.append(f'| {recipe} | {steps} | {len(selected)} | {np.median([r["combined_gradient_l2"] for r in selected]):.7g} | {count("coverage")} | {count("sparsity")} |')
    lines+=['','Signs describe an infinitesimal step along the negative combined gradient. They do not model Adam, finite-step effects, future states or actual learning. Zero/inactive term gradients do not count as improving.','',
        'The proposed experiment uses two recipes x two training seeds,17 updates each, every calibration scene once in a recorded deterministic order, 16-step rollouts, Adam1e-4 and norm clipping1. Evaluate all17 development scenes at16/50 steps with firing seed2 and retain checkpoint/W1 baselines. This is68 logical training updates, not a paid pilot or geometry generalization test.',
        '', 'Before execution verify exact CPU restart with the full composed objective, coefficients and scene order in metadata. Checkpoint every update; cap each run at900 seconds and preserve an interrupted result if reached. No automatic Drive or Colab use. Do not select a final recipe solely from the directional table; retain per-family outcomes and failure cases.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    records=load();config,rows=build(records)
    root=REPO/'experiments/configs';root.mkdir(exist_ok=True)
    write_once(root/'K2-sensitivity.json',config)
    write_once(REPO/'experiments/reports/K2-directional-estimates.json',{'source_run':RUN,'cases':rows})
    with (REPO/'docs/next-phase/SENSITIVITY_PLAN.md').open('x',encoding='utf-8') as f:f.write(render(config,rows))
    print('Prepared K2 fixed recipes and directional estimates; no training launched')
