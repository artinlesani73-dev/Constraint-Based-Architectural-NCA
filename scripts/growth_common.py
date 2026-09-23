"""H1 frozen-model diagnostics; loading weights is not optimizer recovery."""
from pathlib import Path
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from scripts.diagnostic_inputs import load_inputs
from scripts.run_sensitivity import STORE,records
from nca.experiments import digest,read_json
from nca.recovery import metadata_hash
from nca.sensitivity import contexts
from nca.access_training import objective_pair
from nca.access import component_connectivity
from nca.losses import LossSpec
from nca.objective import weighted_total
from nca.e0 import evaluate

REPO=Path(__file__).resolve().parents[1]
CONFIG='experiments/configs/H1-growth.json'


def registry(config):
    sources={}
    for arm,key in [('F1','fitting_run'),('F2','access_training_run')]:
        run=config[key];assert not STORE.verify(run)
        assert read_json(STORE.path(run)/'result.json')['status']=='completed'
        protocol=records(run,'protocol')[0];evaluations=records(run,'evaluation_record')
        for branch,meta in protocol['members'].items():
            rows=[r for r in evaluations if r['branch']==branch and r['trace']['update']==64]
            assert {r['trace']['steps'] for r in rows}=={16,50}
            sources[arm+'-'+branch]={'arm':arm,'branch':branch,'scene':meta['scene'],'recipe':meta['recipe'],
                'source_run':run,'metadata':meta,'update':64,'checkpoint':rows[0]['checkpoint'],
                'anchors':{str(r['trace']['steps']):r for r in rows}}
        if arm=='F1':
            for i,scene in enumerate(config['scenes']):
                branch='mapped_30-r'+str(i);rows=[r for r in evaluations if r['branch']==branch and r['trace']['update']==0]
                sources['original-r'+str(i)]={'arm':'original','branch':'original','scene':scene,'recipe':'mapped_30',
                    'source_run':run,'metadata':protocol['members'][branch],'update':0,'checkpoint':rows[0]['checkpoint'],
                    'anchors':{str(r['trace']['steps']):r for r in rows}}
    assert len(sources)==10
    return sources,protocol['proposal']['recipes']


def load_source(source):
    cfg,original,checkpoint=load_model_c();_,inputs=load_inputs(REPO);meta=source['metadata'];name=source['scene']
    assert cfg==meta['config'] and digest(checkpoint)==meta['checkpoint_sha256']
    assert inputs[name]['scene_hash']==meta['scene_hashes'][name]
    assert inputs[name]['source_fields']['sha256']==meta['input_field_hashes'][name]
    path=STORE.path(source['source_run'])/source['checkpoint']['path']
    assert digest(path)==source['checkpoint']['sha256']
    payload=torch.load(path,map_location='cpu',weights_only=True)
    assert payload['metadata']==meta and payload['metadata_hash']==metadata_hash(meta)
    assert payload['completed_updates']==source['update']
    weights=payload['model']
    if source['arm']=='original':assert all(torch.equal(v,original[n]) for n,v in weights.items())
    model=UrbanPavilionNCA(dict(cfg));model.load_state_dict(weights);model.train()
    ctx,allow=contexts(inputs,cfg,[name])[name]
    return model,weights,inputs[name],ctx,allow


def score(state,raw,item,ctx,allow,cfg,recipes):
    assert torch.isfinite(state).all() and torch.isfinite(raw).all()
    assert torch.equal(state[:,:cfg['n_frozen']],item['seed'][:,:cfg['n_frozen']])
    material=state[:,cfg['ch_structure']]
    assert torch.equal(raw.clamp(0,1)*ctx.permitted,material)
    old,new,details=objective_pair(state,raw,ctx,cfg,allow,LossSpec())
    assert bool(old['context_valid'][0])
    terms=lambda v:{k:float(x[0].detach()) for k,x in v.items()}
    return {'terms':terms(old['terms']),'regularizers':terms(old['regularizers']),
        'mass_ratio':float(old['mass_ratio'][0].detach()),'metrics':evaluate(state.detach(),cfg,item['scene']),
        'candidate_access':float(new['terms']['access'][0].detach()),'candidate_details':details[0],
        'candidate_metrics':component_connectivity(material[0].detach().numpy(),ctx.permitted[0].numpy(),ctx.endpoints[0]),
        'totals_v1':{r:float(weighted_total(old,c['family_weights'],c['regularizer_weights']).detach()) for r,c in recipes.items()},
        'totals_v2':{r:float(weighted_total(new,c['family_weights'],c['regularizer_weights']).detach()) for r,c in recipes.items()},
        'raw_saturation':{'permitted_below_zero':int((raw[ctx.permitted]<0).sum()),
            'permitted_above_one':int((raw[ctx.permitted]>1).sum()),
            'guide_below_zero':int((raw[ctx.coverage]<0).sum()),'guide_voxels':int(ctx.coverage.sum())}}


def transition(previous,current):
    """Difference from the previous sampled horizon, not an equilibrium test."""
    a,b=previous>.5,current>.5;union=int(np.logical_or(a,b).sum())
    return {'absolute_material_change_sum':float(np.abs(current.astype(float)-previous.astype(float)).sum()),
        'binary_added':int((b&~a).sum()),'binary_removed':int((a&~b).sum()),
        'binary_iou':float(np.logical_and(a,b).sum()/union) if union else 1.}


def vector_summary(vectors):
    norms={k:float(np.linalg.norm(v.astype(float))) for k,v in vectors.items()}
    cosines={a:{b:float(np.dot(x.astype(float),y.astype(float))/(norms[a]*norms[b]))
        if norms[a] and norms[b] else None for b,y in vectors.items()} for a,x in vectors.items()}
    return norms,cosines


def cost_gate(growth_seconds,gradient_seconds):
    p=read_json(REPO/CONFIG)
    if len(growth_seconds)!=2 or len(gradient_seconds)!=2 or not all(np.isfinite(v) and v>0 for v in growth_seconds+gradient_seconds):
        raise ValueError('Expected two valid growth and two gradient timings')
    estimate=1.5*(30*max(growth_seconds)+8*max(gradient_seconds))
    return {'estimated_seconds':estimate,'safety_factor':1.5,'cap_seconds':p['study_total_cap_seconds'],
        'max_growth_worker_seconds':max(growth_seconds),'max_gradient_worker_seconds':max(gradient_seconds),
        'admitted':estimate<=p['study_total_cap_seconds']}
