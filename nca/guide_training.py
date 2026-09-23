"""F5 isolated guide architecture; same raw objective and constant16 training."""
from pathlib import Path
import random
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from nca.experiments import digest,read_json
from nca.e0 import evaluate
from nca.conditioning import GuidedNCA,GuideContext,guided_rollout,BASELINE,GUIDED
from nca.fitting import Session as F1Session
from nca.raw_access_training import make_metadata as f4_metadata
from nca.sensitivity import contexts
from nca.recovery import restore_checkpoint
from nca.objective import research_terms,weighted_total
from nca.losses import LossSpec
from nca.access import component_access,component_connectivity
from nca.raw_access import raw_component_access
from nca.interventions import experimental_rollout
from scripts.diagnostic_inputs import load_inputs

REPO=Path(__file__).resolve().parents[1]
PROPOSAL='experiments/configs/F5-persistent-guide.json'
CONSTANT='constant16_v1'
RAW='raw_component_objective_v3'
LEGACY='research_objective_v1'
CANDIDATE='component_objective_v2'
EXTRA_EVALUATION_KEYS=('candidate_access','candidate_details','candidate_metrics','candidate_totals')


def make_metadata(recipe,scene,inputs,config,checkpoint,architecture=GUIDED):
    if architecture not in (BASELINE,GUIDED):raise ValueError('Unknown architecture')
    p=read_json(REPO/PROPOSAL);f4=read_json(REPO/'experiments/configs/F4-raw-access-training.json')
    for key in ('recipes','training_scenes','training_seed','updates_per_member','rollout_steps','optimizer','evaluation','final_evaluation','horizon_schedule'):
        if p[key]!=f4[key]:raise ValueError('F5 must match frozen F4: '+key)
    m=f4_metadata(recipe,scene,inputs,config,checkpoint,RAW)
    m.update(protocol='F5_training_v1',proposal_sha256=digest(REPO/PROPOSAL),architecture_version=architecture)
    m['code_sha256'].update({name:digest(REPO/name) for name in
        (PROPOSAL,'scripts/run_guide_training.py','scripts/report_guide_training.py','scripts/check_guide_late_recovery.py')})
    return m


def next_horizon(metadata,completed):
    if isinstance(completed,bool) or not isinstance(completed,int) or not 0<=completed<=metadata['updates']:
        raise ValueError('Invalid horizon cursor')
    return metadata['horizon_schedule'][completed] if completed<metadata['updates'] else None


def objective_pair(state,raw,ctx,cfg,allowance,spec,objective=CANDIDATE):
    old=research_terms(state,raw,ctx,cfg,allowance,spec)
    if objective==RAW:access,details=raw_component_access(raw,ctx.permitted,ctx.endpoints)
    elif objective==CANDIDATE:access,details=component_access(state[:,cfg['ch_structure']],ctx.permitted,ctx.endpoints)
    else:raise ValueError('Unknown objective')
    new={**old,'terms':{**old['terms'],'access':access},'objective_version':objective}
    return old,new,details


class Session(F1Session):
    def __init__(self,metadata,resume=None):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        cfg,weights,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
        expected=make_metadata(metadata['recipe'],metadata['scene'],inputs,cfg,checkpoint,metadata['architecture_version'])
        if metadata!=expected:raise ValueError('F5 source/config/runtime/scenes/schedule differ from protocol')
        self.metadata,self.config,self.inputs=metadata,cfg,inputs
        self.contexts=contexts(inputs,cfg,[metadata['scene']]);seed=metadata['seed']
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if metadata['architecture_version']==GUIDED:
            self.model=GuidedNCA(dict(cfg));self.model.load_original(weights)
            item=inputs[metadata['scene']]
            self.guide_context=GuideContext(metadata['scene'],item['seed'],item['scaffold'],cfg)
        else:
            self.model=UrbanPavilionNCA(dict(cfg));self.model.load_state_dict(weights);self.guide_context=None
        self.model.train()
        opt=metadata['optimizer']
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),
            eps=opt['eps'],weight_decay=opt['weight_decay'],amsgrad=opt['amsgrad'],foreach=opt['foreach'],fused=opt['fused'])
        self.scheduler=torch.optim.lr_scheduler.ConstantLR(self.optimizer,factor=1.,total_iters=1)
        self.generator=torch.Generator().manual_seed(seed)
        self.completed=restore_checkpoint(resume,self.model,self.optimizer,self.scheduler,self.generator,metadata) if resume else 0
        if not 0<=self.completed<=metadata['updates']:raise ValueError('Checkpoint outside schedule')

    def step(self):
        if self.completed>=self.metadata['updates']:raise ValueError('Schedule complete')
        steps=next_horizon(self.metadata,self.completed)
        name=self.metadata['scene_order'][self.completed];item=self.inputs[name];ctx,allow=self.contexts[name]
        self.optimizer.zero_grad(set_to_none=True)
        out=guided_rollout(self.model,item['seed'],item['scaffold'],'hard_preclamp',steps,self.generator,self.guide_context,name)
        state,raw=out['state'],out['raw_material']
        if not torch.equal(state[:,:self.config['n_frozen']],item['seed'][:,:self.config['n_frozen']]):raise ValueError('Frozen context changed')
        if (state[:,self.config['ch_structure']][~ctx.permitted]!=0).any():raise ValueError('Illegal material')
        old,values,details=objective_pair(state,raw,ctx,self.config,allow,LossSpec(**self.metadata['loss_spec']),self.metadata['objective_version'])
        coeff=self.metadata['coefficients'];loss=weighted_total(values,coeff['family_weights'],coeff['regularizer_weights'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        loss.backward()
        guide_norm = (float(torch.linalg.vector_norm(self.model.guide_weight.grad.detach()))
                      if isinstance(self.model,GuidedNCA) else None)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.metadata['optimizer']['clip_grad_norm'],error_if_nonfinite=True)
        self.optimizer.step();self.scheduler.step()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('Nonfinite weights')
        self.completed+=1
        row={'update':self.completed,'scene':name,'steps':steps,
            'total_loss':float(loss.detach()),'terms':{k:float(v[0].detach()) for k,v in values['terms'].items()},
            'regularizers':{k:float(v[0].detach()) for k,v in values['regularizers'].items()},
            'mass_ratio':float(values['mass_ratio'][0].detach()),'gradient_norm_before_clip':float(norm),
            'learning_rate':self.optimizer.param_groups[0]['lr'],
            'legacy_access':float(old['terms']['access'][0].detach()),'candidate_details':details[0]}
        if guide_norm is not None:
            row.update(guide_gradient_norm_before_clip=guide_norm,
                guide_weight_norm_after_update=float(torch.linalg.vector_norm(self.model.guide_weight.detach())))
        return row,{'material':state[:,self.config['ch_structure']].detach().numpy(),'raw':raw.detach().numpy()}

    @torch.no_grad()
    def score(self,steps,firing_seed):
        name = self.metadata['scene']; item = self.inputs[name]
        ctx, allowance = self.contexts[name]
        out = guided_rollout(self.model, item['seed'], item['scaffold'],
            'hard_preclamp', steps, torch.Generator().manual_seed(firing_seed), self.guide_context, name)
        state, raw = out['state'], out['raw_material']
        if not torch.equal(state[:, :self.config['n_frozen']], item['seed'][:, :self.config['n_frozen']]):
            raise ValueError('Evaluation context changed')
        p = state[:, self.config['ch_structure']]
        if not torch.isfinite(state).all() or not torch.isfinite(raw).all() or (p[~ctx.permitted] != 0).any():
            raise ValueError('Invalid evaluation field')
        values = research_terms(state, raw, ctx, self.config, allowance, LossSpec())
        recipes = read_json(REPO / PROPOSAL)['recipes']
        row = {'update': self.completed, 'scene': name, 'steps': steps, 'firing_seed': firing_seed,
            'terms': {k: float(v[0]) for k, v in values['terms'].items()},
            'regularizers': {k: float(v[0]) for k, v in values['regularizers'].items()},
            'mass_ratio': float(values['mass_ratio'][0]), 'metrics': evaluate(state, self.config, item['scene']),
            'totals_under_both_recipes': {r: float(weighted_total(values, c['family_weights'], c['regularizer_weights'])) for r, c in recipes.items()},
            'raw_saturation': {'permitted_below_zero': int((raw[ctx.permitted] < 0).sum()),
                'permitted_above_one': int((raw[ctx.permitted] > 1).sum()),
                'guide_below_zero': int((raw[ctx.coverage] < 0).sum()),
                'guide_voxels': int(ctx.coverage.sum())}}
        fields = {'material': p.numpy(), 'raw': raw.numpy()}
        ctx,_=self.contexts[self.metadata['scene']];p=torch.from_numpy(fields['material'])
        access,details=component_access(p,ctx.permitted,ctx.endpoints)
        # Recompute totals with tensor arithmetic, preserving original reduction order.
        state=self.inputs[self.metadata['scene']]['seed'].clone();state[:,self.config['ch_structure']]=p
        _,values,_=objective_pair(state,torch.from_numpy(fields['raw']),ctx,self.config,
            self.contexts[self.metadata['scene']][1],LossSpec())
        row.update(candidate_access=float(access[0]),candidate_details=details[0],
            candidate_metrics=component_connectivity(p[0].numpy(),ctx.permitted[0].numpy(),ctx.endpoints[0]),
            candidate_totals={r:float(weighted_total(values,c['family_weights'],c['regularizer_weights']))
                for r,c in read_json(REPO/PROPOSAL)['recipes'].items()})
        raw_access,raw_details=raw_component_access(torch.from_numpy(fields['raw']),ctx.permitted,ctx.endpoints)
        # All other terms are identical; reuse their tensors and reduction order.
        raw_values={**values,'terms':{**values['terms'],'access':raw_access},'objective_version':RAW}
        row.update(raw_access=float(raw_access[0]),raw_details=raw_details[0],
            raw_totals={r:float(weighted_total(raw_values,c['family_weights'],c['regularizer_weights']))
                for r,c in read_json(REPO/PROPOSAL)['recipes'].items()})
        return row,fields


def cost_gate(short_seconds,pair_seconds,grid_seconds,startup_seconds):
    p=read_json(REPO/PROPOSAL)
    groups=[short_seconds,pair_seconds,grid_seconds,startup_seconds]
    if any(not g for g in groups) or not all(np.isfinite(v) and v>=0 for g in groups for v in g):
        raise ValueError('Incomplete or invalid horizon timing pilot')
    short,pair,grid=map(max,groups[:3]);startup=max(5.,max(startup_seconds)+3.)
    per=1.5*(startup+64*short+7*pair+grid)
    return {'max_update_seconds':short,
        'max_evaluation_pair_seconds':pair,'max_extra_grid_seconds':grid,
        'startup_allowance_seconds':startup,'safety_factor':1.5,'estimated_member_seconds':per,
        'estimated_total_seconds':4*per,'admitted':per<=p['member_cap_seconds'] and 4*per<=p['study_cap_seconds']}
