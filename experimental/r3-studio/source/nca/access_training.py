"""F2 opt-in access-only learning; F1 and production defaults remain immutable."""
from pathlib import Path
import random
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from nca.experiments import digest,read_json
from nca.fitting import Session as F1Session,make_metadata as f1_metadata
from nca.sensitivity import contexts
from nca.recovery import restore_checkpoint
from nca.objective import research_terms,weighted_total
from nca.losses import LossSpec
from nca.access import component_access,component_connectivity
from nca.interventions import experimental_rollout
from scripts.diagnostic_inputs import load_inputs

REPO=Path(__file__).resolve().parents[1]
PROPOSAL='experiments/configs/F2-access-training.json'
LEGACY='research_objective_v1'
CANDIDATE='component_objective_v2'
EXTRA_EVALUATION_KEYS=('candidate_access','candidate_details','candidate_metrics','candidate_totals')


def make_metadata(recipe,scene,inputs,config,checkpoint,objective=CANDIDATE):
    if objective not in (LEGACY,CANDIDATE):raise ValueError('Unknown objective version')
    p=read_json(REPO/PROPOSAL);f1=read_json(REPO/'experiments/configs/F1-fitting.json')
    for key in ('recipes','training_scenes','training_seed','updates_per_member','rollout_steps','optimizer','evaluation'):
        if p[key]!=f1[key]:raise ValueError('F2 must match frozen F1: '+key)
    m=f1_metadata(recipe,scene,inputs,config,checkpoint)
    m.update(protocol='F2_training_v1',objective_version=objective,proposal_sha256=digest(REPO/PROPOSAL))
    m['code_sha256'].update({name:digest(REPO/name) for name in
        (PROPOSAL,'scripts/run_access_training.py')})
    return m


def objective_pair(state,raw,ctx,cfg,allowance,spec):
    old=research_terms(state,raw,ctx,cfg,allowance,spec)
    access,details=component_access(state[:,cfg['ch_structure']],ctx.permitted,ctx.endpoints)
    new={**old,'terms':{**old['terms'],'access':access},'objective_version':CANDIDATE}
    return old,new,details


class Session(F1Session):
    def __init__(self,metadata,resume=None):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        cfg,weights,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
        expected=make_metadata(metadata['recipe'],metadata['scene'],inputs,cfg,checkpoint,metadata['objective_version'])
        if metadata!=expected:raise ValueError('F2 source/config/runtime/scenes differ from protocol')
        self.metadata,self.config,self.inputs=metadata,cfg,inputs
        self.contexts=contexts(inputs,cfg,[metadata['scene']]);seed=metadata['seed']
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.model=UrbanPavilionNCA(dict(cfg));self.model.load_state_dict(weights);self.model.train()
        opt=metadata['optimizer']
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),
            eps=opt['eps'],weight_decay=opt['weight_decay'],amsgrad=opt['amsgrad'],foreach=opt['foreach'],fused=opt['fused'])
        self.scheduler=torch.optim.lr_scheduler.ConstantLR(self.optimizer,factor=1.,total_iters=1)
        self.generator=torch.Generator().manual_seed(seed)
        self.completed=restore_checkpoint(resume,self.model,self.optimizer,self.scheduler,self.generator,metadata) if resume else 0
        if not 0<=self.completed<=metadata['updates']:raise ValueError('Checkpoint outside schedule')

    def step(self):
        if self.metadata['objective_version']==LEGACY:
            return super().step()  # Exact legacy implementation for parity gate.
        if self.completed>=self.metadata['updates']:raise ValueError('Schedule complete')
        name=self.metadata['scene_order'][self.completed];item=self.inputs[name];ctx,allow=self.contexts[name]
        self.optimizer.zero_grad(set_to_none=True)
        out=experimental_rollout(self.model,item['seed'],item['scaffold'],'hard_preclamp',self.metadata['rollout_steps'],self.generator)
        state,raw=out['state'],out['raw_material']
        if not torch.equal(state[:,:self.config['n_frozen']],item['seed'][:,:self.config['n_frozen']]):raise ValueError('Frozen context changed')
        if (state[:,self.config['ch_structure']][~ctx.permitted]!=0).any():raise ValueError('Illegal material')
        old,values,details=objective_pair(state,raw,ctx,self.config,allow,LossSpec(**self.metadata['loss_spec']))
        coeff=self.metadata['coefficients'];loss=weighted_total(values,coeff['family_weights'],coeff['regularizer_weights'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.metadata['optimizer']['clip_grad_norm'],error_if_nonfinite=True)
        self.optimizer.step();self.scheduler.step()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('Nonfinite weights')
        self.completed+=1
        row={'update':self.completed,'scene':name,'steps':self.metadata['rollout_steps'],
            'total_loss':float(loss.detach()),'terms':{k:float(v[0].detach()) for k,v in values['terms'].items()},
            'regularizers':{k:float(v[0].detach()) for k,v in values['regularizers'].items()},
            'mass_ratio':float(values['mass_ratio'][0].detach()),'gradient_norm_before_clip':float(norm),
            'learning_rate':self.optimizer.param_groups[0]['lr'],
            'legacy_access':float(old['terms']['access'][0].detach()),'candidate_details':details[0]}
        return row,{'material':state[:,self.config['ch_structure']].detach().numpy(),'raw':raw.detach().numpy()}

    @torch.no_grad()
    def score(self,steps,firing_seed):
        row,fields=super().score(steps,firing_seed)
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
        return row,fields


def cost_gate(update_seconds,evaluation_pair_seconds,startup_seconds):
    p=read_json(REPO/PROPOSAL)
    if not update_seconds or not evaluation_pair_seconds or not startup_seconds:raise ValueError('Incomplete timing pilot')
    if not all(np.isfinite(v) and v>=0 for v in update_seconds+evaluation_pair_seconds+startup_seconds):raise ValueError('Invalid timings')
    update=float(np.quantile(update_seconds,.9));pair=max(evaluation_pair_seconds);startup=max(5.,max(startup_seconds)+3.)
    per_member=1.5*(startup+p['updates_per_member']*update+len(p['evaluation']['boundaries'])*pair)
    return {'p90_update_seconds':update,'max_evaluation_pair_seconds':pair,'startup_allowance_seconds':startup,
        'safety_factor':1.5,'estimated_member_seconds':per_member,'estimated_total_seconds':4*per_member,
        'admitted':per_member<=p['member_cap_seconds'] and 4*per_member<=p['study_cap_seconds']}
