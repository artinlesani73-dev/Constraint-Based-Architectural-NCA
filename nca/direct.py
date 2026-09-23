"""Direct voxel optimization control; no shared or learned NCA update rule."""
from dataclasses import asdict
from pathlib import Path
import random,sys
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from nca.experiments import read_json,digest
from nca.sensitivity import contexts
from nca.objective import research_terms,weighted_total
from nca.losses import LossSpec
from nca.recovery import restore_checkpoint
from scripts.diagnostic_inputs import load_inputs

REPO=Path(__file__).resolve().parents[1]
CONFIG='experiments/configs/D1-direct.json'

class DirectField(torch.nn.Module):
    def __init__(self,raw,permitted):
        super().__init__()
        if raw.ndim!=4 or not raw.is_floating_point() or not torch.isfinite(raw).all():
            raise ValueError('Expected finite floating [B,D,H,W] raw field')
        if permitted.dtype!=torch.bool or permitted.shape!=raw.shape or permitted.device!=raw.device:
            raise ValueError('Expected matching boolean permitted mask')
        self.raw=torch.nn.Parameter(raw.detach().clone())
        self.register_buffer('permitted',permitted.detach().clone())
    def forward(self):
        return self.raw.clamp(0,1)*self.permitted


def metadata(scene,recipe,cfg,checkpoint,inputs):
    proposal=read_json(REPO/CONFIG)
    if scene not in proposal['scenes'] or recipe not in proposal['recipes']:
        raise ValueError('Unregistered scene/recipe')
    if not bool(inputs[scene]['feasible'][0]):raise ValueError('Infeasible direct solve')
    files=list((REPO/'nca').glob('*.py'))+[REPO/p for p in (
        'deploy/checkpoints.py','deploy/model_utils.py','scripts/diagnostic_inputs.py',
        'scripts/report_corridor_comparison.py','scripts/run_sensitivity.py','scripts/run_direct.py',CONFIG)]
    return {'version':'direct_field_v1','scene':scene,'scene_hash':inputs[scene]['scene_hash'],
        'input_field_sha256':inputs[scene]['source_fields']['sha256'],'recipe':recipe,
        'coefficients':proposal['recipes'][recipe],'config':cfg,'loss_spec':asdict(LossSpec()),
        'checkpoint_sha256':digest(checkpoint),'proposal_sha256':digest(REPO/CONFIG),
        'code_sha256':{p.relative_to(REPO).as_posix():digest(p) for p in sorted(files)},
        'initialization':'clamp(seed_material + 0.15 * corridor_legal_v1 scaffold,0,1)',
        'projection':'clamp(raw,0,1) times permitted; raw remains unbounded',
        'optimizer':proposal['optimizer'],'max_updates':proposal['full_updates'],
        'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),'python_version':sys.version,
        'device':'cpu','threads':2,'seed':0,'firing':'none; direct per-scene parameters','pool':False,'amp':False}


class DirectSession:
    def __init__(self,meta,resume=None):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        cfg,_,checkpoint=load_model_c();_,inputs=load_inputs(REPO)
        if meta!=metadata(meta['scene'],meta['recipe'],cfg,checkpoint,inputs):
            raise ValueError('Direct source/config/runtime/scene differs from metadata')
        self.meta=meta;self.config=cfg;self.item=inputs[meta['scene']]
        self.context,self.allowance=contexts(inputs,cfg,[meta['scene']])[meta['scene']]
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        initial=(self.item['seed'][:,cfg['ch_structure']]+.15*self.item['scaffold']).clamp(0,1)
        self.model=DirectField(initial,self.context.permitted)
        opt=meta['optimizer'];self.optimizer=torch.optim.Adam(self.model.parameters(),lr=opt['lr'],
            betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=0.,amsgrad=False,foreach=False,fused=False)
        self.scheduler=torch.optim.lr_scheduler.ConstantLR(self.optimizer,factor=1.,total_iters=1)
        self.generator=torch.Generator().manual_seed(0) # retained for standard checkpoint schema; unused
        self.completed=restore_checkpoint(resume,self.model,self.optimizer,self.scheduler,self.generator,meta) if resume else 0
        if not 0<=self.completed<=meta['max_updates']:raise ValueError('Invalid completed counter')

    def objective(self):
        material=self.model();seed=self.item['seed'];ch=self.config['ch_structure']
        state=torch.cat((seed[:,:ch],material[:,None],seed[:,ch+1:]),dim=1)
        values=research_terms(state,self.model.raw,self.context,self.config,self.allowance,LossSpec(**self.meta['loss_spec']))
        c=self.meta['coefficients'];loss=weighted_total(values,c['family_weights'],c['regularizer_weights'])
        return state,values,loss

    def step(self):
        if self.completed>=self.meta['max_updates']:raise ValueError('Schedule complete')
        self.optimizer.zero_grad(set_to_none=True)
        state,values,loss=self.objective()
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        loss.backward();gradient=self.model.raw.grad
        if (gradient[~self.context.permitted]!=0).any():raise ValueError('Illegal raw gradient')
        before=gradient.detach().numpy().copy()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.meta['optimizer']['clip_grad_norm'],error_if_nonfinite=True)
        trace={'update':self.completed+1,'loss_before_update':float(loss.detach()),
            'terms_before_update':{k:float(v[0].detach()) for k,v in values['terms'].items()},
            'regularizers_before_update':{k:float(v[0].detach()) for k,v in values['regularizers'].items()},
            'gradient_norm_before_clip':float(norm),'learning_rate':self.optimizer.param_groups[0]['lr']}
        self.optimizer.step();self.scheduler.step();self.completed+=1
        if not torch.isfinite(self.model.raw).all():raise ValueError('Nonfinite direct field')
        return trace,before
