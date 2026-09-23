"""K2 CPU fine-tuning loop; explicit proposal, source and scene provenance."""
from dataclasses import asdict
from pathlib import Path
import random
import sys
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from nca.experiments import digest, read_json
from nca.losses import LossSpec, context_from_scenes, material_envelope
from nca.facade import endpoint_allowance
from nca.interventions import experimental_rollout
from nca.objective import research_terms, weighted_total
from nca.recovery import restore_checkpoint
from scripts.diagnostic_inputs import load_inputs

REPO = Path(__file__).resolve().parents[1]
PROPOSAL = 'experiments/configs/K2-sensitivity.json'


def code_hashes():
    files = list((REPO/'nca').glob('*.py')) + [REPO/p for p in (
        'deploy/model_utils.py', 'deploy/checkpoints.py', 'scripts/diagnostic_inputs.py',
        'scripts/report_corridor_comparison.py', 'scripts/run_sensitivity.py', PROPOSAL)]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(files)}


def contexts(inputs, config, scenes):
    result = {}
    for name in scenes:
        item = inputs[name]
        if not bool(item['feasible'][0]):
            raise ValueError('Infeasible training scene: '+name)
        ctx = context_from_scenes(item['seed'], config, [item['scene']], item['guide'],
                                 material_envelope(item['guide'], item['permitted'], 6), item['feasible'])
        result[name] = (ctx, endpoint_allowance(item['scene'], item['permitted'])[0])
    return result


def make_metadata(recipe, seed, inputs, config, checkpoint):
    proposal = read_json(REPO/PROPOSAL)
    if recipe not in proposal['recipes'] or seed not in proposal['training_seeds']:
        raise ValueError('Unknown recipe or training seed')
    scenes = proposal['training_scenes']
    if len(scenes) != len(set(scenes)) or len(scenes) != proposal['updates_per_run']:
        raise ValueError('Expected each scene exactly once')
    if set(scenes) != {n for n,i in inputs.items() if bool(i['feasible'][0])}:
        raise ValueError('Scene set differs from frozen feasible inputs')
    # Order is materialized before training. It does not consume the firing RNG.
    order = np.random.default_rng(seed).permutation(scenes).tolist()
    return {'protocol': 'K2_training_v1', 'proposal_sha256': digest(REPO/PROPOSAL),
        'config': config, 'checkpoint_sha256': digest(checkpoint), 'code_sha256': code_hashes(),
        'recipe': recipe, 'seed': seed, 'coefficients': proposal['recipes'][recipe],
        'scene_order': order, 'scene_hashes': {n: inputs[n]['scene_hash'] for n in scenes},
        'input_field_hashes': {n: inputs[n]['source_fields']['sha256'] for n in scenes},
        'loss_spec': asdict(LossSpec()), 'rollout_steps': proposal['rollout_steps'],
        'optimizer': {**proposal['optimizer'], 'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False, 'foreach': False, 'fused': False},
        'scheduler': proposal['scheduler'], 'updates': proposal['updates_per_run'],
        'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
        'python_version': sys.version, 'threads': 2, 'device': 'cpu',
        'projection': 'hard_preclamp', 'envelope_radius': 6, 'facade': 'facade_endpoint_v1',
        'seed_scale': .15, 'pool': False, 'amp': False}


class Session:
    def __init__(self, metadata, resume=None):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        cfg, weights, checkpoint = load_model_c()
        _, inputs = load_inputs(REPO)
        expected = make_metadata(metadata['recipe'], metadata['seed'], inputs, cfg, checkpoint)
        if metadata != expected:
            raise ValueError('Training source/config/runtime/scenes differ from checkpoint protocol')
        self.metadata, self.config, self.inputs = metadata, cfg, inputs
        self.contexts = contexts(inputs, cfg, metadata['scene_order'])
        seed = metadata['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.model = UrbanPavilionNCA(dict(cfg)); self.model.load_state_dict(weights); self.model.train()
        opt = metadata['optimizer']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'], betas=tuple(opt['betas']),
            eps=opt['eps'], weight_decay=opt['weight_decay'], amsgrad=opt['amsgrad'], foreach=False, fused=False)
        if metadata['scheduler'] != 'constant':
            raise ValueError('Only the frozen constant scheduler is supported')
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1., total_iters=1)
        self.generator = torch.Generator().manual_seed(seed)
        self.completed = restore_checkpoint(resume, self.model, self.optimizer, self.scheduler,
                                           self.generator, metadata) if resume else 0
        if not 0 <= self.completed <= metadata['updates']:
            raise ValueError('Checkpoint update count outside schedule')

    def step(self):
        if self.completed >= self.metadata['updates']:
            raise ValueError('Frozen schedule already complete')
        name = self.metadata['scene_order'][self.completed]
        item = self.inputs[name]; ctx, allowance = self.contexts[name]
        self.optimizer.zero_grad(set_to_none=True)
        out = experimental_rollout(self.model, item['seed'], item['scaffold'], 'hard_preclamp',
                                   self.metadata['rollout_steps'], self.generator)
        state, raw = out['state'], out['raw_material']
        if not torch.equal(state[:,:self.config['n_frozen']], item['seed'][:,:self.config['n_frozen']]):
            raise ValueError('Frozen context changed')
        if (state[:,self.config['ch_structure']][~ctx.permitted] != 0).any():
            raise ValueError('Illegal material')
        values = research_terms(state, raw, ctx, self.config, allowance, LossSpec(**self.metadata['loss_spec']))
        coeff = self.metadata['coefficients']
        loss = weighted_total(values, coeff['family_weights'], coeff['regularizer_weights'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.metadata['optimizer']['clip_grad_norm'], error_if_nonfinite=True)
        self.optimizer.step(); self.scheduler.step()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('Nonfinite weight')
        self.completed += 1
        row = {'update': self.completed, 'scene': name, 'steps': self.metadata['rollout_steps'],
            'total_loss': float(loss.detach()), 'terms': {k: float(v[0].detach()) for k,v in values['terms'].items()},
            'regularizers': {k:float(v[0].detach()) for k,v in values['regularizers'].items()},
            'mass_ratio': float(values['mass_ratio'][0].detach()), 'gradient_norm_before_clip': float(norm),
            'learning_rate': self.optimizer.param_groups[0]['lr']}
        # Fields correspond to the forward pass BEFORE this update; checkpoint is AFTER.
        return row, {'material': state[:,self.config['ch_structure']].detach().numpy(), 'raw': raw.detach().numpy()}
