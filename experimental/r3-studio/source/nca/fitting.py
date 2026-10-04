"""F1 repeated-scene fitting, using the unchanged K2 optimizer step."""
from dataclasses import asdict
from pathlib import Path
import random
import sys
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from nca.experiments import digest, read_json
from nca.losses import LossSpec
from nca.sensitivity import Session as K2Session, contexts
from nca.recovery import restore_checkpoint
from nca.interventions import experimental_rollout
from nca.objective import research_terms, weighted_total
from nca.e0 import evaluate
from scripts.diagnostic_inputs import load_inputs

REPO = Path(__file__).resolve().parents[1]
PROPOSAL = 'experiments/configs/F1-fitting.json'


def make_metadata(recipe, scene, inputs, config, checkpoint):
    p = read_json(REPO / PROPOSAL)
    if recipe not in p['recipes'] or scene not in p['training_scenes']:
        raise ValueError('Unknown fitting recipe or scene')
    if not bool(inputs[scene]['feasible'][0]):
        raise ValueError('Infeasible fitting scene')
    files = list((REPO / 'nca').glob('*.py')) + [REPO / name for name in (
        'deploy/model_utils.py', 'deploy/checkpoints.py', 'scripts/diagnostic_inputs.py',
        'scripts/report_corridor_comparison.py', 'scripts/run_sensitivity.py',
        'scripts/run_fitting.py', PROPOSAL)]
    return {'protocol': 'F1_training_v1', 'proposal_sha256': digest(REPO / PROPOSAL),
        'code_sha256': {f.relative_to(REPO).as_posix(): digest(f) for f in sorted(files)},
        'config': config, 'checkpoint_sha256': digest(checkpoint), 'recipe': recipe,
        'scene': scene, 'seed': p['training_seed'], 'coefficients': p['recipes'][recipe],
        'scene_order': [scene] * p['updates_per_member'],
        'scene_hashes': {scene: inputs[scene]['scene_hash']},
        'input_field_hashes': {scene: inputs[scene]['source_fields']['sha256']},
        'loss_spec': asdict(LossSpec()), 'rollout_steps': p['rollout_steps'],
        'optimizer': p['optimizer'], 'scheduler': 'constant', 'updates': p['updates_per_member'],
        'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
        'python_version': sys.version, 'threads': 2, 'device': 'cpu',
        'projection': 'hard_preclamp', 'envelope_radius': 6, 'facade': 'facade_endpoint_v1',
        'seed_scale': .15, 'pool': False, 'amp': False}


class Session(K2Session):
    # Inherit K2Session.step exactly. Only schedule and metadata validation differ.
    def __init__(self, metadata, resume=None):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        cfg, weights, checkpoint = load_model_c()
        _, inputs = load_inputs(REPO)
        expected = make_metadata(metadata['recipe'], metadata['scene'], inputs, cfg, checkpoint)
        if metadata != expected:
            raise ValueError('Fitting source/config/runtime/scenes differ from protocol')
        self.metadata, self.config, self.inputs = metadata, cfg, inputs
        self.contexts = contexts(inputs, cfg, [metadata['scene']])
        seed = metadata['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.model = UrbanPavilionNCA(dict(cfg))
        self.model.load_state_dict(weights); self.model.train()
        opt = metadata['optimizer']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'],
            betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'],
            amsgrad=opt['amsgrad'], foreach=opt['foreach'], fused=opt['fused'])
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1., total_iters=1)
        self.generator = torch.Generator().manual_seed(seed)
        self.completed = restore_checkpoint(resume, self.model, self.optimizer,
            self.scheduler, self.generator, metadata) if resume else 0
        if not 0 <= self.completed <= metadata['updates']:
            raise ValueError('Checkpoint update count outside schedule')

    @torch.no_grad()
    def score(self, steps, firing_seed):
        name = self.metadata['scene']; item = self.inputs[name]
        ctx, allowance = self.contexts[name]
        out = experimental_rollout(self.model, item['seed'], item['scaffold'],
            'hard_preclamp', steps, torch.Generator().manual_seed(firing_seed))
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
        return row, {'material': p.numpy(), 'raw': raw.numpy()}


def cost_gate(update_seconds, evaluation_pair_seconds, startup_seconds):
    """Timing-only admission; quality values never enter this decision."""
    p = read_json(REPO / PROPOSAL)
    if not update_seconds or not evaluation_pair_seconds or not startup_seconds:
        raise ValueError('Incomplete timing pilot')
    values = update_seconds + evaluation_pair_seconds + startup_seconds
    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Invalid timings')
    update = float(np.quantile(update_seconds, .9))
    pair = max(evaluation_pair_seconds)
    startup = max(5., max(startup_seconds) + 3.)
    per_member = 1.5 * (startup + p['updates_per_member'] * update + len(p['evaluation']['boundaries']) * pair)
    total = 4 * per_member
    return {'p90_update_seconds': update, 'max_evaluation_pair_seconds': pair,
        'startup_allowance_seconds': startup, 'safety_factor': 1.5,
        'estimated_member_seconds': per_member, 'estimated_total_seconds': total,
        'admitted': per_member <= p['member_cap_seconds'] and total <= p['study_cap_seconds']}
