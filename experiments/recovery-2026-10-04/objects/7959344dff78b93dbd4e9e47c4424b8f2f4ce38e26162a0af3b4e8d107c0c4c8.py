"""MD1 independent voxel fitting; CPU only, not an NCA or learned generator."""
from dataclasses import asdict
from hashlib import sha256
import random
import numpy as np
import torch
from nca.massing_objective import massing_residuals
from nca.massing_targets import evaluate_targets, FAMILIES
from nca.recovery import metadata_hash, save_checkpoint, restore_checkpoint

VERSION = 'direct_massing_v1'


class MassLogits(torch.nn.Module):
    def __init__(self, initial, domain):
        super().__init__()
        if initial.dtype != bool or domain.dtype != bool or initial.shape != domain.shape or (initial & ~domain).any():
            raise ValueError('Boolean initial field must lie inside domain')
        p = torch.from_numpy(np.where(initial, .95, .05)).double()
        self.logits = torch.nn.Parameter(torch.logit(p))
        self.register_buffer('domain', torch.from_numpy(domain.copy()))

    def forward(self):
        return self.logits.sigmoid() * self.domain


class MassingSession:
    def __init__(self, initial, context, recipe, identity):
        random.seed(recipe['seed']); np.random.seed(recipe['seed']); torch.manual_seed(recipe['seed'])
        torch.set_num_threads(recipe['threads'])
        self.context, self.recipe = context, recipe
        self.model = MassLogits(initial, context.domain)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **recipe['adam'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.generator = torch.Generator().manual_seed(recipe['seed'])
        self.completed = 0
        if set(recipe['weights']) != set(FAMILIES):
            raise ValueError('Exactly nine family weights required')
        self.metadata = {'version': VERSION, 'identity': identity, 'recipe': recipe,
                         'scene': context.scene, 'spec': asdict(context.spec),
                         'initial_sha256': sha256(initial.tobytes()).hexdigest(),
                         'domain_sha256': sha256(context.domain.tobytes()).hexdigest(),
                         'mask_sha256': {k: sha256(v.tobytes()).hexdigest() for k, v in context.masks.items()}}

    def objective(self):
        p = self.model()
        terms, _ = massing_residuals(p, self.context)
        request = self.recipe['request_fraction']
        preference = ((p.sum() / int(self.context.domain.sum()) - request) / request).square()
        total = sum(self.recipe['weights'][k] * v for k, v in terms.items())
        total = total + self.recipe['request_weight'] * preference
        return total, terms, preference

    def step(self):
        self.optimizer.zero_grad(set_to_none=True)
        loss, terms, preference = self.objective()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite weighted objective')
        loss.backward()
        grad = self.model.logits.grad
        if grad is None or not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite or missing logit gradient')
        evidence = {'update': self.completed + 1, 'pre_update_loss': float(loss.detach()),
                    'pre_update_residuals': {k: float(v.detach()) for k, v in terms.items()},
                    'pre_update_request_preference': float(preference.detach()),
                    'gradient_norm': float(grad.norm()), 'gradient_finite': True}
        self.optimizer.step(); self.scheduler.step()
        if not torch.isfinite(self.model.logits).all():
            raise FloatingPointError('Nonfinite logits after Adam update')
        self.completed += 1
        return evidence

    def evaluate(self):
        with torch.no_grad():
            p = self.model().numpy().copy()
            loss, terms, preference = self.objective()
        binary = p > .5
        c = self.context
        report, masks = evaluate_targets(binary, c.scene, c.masks, c.domain, c.spec)
        request = self.recipe['request_fraction']
        record = {'completed_updates': self.completed, 'loss': float(loss),
                  'residuals': {k: float(v) for k, v in terms.items()},
                  'request_preference': float(preference),
                  'continuous_volume_fraction': float(p.sum() / c.domain.sum()),
                  'binary_request_error_fraction': float(binary.sum() / c.domain.sum() - request),
                  'occupied_zyx': np.argwhere(binary).tolist(),
                  'bulk_zyx': np.argwhere(masks['bulk']).tolist(),
                  'field_sha256': sha256(binary.tobytes()).hexdigest(),
                  'probabilities_sha256': sha256(p.tobytes()).hexdigest(),
                  'targets': report, 'learned': False, 'threshold': 'strict p > 0.5',
                  'enforced_by_projection': ['ground', 'legality', 'spill']}
        return record, p

    def save(self, path):
        return save_checkpoint(path, self.model, self.optimizer, self.scheduler, self.generator,
                               self.metadata, self.completed)

    def restore(self, path):
        self.completed = restore_checkpoint(path, self.model, self.optimizer, self.scheduler,
                                            self.generator, self.metadata)
        return self.completed
