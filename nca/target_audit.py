"""T1 target geometry and necessary joint bounds; no production objective change."""
import torch
from nca.losses import LossSpec, material_envelope
from nca.interventions import budget_bounds


def target_candidates(guide, scaffold, permitted):
    """Explicit binary witnesses/controls, not architectural designs."""
    if guide.dtype != torch.bool or permitted.dtype != torch.bool or guide.shape != permitted.shape:
        raise ValueError('Expected matching boolean guide/permitted fields')
    if scaffold.shape != guide.shape or not torch.isfinite(scaffold).all():
        raise ValueError('Invalid scaffold')
    if (guide & ~permitted).any() or ((scaffold > .5) & ~permitted).any():
        raise ValueError('Targets must be legal')
    return {'empty':torch.zeros_like(guide), 'guide':guide.clone(),
            'scaffold':scaffold > .5,
            **{f'radius{r}':material_envelope(guide,permitted,r) for r in (1,3,6)}}


def joint_bounds(context, contract, spec=LossSpec()):
    """Necessary zero-loss bounds for coverage/spill/mass/facade, not sufficiency.

    Coverage zero fixes every guide voxel to one. Its facade cells require
    total mass >= contact/max_facade_ratio. Zero spill and legality confine mass
    to the legal envelope. Optional contact adds another necessary capacity bound:
    even filling all non-facade cells must leave contact fraction <= the cap.
    No new constraint family or softened original bound is introduced.
    """
    b=budget_bounds(context,contract,spec)
    count=lambda mask:mask.flatten(1).sum(1).double()
    required=count(context.coverage & context.facade)
    nonfacade=count(context.envelope & context.permitted & ~context.facade)
    facade_mass_lower=required/spec.max_facade_ratio if spec.max_facade_ratio else torch.where(required>0,torch.inf,0.)
    lower=torch.maximum(torch.maximum(b['minimum_mass'].double(),b['coverage_voxels'].double()),facade_mass_lower)
    upper=torch.minimum(b['maximum_mass'].double(),b['envelope_capacity'].double())
    # mandatory facade fraction cannot be diluted beyond this available volume.
    denominator=required+nonfacade
    best_fraction=torch.where(denominator>0,required/denominator.clamp_min(1),torch.zeros_like(required))
    compatible=(lower <= upper+1e-8) & (best_fraction <= spec.max_facade_ratio+1e-8)
    return {**b,'mandatory_facade_voxels':required,'nonfacade_capacity':nonfacade,
            'facade_required_mass':facade_mass_lower,'joint_minimum_mass':lower,
            'joint_maximum_mass':upper,'minimum_possible_facade_fraction':best_fraction,
            'joint_necessary_compatible':compatible}
