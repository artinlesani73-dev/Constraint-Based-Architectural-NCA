"""Intervention tests compare real forward paths and actual finite derivatives."""
import unittest
from dataclasses import replace
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA, UrbanSceneGenerator
from nca.contract import load_reference_set, to_generator_params
from nca.legal_corridor import compute_legal_corridor_v1
from nca.rollout import historical_training, run_rollout
from nca.losses import LossSpec, material_envelope
from nca.interventions import experimental_rollout, guidance_loss, smooth_clip, budget_bounds, objective_terms
from test_losses import fixture


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_hard_arms_preserve_forward_firing_and_rng(self):
        cfg, weights, _ = load_model_c()
        model = UrbanPavilionNCA(cfg); model.load_state_dict(weights)
        scene = load_reference_set()['ref-01-ground-pair']
        seed, _ = UrbanSceneGenerator(cfg).generate(to_generator_params(scene), device='cpu')
        scaffold = compute_legal_corridor_v1(seed, cfg, [scene])['target']
        profile = historical_training(cfg).replace(rng_source='explicit')
        before = seed.clone()
        generators = [torch.Generator().manual_seed(0) for _ in range(3)]
        with torch.no_grad():
            old = run_rollout(model, seed, profile, scaffold, steps=4, schedule_position=60, generator=generators[0])['state']
            for arm, gen in zip(('hard_projected', 'hard_preclamp'), generators[1:]):
                new = experimental_rollout(model, seed, scaffold, arm, 4, gen)['state']
                self.assertTrue(torch.equal(old, new))
                self.assertTrue(torch.equal(gen.get_state(), generators[0].get_state()))
        self.assertTrue(torch.equal(seed, before))

    def test_preclamp_gradient_survives_negative_candidate_and_matches_difference(self):
        raw = torch.tensor([[[[-.01, .2, 1.2]]]], dtype=torch.float64, requires_grad=True)
        state = raw.clamp(0, 1)[:, None]
        guide = torch.ones_like(raw, dtype=torch.bool)
        config = {'ch_structure': 0}
        old = guidance_loss(state, raw, guide, config, 'hard_projected')
        new = guidance_loss(state, raw, guide, config, 'hard_preclamp')
        g_old, = torch.autograd.grad(old.sum(), raw, retain_graph=True)
        g_new, = torch.autograd.grad(new.sum(), raw)
        self.assertEqual(g_old[0, 0, 0, 0], 0)
        self.assertAlmostEqual(g_new[0, 0, 0, 0].item(), -1/3)
        h = 1e-6
        plus = raw.detach().clone(); plus[0, 0, 0, 0] += h
        minus = raw.detach().clone(); minus[0, 0, 0, 0] -= h
        diff = (guidance_loss(plus[:, None], plus, guide, config, 'hard_preclamp') - guidance_loss(minus[:, None], minus, guide, config, 'hard_preclamp'))/(2*h)
        self.assertAlmostEqual(diff.item(), g_new[0, 0, 0, 0].item(), places=9)

    def test_smooth_forward_derivative_and_background_are_explicit(self):
        raw = torch.tensor([-.2, -.01, 0., .3, .8, 1.01, 1.2], dtype=torch.float64, requires_grad=True)
        output = smooth_clip(raw)
        gradient, = torch.autograd.grad(output.sum(), raw)
        h = 1e-6
        numeric = (smooth_clip(raw.detach()+h)-smooth_clip(raw.detach()-h))/(2*h)
        self.assertTrue(torch.allclose(gradient, numeric, atol=1e-9))
        self.assertTrue(((output > 0) & (output < 1)).all())
        self.assertGreater(output[2], .03)

    def test_budget_change_is_explicit_and_does_not_auto_expand(self):
        c = fixture()
        c = replace(c, envelope=material_envelope(c.coverage, c.permitted, 2))
        a, b = budget_bounds(c, 'site'), budget_bounds(c, 'envelope')
        self.assertEqual(a['denominator_voxels'][0], 343)
        self.assertEqual(b['denominator_voxels'][0], c.envelope.sum())
        self.assertLess(b['minimum_mass'][0], a['minimum_mass'][0])
        self.assertTrue(torch.equal(a['envelope_capacity'], b['envelope_capacity']))
        narrow = replace(c, envelope=c.coverage)
        self.assertFalse(budget_bounds(narrow, 'envelope')['budget_compatible'][0])  # guide exceeds 12%
        with self.assertRaises(ValueError): budget_bounds(c, 'automatic')

    def test_envelope_budget_still_counts_spilled_material(self):
        c = fixture()
        c = replace(c, envelope=material_envelope(c.coverage, c.permitted, 2))
        p = torch.zeros(c.permitted.shape, dtype=torch.float64)
        p[c.permitted & ~c.envelope] = .2
        r = objective_terms(p[:, None], p, c, {'ch_structure':0}, 'hard_projected', 'envelope', LossSpec(reach_hops=4))
        expected = p.sum()/c.envelope.sum()
        self.assertAlmostEqual(r['mass_ratio'][0].item(), expected.item())
        self.assertGreater(r['terms']['spill'][0], 0)

    def test_smooth_rollout_preserves_frozen_context_and_hard_legality(self):
        cfg, weights, _ = load_model_c()
        model = UrbanPavilionNCA(cfg); model.load_state_dict(weights)
        scene = load_reference_set()['ref-01-ground-pair']
        seed, _ = UrbanSceneGenerator(cfg).generate(to_generator_params(scene), device='cpu')
        scaffold = compute_legal_corridor_v1(seed, cfg, [scene])['target']
        with torch.no_grad():
            r=experimental_rollout(model, seed, scaffold, 'smooth_projected', 4, torch.Generator().manual_seed(0))
        from deploy.model_utils import LocalLegalityLoss
        legal=LocalLegalityLoss(cfg).compute_legality_field(seed)>.5
        self.assertTrue(torch.equal(r['state'][:,:cfg['n_frozen']],seed[:,:cfg['n_frozen']]))
        self.assertEqual(r['state'][:,cfg['ch_structure']][~legal].abs().sum(),0)
        self.assertTrue(torch.isfinite(r['state']).all())


if __name__ == '__main__': unittest.main()
