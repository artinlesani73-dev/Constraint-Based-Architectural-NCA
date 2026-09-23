"""Meaningful shape, geometry, feasibility and gradient regressions for losses."""
from dataclasses import replace
import unittest
import torch
from nca.losses import (LossSpec, LossContext, loss_terms, mean_terms, soft_reach,
                        zero_padded_erosion, material_envelope, FAMILIES)


def fixture(batch=1, side=7):
    shape = (batch, side, side, side)
    permitted = torch.ones(shape, dtype=torch.bool)
    permitted[:, 0] = False
    protected = ~permitted
    guide = torch.zeros_like(permitted)
    guide[:, 3, 3, 1:6] = True
    source = torch.zeros_like(guide)
    source[:, 3, 3, 1] = True
    endpoints = []
    for i in range(batch):
        a, b = torch.zeros_like(guide[i]), torch.zeros_like(guide[i])
        a[3, 3, 1] = b[3, 3, 5] = True
        endpoints.append({'a': a, 'b': b})
    facade = torch.zeros_like(guide)
    facade[:, 1] = True
    return LossContext(permitted, protected, torch.ones_like(guide), facade,
                       facade, guide, permitted.clone(), source, tuple(endpoints),
                       ('a',) * batch, torch.ones(batch, dtype=torch.bool))


def sample(context):
    generator = torch.Generator().manual_seed(27)
    return torch.rand(context.permitted.shape, generator=generator, dtype=torch.float64) * .8 + .05


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_finite_values_and_gradients_for_every_family(self):
        context = fixture(2)
        p = sample(context).requires_grad_()
        before = p.detach().clone()
        result = loss_terms(p, context, LossSpec(reach_hops=9, thickness_radius=1))
        self.assertEqual(set(result['terms']), set(FAMILIES))
        for name, value in result['terms'].items():
            self.assertEqual(tuple(value.shape), (2,), name)
            self.assertTrue(torch.isfinite(value).all(), name)
            grad, = torch.autograd.grad(value.mean(), p, retain_graph=True)
            self.assertTrue(torch.isfinite(grad).all(), name)
        self.assertTrue(torch.equal(p, before))

    def test_directional_finite_difference_for_all_terms(self):
        context = fixture()
        p = sample(context).requires_grad_()
        direction = torch.randn(p.shape, generator=torch.Generator().manual_seed(4), dtype=p.dtype)
        direction /= direction.norm()
        spec = LossSpec(reach_hops=8, thickness_radius=1)
        values = loss_terms(p, context, spec)['terms']
        h = 1e-6
        plus = loss_terms(p.detach() + h * direction, context, spec)['terms']
        minus = loss_terms(p.detach() - h * direction, context, spec)['terms']
        for name, term in values.items():
            grad, = torch.autograd.grad(term.sum(), p, retain_graph=True)
            expected = ((plus[name] - minus[name]) / (2*h)).item()
            actual = (grad * direction).sum().item()
            self.assertAlmostEqual(actual, expected, delta=2e-6, msg=name)

    def test_batch_equals_individual_values_and_gradients(self):
        context = fixture(2)
        p = sample(context).requires_grad_()
        spec = LossSpec(reach_hops=8, thickness_radius=1)
        batch = loss_terms(p, context, spec)
        for i in range(2):
            separate = p[i:i+1].detach().clone().requires_grad_()
            single = loss_terms(separate, fixture(), spec)
            for name in FAMILIES:
                self.assertTrue(torch.allclose(batch['terms'][name][i], single['terms'][name][0], atol=1e-12), name)
                full_grad, = torch.autograd.grad(batch['terms'][name].mean(), p, retain_graph=True)
                small_grad, = torch.autograd.grad(single['terms'][name].mean(), separate, retain_graph=True)
                self.assertTrue(torch.allclose(full_grad[i:i+1] * 2, small_grad, atol=1e-12), name)
                self.assertTrue(torch.isfinite(small_grad).all(), name)

    def test_conflicting_mass_and_envelope_refused_without_rescaling(self):
        context = fixture()
        narrow = replace(context, envelope=context.coverage.clone())
        result = loss_terms(sample(context), narrow, LossSpec(reach_hops=4))
        self.assertFalse(result['budget_compatible'][0])
        self.assertEqual(result['coverage_voxels'][0], 5)
        self.assertAlmostEqual(result['minimum_mass'][0].item(), 343 * .03)
        with self.assertRaisesRegex(ValueError, 'indices'):
            mean_terms(result)
        self.assertTrue(loss_terms(sample(context), context, LossSpec(reach_hops=4))['context_valid'][0])

    def test_empty_target_and_infeasible_scene_cannot_hide_in_batch(self):
        context = fixture(2)
        guide = context.coverage.clone()
        guide[1] = False
        for bad in (replace(context, coverage=guide),
                    replace(context, route_feasible=torch.tensor([True, False]))):
            result = loss_terms(sample(context), bad, LossSpec(reach_hops=4))
            self.assertTrue(result['context_valid'][0])
            self.assertFalse(result['context_valid'][1])
            with self.assertRaises(ValueError):
                mean_terms(result)

    def test_coverage_cannot_exceed_upper_budget_without_flag(self):
        context = fixture()
        context = replace(context, coverage=context.permitted)
        result = loss_terms(sample(context), context, LossSpec(reach_hops=4))
        self.assertFalse(result['budget_compatible'][0])

    def test_gradient_directions_for_coverage_spill_and_mass_bounds(self):
        context = fixture()
        envelope = material_envelope(context.coverage, context.permitted, 2)
        context = replace(context, envelope=envelope)
        p = torch.full(context.permitted.shape, .01, dtype=torch.float64, requires_grad=True)
        values = loss_terms(p, context, LossSpec(reach_hops=4))['terms']
        coverage, = torch.autograd.grad(values['coverage'].sum(), p, retain_graph=True)
        spill, = torch.autograd.grad(values['spill'].sum(), p, retain_graph=True)
        sparse, = torch.autograd.grad(values['sparsity'].sum(), p)
        self.assertTrue((coverage[context.coverage] < 0).all())
        self.assertTrue((coverage[~context.coverage] == 0).all())
        self.assertTrue((spill[context.permitted & ~envelope] > 0).all())
        self.assertTrue((sparse[context.budget] < 0).all())
        high = torch.full_like(p, .5, requires_grad=True)
        gradient, = torch.autograd.grad(loss_terms(high, context, LossSpec(reach_hops=4))['terms']['sparsity'].sum(), high)
        self.assertTrue((gradient[context.budget] > 0).all())

    def test_zero_background_thickness_and_boundary(self):
        empty = torch.zeros(1, 7, 7, 7, dtype=torch.float64)
        thin = empty.clone(); thin[:, 3, 2:5, 2:5] = 1
        thick = empty.clone(); thick[:, 2:5, 2:5, 2:5] = 1
        self.assertEqual(zero_padded_erosion(empty, 1).sum(), 0)
        self.assertEqual(zero_padded_erosion(thin, 1).sum(), 0)
        self.assertEqual(zero_padded_erosion(thick, 1).sum(), 1)
        self.assertEqual(zero_padded_erosion(torch.ones_like(empty), 1).sum(), 125)
        result = loss_terms(empty.requires_grad_(), fixture(), LossSpec(reach_hops=4))
        self.assertEqual(result['terms']['thickness'][0], 0)
        self.assertFalse(result['nonempty_material'][0])
        for value in result['terms'].values():
            grad, = torch.autograd.grad(value.sum(), empty, retain_graph=True)
            self.assertTrue(torch.isfinite(grad).all())

    def test_single_source_reach_rejects_disconnected_and_diagonal_routes(self):
        p = torch.zeros(1, 3, 3, 7, dtype=torch.float64)
        p[0, 1, 1, 1:6] = 1
        source = torch.zeros_like(p, dtype=torch.bool); source[0, 1, 1, 1] = True
        self.assertEqual(soft_reach(p, source, 4)[0, 1, 1, 5], 1)
        self.assertEqual(soft_reach(p, source, 3)[0, 1, 1, 5], 0)
        p[0, 1, 1, 3] = 0; p[0, 2, 2, 3] = 1
        self.assertEqual(soft_reach(p, source, 30)[0, 1, 1, 5], 0)

    def test_access_gradient_reaches_unique_bottleneck(self):
        p = torch.zeros(1, 1, 1, 5, dtype=torch.float64)
        p[0, 0, 0] = torch.tensor([.9, .8, .2, .7, .6])
        p.requires_grad_()
        source = torch.zeros_like(p, dtype=torch.bool); source[0, 0, 0, 0] = True
        score = soft_reach(p, source, 4)[0, 0, 0, -1]
        self.assertAlmostEqual(score.item(), .2, places=6)
        grad, = torch.autograd.grad(1-score, p)
        self.assertEqual(grad[0, 0, 0, 2], -1)
        self.assertEqual(grad.abs().sum(), 1)

    def test_support_does_not_transmit_through_empty_space(self):
        p = torch.zeros(1, 1, 1, 6)
        p[0, 0, 0, 3:] = 1
        supports = torch.zeros_like(p, dtype=torch.bool); supports[..., 0] = True
        self.assertEqual(soft_reach(p, supports, 20, True)[..., 5], 0)
        p[..., 1:3] = 1
        self.assertEqual(soft_reach(p, supports, 5, True)[..., 5], 1)

    def test_envelope_radius_does_not_cascade_or_cross_forbidden_wall(self):
        allowed = torch.ones(1, 3, 3, 9, dtype=torch.bool)
        allowed[..., 4] = False
        seed = torch.zeros_like(allowed); seed[0, 1, 1, 2] = True
        result = material_envelope(seed, allowed, 2)
        self.assertTrue(result[0, 1, 1, 0])
        self.assertFalse(result[..., 4:].any())
        self.assertEqual(material_envelope(seed, allowed, 0).sum(), 1)
        self.assertEqual(seed.sum(), 1)

    def test_projection_explains_zero_model_gradient_for_legality_and_ground(self):
        context = fixture()
        raw = sample(context).requires_grad_()
        projected = raw * context.permitted
        result = loss_terms(projected, context, LossSpec(reach_hops=4))
        for name in ('legality', 'ground'):
            gradient, = torch.autograd.grad(result['terms'][name].sum(), raw, retain_graph=True)
            self.assertEqual(result['terms'][name].sum(), 0)
            self.assertEqual(gradient.abs().sum(), 0)

    def test_scene_adapter_certifies_guide_and_preserves_explicit_source(self):
        from deploy.checkpoints import load_model_c
        from deploy.model_utils import UrbanSceneGenerator
        from nca.contract import load_reference_set, to_generator_params
        from nca.legal_corridor import compute_legal_corridor_v1
        from nca.losses import context_from_scenes
        config, _, _ = load_model_c()
        scene = load_reference_set()['ref-01-ground-pair']
        seed, _ = UrbanSceneGenerator(config).generate(to_generator_params(scene), device='cpu')
        target = compute_legal_corridor_v1(seed, config, [scene])
        guide = target['centerline'] > .5
        permitted = target['target'] > .5
        context = context_from_scenes(seed, config, [scene], guide, permitted, torch.tensor([True]))
        self.assertEqual(context.source.sum(), 1)
        self.assertTrue((context.source & guide).any())
        broken = torch.zeros_like(guide)
        with self.assertRaisesRegex(ValueError, 'connectivity'):
            context_from_scenes(seed, config, [scene], broken, permitted, torch.tensor([True]))

    def test_invalid_inputs_and_multi_source_rejected(self):
        context = fixture()
        for p in (torch.zeros(7, 7, 7), torch.full((1, 7, 7, 7), float('nan')),
                  torch.full((1, 7, 7, 7), 1.01)):
            with self.assertRaises(ValueError): loss_terms(p, context)
        source = context.source.clone(); source[0, 3, 3, 2] = True
        with self.assertRaises(ValueError): loss_terms(sample(context), replace(context, source=source))
        for kwargs in ({'min_mass_ratio': float('nan')}, {'reach_hops': True}, {'min_mass_ratio': .3, 'max_mass_ratio': .2}):
            with self.assertRaises(ValueError): LossSpec(**kwargs)


if __name__ == '__main__':
    unittest.main()
