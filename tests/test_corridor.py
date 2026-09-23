"""Corridor regressions against independent window and graph expectations."""
import unittest
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import compute_corridor_target_v31, UrbanSceneGenerator
from nca.corridor import bounded_vertical_envelope, compute_corridor_target_bounded_v1
from nca.legal_corridor import route_legal_corridor, compute_legal_corridor_v1
from nca.contract import load_reference_set, to_generator_params, fields_from_state
from nca.evaluation import endpoint_connectivity, flood_fill


def point(shape, p):
    result = np.zeros(shape, bool)
    result[p] = True
    return result


class EnvelopeTests(unittest.TestCase):
    def test_exact_impulse_extent_and_no_mutation(self):
        for z in (0, 1, 4, 8):
            original = torch.zeros(9, 3, 4)
            original[z, 1, 2] = 1
            before = original.clone()
            got = bounded_vertical_envelope(original, 2)
            expected = torch.zeros_like(original)
            expected[max(0, z - 2):min(9, z + 3), 1, 2] = 1
            self.assertTrue(torch.equal(got, expected))
            self.assertTrue(torch.equal(original, before))

    def test_window_oracle_symmetry_leading_axes_and_dtype(self):
        torch.manual_seed(9)
        field = torch.rand(2, 3, 7, 4, 5, dtype=torch.float64)
        for radius in (0, 1, 3, 12):
            expected = torch.stack([field[..., max(0, z-radius):z+radius+1, :, :].amax(-3)
                                    for z in range(7)], dim=-3)
            got = bounded_vertical_envelope(field, radius)
            self.assertTrue(torch.equal(got, expected))
            self.assertEqual(got.dtype, field.dtype)
            self.assertEqual(got.device, field.device)
            self.assertNotEqual(got.data_ptr(), field.data_ptr())
            self.assertTrue(torch.equal(got, bounded_vertical_envelope(field.flip(-3), radius).flip(-3)))

    def test_invalid_radius_and_fields(self):
        for radius in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                bounded_vertical_envelope(torch.zeros(3, 3, 3), radius)
        for field in (torch.zeros(3, 3), torch.zeros(0, 3, 3),
                      torch.full((3, 3, 3), float('nan')), -torch.ones(3, 3, 3)):
            with self.assertRaises(ValueError):
                bounded_vertical_envelope(field, 1)

    def test_legacy_zero_radius_parity_and_fallback(self):
        torch.set_num_threads(2)
        config = {'grid_size': 9, 'ch_access': 0, 'ch_existing': 1, 'corridor_z_margin': 2}
        state = torch.zeros(2, 2, 9, 9, 9)
        state[0, 0, 2:4, 2:4, 1:3] = 1
        state[0, 0, 5:7, 4:6, 6:8] = 1
        state[1, 0, 2:4, 2:4, 1:3] = 1  # fallback branch
        state[:, 1, 0:5, 4:6, 4] = 1
        before = state.clone()
        for radius in (0, 1):
            old = compute_corridor_target_v31(state, config, 1, radius)
            new = compute_corridor_target_bounded_v1(state, config, 1, radius)
            self.assertEqual(new.shape, old.shape)
            self.assertEqual(new.dtype, old.dtype)
            self.assertTrue(torch.equal(new[1], old[1]))
            self.assertTrue(torch.all(new <= old))
            self.assertTrue(torch.all(new[state[:, 1] > 0.5] == 0))
            if radius == 0:
                self.assertTrue(torch.equal(new, old))
        self.assertTrue(torch.equal(state, before))


class LegalRouterTests(unittest.TestCase):
    def test_obstacle_detour_can_exceed_endpoint_height_band(self):
        allowed = np.ones((7, 3, 7), bool)
        allowed[:, :, 3] = False
        allowed[6, 1, 3] = True
        endpoints = {'a': point(allowed.shape, (0, 1, 1)), 'b': point(allowed.shape, (0, 1, 5))}
        got = route_legal_corridor(allowed, endpoints, 0, 0)
        self.assertEqual(got['report']['edges'][0]['length_edges'], 16)
        self.assertTrue(got['target'][6, 1, 3])
        self.assertTrue(endpoint_connectivity(got['target'], endpoints, 'a')['all_connected'])
        self.assertFalse((got['target'] & ~allowed).any())

    def test_no_sixty_four_step_cutoff(self):
        allowed = np.zeros((1, 7, 51), bool)
        allowed[0, ::2] = True
        allowed[0, 1, 50] = allowed[0, 3, 0] = allowed[0, 5, 50] = True
        endpoints = {'a': point(allowed.shape, (0, 0, 0)), 'b': point(allowed.shape, (0, 6, 0))}
        got = route_legal_corridor(allowed, endpoints, 0, 0)
        self.assertEqual(got['report']['edges'][0]['length_edges'], 206)
        self.assertTrue(np.array_equal(got['target'], allowed))

    def test_diagonal_contact_is_infeasible_and_partial_evidence_retained(self):
        allowed = np.eye(3, dtype=bool)[None]
        endpoints = {'a': point(allowed.shape, (0, 0, 0)), 'b': point(allowed.shape, (0, 2, 2))}
        got = route_legal_corridor(allowed, endpoints)
        self.assertEqual(got['report']['status'], 'infeasible')
        self.assertFalse(endpoint_connectivity(got['target'], endpoints, 'a')['all_connected'])
        self.assertEqual(got['centerline'].sum(), 2)

    def test_touching_ids_stay_separate_and_input_order_does_not_change_paths(self):
        allowed = np.ones((4, 4, 4), bool)
        endpoints = {name: point(allowed.shape, p) for name, p in
                     [('a', (1, 1, 1)), ('b', (1, 1, 2)), ('c', (3, 3, 3))]}
        first = route_legal_corridor(allowed, endpoints)
        second = route_legal_corridor(allowed, dict(reversed(list(endpoints.items()))))
        self.assertEqual(first['report'], second['report'])
        self.assertTrue(np.array_equal(first['target'], second['target']))
        self.assertEqual(len(first['report']['edges']), 2)
        self.assertEqual(first['report']['endpoint_ids'], ['a', 'b', 'c'])

    def test_thickening_cannot_jump_a_wall_or_write_illegal_voxels(self):
        allowed = np.ones((3, 3, 5), bool)
        allowed[:, :, 2] = False
        endpoints = {'a': point(allowed.shape, (1, 1, 0)), 'b': point(allowed.shape, (1, 1, 1))}
        got = route_legal_corridor(allowed, endpoints, 2, 1)
        self.assertGreater(got['report']['discarded_isolated_dilation_voxels'], 0)
        self.assertFalse(got['target'][:, :, 2:].any())
        self.assertFalse((got['centerline'] & ~got['target']).any())

    def test_blocked_endpoint_reports_infeasible(self):
        allowed = np.ones((3, 3, 3), bool)
        allowed[0, 0, 0] = False
        endpoints = {'a': point(allowed.shape, (0, 0, 0)), 'b': point(allowed.shape, (2, 2, 2))}
        got = route_legal_corridor(allowed, endpoints)
        self.assertEqual(got['report']['blocked_endpoints'], ['a'])
        self.assertFalse(got['report']['all_endpoints_connected'])

    def test_rejects_ambiguous_regions_and_nonboolean_input(self):
        allowed = np.ones((3, 3, 3), bool)
        a = point(allowed.shape, (0, 0, 0))
        b = point(allowed.shape, (2, 2, 2))
        for permitted, endpoints in ((allowed.astype(float), {'a': a, 'b': b}),
                                      (allowed, {'a': a, 'b': a}),
                                      (allowed, {'a': a | b, 'b': point(allowed.shape, (1, 1, 1))})):
            with self.assertRaises(ValueError):
                route_legal_corridor(permitted, endpoints)

    def test_real_scene_batch_matches_individual_and_preserves_state(self):
        torch.set_num_threads(2)
        config, _, _ = load_model_c()
        scenes = list(load_reference_set().values())[:2]
        states = [UrbanSceneGenerator(config).generate(to_generator_params(s), device='cpu')[0] for s in scenes]
        batch = torch.cat(states).to(torch.float64)
        original = batch.clone()
        got = compute_legal_corridor_v1(batch, config, scenes)
        self.assertTrue(torch.equal(batch, original))
        self.assertEqual(got['target'].dtype, batch.dtype)
        for i, scene in enumerate(scenes):
            single = compute_legal_corridor_v1(batch[i:i+1], config, [scene])
            self.assertTrue(torch.equal(got['target'][i], single['target'][0]))
            fields = fields_from_state(batch[i], config, scene)
            target = got['target'][i].numpy() > 0.5
            self.assertTrue(endpoint_connectivity(target, fields['endpoints'], sorted(fields['endpoints'])[0])['all_connected'])
            self.assertFalse((target & ~fields['permitted']).any())
        with self.assertRaises(ValueError):
            compute_legal_corridor_v1(batch, config, scenes[:1])


if __name__ == '__main__':
    unittest.main()
