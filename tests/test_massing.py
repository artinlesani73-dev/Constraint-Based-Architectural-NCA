import copy
import unittest
import numpy as np
from nca.massing import complete_mass, measure_mass, opportunity_region
from nca.volumetric import fixed_probes
from scripts.run_spatial_prototype import cases


class MassingTests(unittest.TestCase):
    def shell(self):
        a = np.zeros((9, 9, 9), bool); a[2:7, 2:7, 2:7] = True; a[3:6, 3:6, 3:6] = False
        return a

    def complete(self, a, **kwargs):
        return complete_mass(a, np.zeros_like(a), np.ones_like(a), **kwargs)

    def test_closed_shell_exact_volume_and_input_preserved(self):
        a = self.shell(); before = a.copy(); out, r, masks = self.complete(a)
        self.assertEqual(int(out.sum()), 125); self.assertEqual(r['added_voxels'], 27)
        np.testing.assert_array_equal(a, before)
        np.testing.assert_array_equal(out, a | masks['added'])
        self.assertFalse(np.any(a & masks['added']))

    def test_open_shell_not_closed_by_context(self):
        a = self.shell(); a[2, 3:6, 3:6] = False
        e = np.zeros_like(a); e[2, 3:6, 3:6] = True
        out, r, _ = complete_mass(a, e, ~e)
        np.testing.assert_array_equal(out, a); self.assertEqual(r['added_voxels'], 0)

    def test_domain_does_not_seal_open_form(self):
        a = self.shell(); a[2, 3:6, 3:6] = False
        region = np.zeros_like(a); region[3:6, 3:6, 3:6] = True
        self.assertEqual(complete_mass(a, np.zeros_like(a), region)[1]['requested_additions'], 0)

    def test_rejected_additions_and_source_violations_retained(self):
        a = self.shell(); e = np.zeros_like(a); e[4, 4, 4] = True; e[2, 2, 2] = True
        region = np.ones_like(a); region[3, 3, 3] = False; region[2, 2, 2] = False
        out, r, masks = complete_mass(a, e, region)
        self.assertEqual(r['requested_additions'], 27); self.assertEqual(r['added_voxels'], 25)
        self.assertEqual(r['rejected_additions'], 2); self.assertEqual(r['source_in_context'], 1)
        self.assertEqual(r['source_outside_domain'], 1); self.assertTrue(out[2, 2, 2])
        self.assertFalse(out[4, 4, 4]); self.assertEqual(int(masks['rejected'].sum()), 2)

    def test_axis_gap_limit_in_physical_units(self):
        a = np.zeros((9, 3, 3), bool); a[1, 1, 1] = True; a[6, 1, 1] = True
        self.assertEqual(self.complete(a, method='axis_span', max_gap_m=3.2)[1]['added_voxels'], 4)
        self.assertEqual(self.complete(a, method='axis_span', max_gap_m=3.19)[1]['added_voxels'], 0)

    def test_consecutive_gaps_not_outer_span_limit(self):
        a = np.zeros((9, 3, 3), bool); a[[1, 4, 7], 1, 1] = True
        self.assertEqual(self.complete(a, method='axis_span', max_gap_m=1.6)[1]['added_voxels'], 4)

    def test_full_grid_boundary_and_empty_are_not_filled(self):
        for a in (np.zeros((5, 5, 5), bool), np.ones((5, 5, 5), bool)):
            np.testing.assert_array_equal(self.complete(a)[0], a)
        a = np.zeros((5, 5, 5), bool); a[:, 2, :] = True
        np.testing.assert_array_equal(self.complete(a)[0], a)

    def test_axis_permutation_and_idempotence(self):
        a = fixed_probes()['open_ends_512']
        out = self.complete(a, method='axis_span')[0]
        transposed = self.complete(a.transpose(2, 1, 0), method='axis_span', axis=2)[0]
        np.testing.assert_array_equal(transposed, out.transpose(2, 1, 0))
        np.testing.assert_array_equal(self.complete(out, method='axis_span')[0], out)

    def test_open_tube_and_detached_plates_expose_same_completion(self):
        a = fixed_probes()['open_ends_512']; plates = np.zeros_like(a)
        plates[7, 11:20, 8:24] = True; plates[15, 11:20, 8:24] = True
        filled = self.complete(a, method='axis_span')[0]
        self.assertEqual(int(filled.sum()), 1296)
        np.testing.assert_array_equal(self.complete(plates, method='axis_span')[0], filled)
        self.assertEqual(self.complete(plates)[1]['added_voxels'], 0)

    def test_mass_volume_is_not_floor_area_and_domain_is_fixed(self):
        a = self.shell(); region = np.ones_like(a); e = np.zeros_like(a)
        r = measure_mass(a, e, region, .5)
        self.assertEqual(r['gross_volume_m3'], 98 * .125); self.assertEqual(r['domain_voxels'], 729)
        self.assertEqual(measure_mass(~a, e, region)['domain_voxels'], 729)
        self.assertIsNone(r['budget_target'])

    def test_scene_domain_preserves_physical_volume_on_refinement(self):
        scene = dict(cases())['aligned']; permitted = np.ones((32, 32, 32), bool); permitted[:6] = False
        domain, report = opportunity_region(scene, permitted)
        fine = copy.deepcopy(scene); fine['grid_size'] *= 2; fine['voxel_size_m'] /= 2; fine['street_levels'] *= 2
        for b in fine['buildings']:
            for key in ('x', 'y', 'z'): b[key] = [v * 2 for v in b[key]]
            b['gap_facing_x'] *= 2
        for e in fine['entrances']:
            for key in ('x', 'y', 'z', 'extent'): e[key] *= 2
        domain_fine, report_fine = opportunity_region(fine, permitted.repeat(2, 0).repeat(2, 1).repeat(2, 2))
        np.testing.assert_array_equal(domain_fine, domain.repeat(2, 0).repeat(2, 1).repeat(2, 2))
        self.assertAlmostEqual(report['domain_volume_m3'], report_fine['domain_volume_m3'])
        self.assertEqual(report['domain_voxels'], 3456)

    def test_invalid_inputs_rejected(self):
        a = self.shell()
        for kwargs in ({'axis': True}, {'method': 'blue'}, {'voxel_size_m': float('nan')}, {'max_gap_m': 0}):
            with self.assertRaises(ValueError): self.complete(a, **kwargs)
        with self.assertRaises(ValueError): complete_mass(a, a, np.zeros_like(a))
        with self.assertRaises(ValueError): measure_mass(a.astype(float), a, ~a)
        with self.assertRaises(ValueError): opportunity_region(dict(cases())['aligned'], np.ones((32, 32, 32), bool), (-1, 0, 0))


if __name__ == '__main__': unittest.main()
