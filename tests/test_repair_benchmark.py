import unittest
import numpy as np
from nca.repair_benchmark import damage, closing_repair, repair_metrics, assert_split_integrity, load_example


class RepairBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((12,)*3, bool)
        self.target[2:10, 2:10, 2:10] = True

    def test_damage_is_repeatable_subset_without_mutating_teacher(self):
        original = self.target.copy()
        for kind in ('cube5', 'slab2'):
            a, cut = damage(self.target, kind, 'case')
            b, _ = damage(self.target, kind, 'case')
            self.assertTrue(np.array_equal(a, b))
            self.assertTrue(np.array_equal(a, self.target & ~cut))
            self.assertGreater(int((self.target & ~a).sum()), 0)
            self.assertTrue(a.any())
        np.testing.assert_array_equal(self.target, original)

    def test_intact_and_empty_guard(self):
        a, cut = damage(self.target, 'intact', 'case')
        np.testing.assert_array_equal(a, self.target)
        self.assertFalse(cut.any())
        with self.assertRaises(ValueError):
            damage(np.zeros_like(a), 'cube5', 'blocked')

    def test_repair_closes_interior_slab_but_keeps_exclusion(self):
        a, _ = damage(self.target, 'slab2', 'case')
        allowed = np.ones_like(a)
        repaired = closing_repair(a, allowed, allowed)
        np.testing.assert_array_equal(repaired, self.target)
        allowed[5, 5, 6] = False
        self.assertFalse(closing_repair(a, allowed, allowed)[5, 5, 6])

    def test_metrics_detect_collateral_damage_and_false_addition(self):
        a, _ = damage(self.target, 'slab2', 'case')
        b = self.target.copy(); b[2, 2, 2] = False; b[0, 0, 0] = True
        m = repair_metrics(b, self.target, a, np.ones_like(a), .24)
        self.assertEqual(m['symmetric_difference_cells'], 2)
        self.assertEqual(m['surviving_cells_removed'], 1)
        self.assertEqual(m['false_positive_cells'], 1)
        self.assertEqual(m['recovery_fraction'], 1.)

    def test_geometry_and_teacher_leakage_rejected(self):
        rows = [{'case':'a','split':'train','context_sha256':'geometry','target_sha256':'target'}]
        self.assertEqual(assert_split_integrity(rows)['distinct_targets'], 1)
        for context, target in [('geometry','other'), ('other','target')]:
            with self.assertRaises(ValueError):
                assert_split_integrity(rows + [{'case':'b','split':'test','context_sha256':context,'target_sha256':target}])

    def test_default_loader_refuses_validation_before_opening_files(self):
        with self.assertRaises(ValueError):
            load_example('.', {'split':'validation'})


if __name__ == '__main__':
    unittest.main()
