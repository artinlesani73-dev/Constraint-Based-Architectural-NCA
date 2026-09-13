import unittest
import numpy as np
from nca.evaluation import (endpoint_connectivity, eroded_core, geometric_support,
                            ground_openness, material_legality)


class BinaryEvaluationTests(unittest.TestCase):
    def endpoints(self, shape, a, b):
        result = {"A": np.zeros(shape, bool), "B": np.zeros(shape, bool)}
        result["A"][a] = True
        result["B"][b] = True
        return result

    def test_wall_disconnects_entrances_at_elevated_level(self):
        space = np.ones((7, 7, 7), bool)
        space[:, :, 3] = False
        ends = self.endpoints(space.shape, (5, 3, 1), (5, 3, 5))
        result = endpoint_connectivity(space, ends, "A")
        self.assertFalse(result["all_connected"])
        self.assertEqual(result["fraction_reached"], 0)
        space[5, 3, 3] = True
        self.assertTrue(endpoint_connectivity(space, ends, "A")["all_connected"])

    def test_corner_contact_depends_on_declared_neighborhood(self):
        space = np.zeros((3, 3, 3), bool)
        space[0, 0, 0] = space[1, 1, 1] = True
        ends = self.endpoints(space.shape, (0, 0, 0), (1, 1, 1))
        self.assertFalse(endpoint_connectivity(space, ends, "A", 6)["all_connected"])
        self.assertFalse(endpoint_connectivity(space, ends, "A", 18)["all_connected"])
        self.assertTrue(endpoint_connectivity(space, ends, "A", 26)["all_connected"])

    def test_empty_field_is_not_maximally_thick_or_materially_valid(self):
        empty = np.zeros((5, 5, 5), bool)
        self.assertIsNone(eroded_core(empty)["core_fraction"])
        self.assertFalse(material_legality(empty, ~empty)["nonempty"])
        self.assertIsNone(geometric_support(empty, empty)["supported_fraction"])

    def test_erosion_known_cube_and_sheet(self):
        cube = np.zeros((7, 7, 7), bool)
        cube[2:5, 2:5, 2:5] = True
        self.assertEqual(eroded_core(cube)["core_voxels"], 1)
        cube[:] = False
        cube[3, :, :] = True
        self.assertEqual(eroded_core(cube)["core_voxels"], 0)

    def test_openness_counts_empty_space_not_location_of_mass(self):
        empty = np.zeros((3, 3, 3), bool)
        protected = empty.copy()
        protected[0] = True
        self.assertEqual(ground_openness(empty, empty, protected)["open_fraction"], 1)
        self.assertEqual(ground_openness(protected, empty, protected)["open_fraction"], 0)
        self.assertEqual(ground_openness(empty, protected, protected)["open_fraction"], 0)

    def test_floating_component_is_unsupported(self):
        material = np.zeros((5, 5, 5), bool)
        material[1, 0, 0] = material[4, 4, 4] = True
        support = np.zeros_like(material)
        support[0, 0, 0] = True
        self.assertEqual(geometric_support(material, support)["unsupported_voxels"], 1)

    def test_reject_ambiguous_inputs(self):
        space = np.ones((3, 3, 3), bool)
        ends = self.endpoints(space.shape, (0, 0, 0), (2, 2, 2))
        with self.assertRaises(ValueError):
            endpoint_connectivity(space.astype(float), ends, "A")
        ends["A"][2, 0, 0] = True
        with self.assertRaises(ValueError):
            endpoint_connectivity(space, ends, "A")
        with self.assertRaises(ValueError):
            ground_openness(space, space, np.zeros_like(space))


if __name__ == "__main__":
    unittest.main()
