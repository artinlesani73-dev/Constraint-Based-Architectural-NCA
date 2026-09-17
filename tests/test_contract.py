"""Scene contract regressions. NumPy only; no model runs here.

These check that the contract refuses the ambiguous scenes the historical path
accepted, that the frozen reference set is intact, and that derived regions have
the meaning the contract claims. Nothing here measures architectural quality.
"""
import json
import tempfile
import unittest
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from nca.contract import (CONTRACT_VERSION, MANIFEST_NAME, binarise, build_manifest,
                          canonical_json, declared_existing, entrance_masks,
                          fields_from_state, load_reference_set, permitted_region,
                          protected_void, scene_hash, support_region,
                          to_generator_params, validate_scene,
                          verify_state_matches_scene)
from nca.evaluation import endpoint_connectivity, ground_openness, material_legality

CONFIG = {"grid_size": 8, "n_channels": 8, "n_frozen": 4, "ch_ground": 0,
          "ch_existing": 1, "ch_access": 2, "ch_anchors": 3, "ch_structure": 4,
          "street_levels": 4}


def base_scene():
    return {
        "contract_version": CONTRACT_VERSION,
        "scene_id": "unit-pair",
        "description": "Two low blocks with one ground and one facade entrance.",
        "grid_size": 8,
        "voxel_size_m": 0.8,
        "street_levels": 4,
        "ceiling_z": None,
        "buildings": [
            {"id": "B_west", "x": [0, 2], "y": [0, 8], "z": [0, 7],
             "gap_facing_x": 2, "side": "left"},
            {"id": "B_east", "x": [6, 8], "y": [0, 8], "z": [0, 7],
             "gap_facing_x": 6, "side": "right"},
        ],
        "entrances": [
            {"id": "E_ground", "kind": "ground", "x": 3, "y": 3, "z": 0, "extent": 2},
            {"id": "E_facade", "kind": "facade", "x": 2, "y": 3, "z": 5, "extent": 2},
        ],
        "notes": [],
    }


class ValidationTests(unittest.TestCase):
    def reject(self, mutate, fragment):
        scene = base_scene()
        mutate(scene)
        with self.assertRaises(ValueError) as caught:
            validate_scene(scene)
        self.assertIn(fragment, str(caught.exception))

    def test_accepts_the_base_scene(self):
        scene = validate_scene(base_scene())
        self.assertEqual(scene["contract_version"], CONTRACT_VERSION)
        self.assertEqual([b["id"] for b in scene["buildings"]], ["B_west", "B_east"])

    def test_entrance_block_may_not_leave_the_grid(self):
        # The historical generator wrote state[..., z:z+2, y:y+2, x:x+2] with no
        # bounds check, silently producing a one-voxel-thick entrance at the edge.
        self.reject(lambda s: s["entrances"][0].update({"x": 7}), "leaves the grid")

    def test_entrance_may_not_sit_inside_a_building(self):
        self.reject(lambda s: s["entrances"][0].update({"x": 0}),
                    "intersects an existing building")

    def test_entrance_blocks_may_not_overlap(self):
        self.reject(lambda s: s["entrances"][1].update({"kind": "ground", "z": 0, "x": 3}),
                    "entrance blocks overlap")

    def test_ground_entrance_must_stay_inside_the_street_band(self):
        self.reject(lambda s: s["entrances"][0].update({"z": 3}), "street band")

    def test_facade_entrance_must_be_above_the_street_band_and_touch_a_building(self):
        self.reject(lambda s: s["entrances"][1].update({"z": 2}), "at or above")
        # x=3 puts the block at x in [3, 5); neither neighbouring column is built.
        self.reject(lambda s: s["entrances"][1].update({"x": 3}), "face-adjacent")

    def test_facade_adjacency_does_not_wrap_around_the_grid(self):
        scene = base_scene()
        scene["buildings"] = [{"id": "B_west", "x": [0, 2], "y": [0, 8], "z": [0, 7],
                               "gap_facing_x": None, "side": None}]
        scene["entrances"] = [
            {"id": "E_ground", "kind": "ground", "x": 3, "y": 3, "z": 0, "extent": 2},
            # x=6,7 is at the far face; only grid wrapping would touch B_west.
            {"id": "E_far", "kind": "facade", "x": 6, "y": 3, "z": 5, "extent": 2},
        ]
        with self.assertRaises(ValueError):
            validate_scene(scene)

    def test_facade_metadata_must_be_declared_in_full(self):
        self.reject(lambda s: s["buildings"][0].update({"side": None}),
                    "gap_facing_x and side together")

    def test_identity_rules(self):
        self.reject(lambda s: s["entrances"].pop(), "at least two entrances")
        self.reject(lambda s: s["entrances"][1].update({"id": "E_ground"}),
                    "Duplicate entrance id")
        self.reject(lambda s: s["buildings"][1].update({"id": "B_west"}),
                    "Duplicate building id")
        self.reject(lambda s: s.update({"contract_version": "scene_v0"}),
                    "contract_version")

    def test_extent_and_unit_rules(self):
        self.reject(lambda s: s["buildings"][0].update({"x": [2, 2]}), "start < end")
        self.reject(lambda s: s["buildings"][0].update({"x": [0, 99]}), "start < end")
        self.reject(lambda s: s.update({"voxel_size_m": 0}), "finite and positive")
        self.reject(lambda s: s.update({"street_levels": 0}), "street_levels must lie")


class SerialisationTests(unittest.TestCase):
    def test_hash_ignores_key_order_but_not_geometry(self):
        scene = base_scene()
        reordered = dict(reversed(list(scene.items())))
        self.assertEqual(scene_hash(scene), scene_hash(reordered))
        moved = deepcopy(scene)
        moved["entrances"][0]["x"] = 4
        self.assertNotEqual(scene_hash(scene), scene_hash(moved))

    def test_defaults_are_recorded_explicitly_in_the_canonical_form(self):
        scene = base_scene()
        del scene["entrances"][0]["extent"]
        stored = json.loads(canonical_json(scene))
        self.assertEqual(stored["entrances"][0]["extent"], 2)

    def test_generator_params_preserve_declared_geometry(self):
        params = to_generator_params(base_scene())
        self.assertEqual([p["type"] for p in params["access_points"]], ["ground", "facade"])
        self.assertEqual(params["buildings"][0]["x"], (0, 2))

    def test_generator_params_refuse_an_inexpressible_extent(self):
        scene = base_scene()
        scene["entrances"][0]["extent"] = 3
        with self.assertRaises(ValueError) as caught:
            to_generator_params(scene)
        self.assertIn("cannot express extent", str(caught.exception))


class DerivedRegionTests(unittest.TestCase):
    def test_declared_buildings_and_entrances_use_half_open_extents(self):
        existing = declared_existing(base_scene())
        self.assertTrue(existing[0, 0, 1])
        self.assertFalse(existing[0, 0, 2])
        self.assertFalse(existing[7, 0, 0])
        masks = entrance_masks(base_scene())
        self.assertEqual(int(masks["E_ground"].sum()), 8)
        self.assertTrue(masks["E_ground"][0, 3, 3])
        self.assertFalse(masks["E_ground"][2, 3, 3])

    def test_permitted_excludes_buildings_and_unanchored_street_space(self):
        existing = np.zeros((4, 4, 4), bool)
        existing[:, :, 0] = True
        anchors = np.zeros((4, 4, 4), bool)
        anchors[0, 1, 1] = True
        permitted = permitted_region(existing, anchors, street_levels=2)
        self.assertFalse(permitted[0, 0, 1])   # street band, unanchored
        self.assertTrue(permitted[0, 1, 1])    # street band, anchored
        self.assertTrue(permitted[2, 0, 1])    # above the street band
        self.assertFalse(permitted[2, 0, 0])   # inside a building

    def test_protected_void_is_the_unanchored_street_band(self):
        existing = np.zeros((4, 4, 4), bool)
        anchors = np.zeros((4, 4, 4), bool)
        anchors[0, 0, 0] = True
        protected = protected_void(existing, anchors, street_levels=2)
        self.assertFalse(protected[0, 0, 0])
        self.assertTrue(protected[1, 3, 3])
        self.assertFalse(protected[2, 3, 3])

    def test_support_region_is_buildings_plus_anchored_street_footprint(self):
        existing = np.zeros((4, 4, 4), bool)
        existing[0:2, 0, 0] = True
        anchors = np.zeros((4, 4, 4), bool)
        anchors[0, 2, 2] = True
        anchors[3, 3, 3] = True   # above the band; not a support cell
        support = support_region(existing, anchors, street_levels=2)
        self.assertTrue(support[1, 0, 0])
        self.assertTrue(support[0, 2, 2])
        self.assertFalse(support[3, 3, 3])

    def test_binarisation_is_strict_and_requires_an_explicit_threshold(self):
        field = np.array([[[0.0, 0.5, 0.51]]])
        self.assertEqual(binarise(field, 0.5).tolist(), [[[False, False, True]]])
        for bad in (0, 1, True, "0.5"):
            with self.assertRaises(ValueError):
                binarise(field, bad)


class StateReadoutTests(unittest.TestCase):
    def state(self, scene):
        grid = scene["grid_size"]
        array = np.zeros((CONFIG["n_channels"], grid, grid, grid), np.float32)
        array[CONFIG["ch_ground"], 0] = 1.0
        array[CONFIG["ch_existing"]] = declared_existing(scene)
        union = np.zeros((grid,) * 3, bool)
        for mask in entrance_masks(scene).values():
            union |= mask
        array[CONFIG["ch_access"]] = union
        return array

    def test_matching_state_reports_no_problems(self):
        scene = base_scene()
        self.assertEqual(verify_state_matches_scene(self.state(scene), CONFIG, scene), [])

    def test_state_that_lost_geometry_is_reported_not_ignored(self):
        scene = base_scene()
        array = self.state(scene)
        array[CONFIG["ch_access"], 0, 3, 3] = 0.0
        array[CONFIG["ch_existing"], 0, 0, 0] = 0.0
        problems = verify_state_matches_scene(array, CONFIG, scene)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("access channel" in p for p in problems))
        self.assertTrue(any("existing channel" in p for p in problems))

    def test_fields_refuse_a_config_that_disagrees_with_the_scene(self):
        scene = base_scene()
        config = dict(CONFIG, street_levels=2)
        with self.assertRaises(ValueError) as caught:
            fields_from_state(self.state(scene), config, scene)
        self.assertIn("street_levels disagree", str(caught.exception))

    def test_fields_refuse_a_batch(self):
        scene = base_scene()
        batch = np.stack([self.state(scene), self.state(scene)])
        with self.assertRaises(ValueError):
            fields_from_state(batch, CONFIG, scene)

    def test_empty_structure_cannot_read_as_a_valid_design(self):
        scene = base_scene()
        fields = fields_from_state(self.state(scene)[None], CONFIG, scene)
        self.assertFalse(fields["material"].any())
        legality = material_legality(fields["material"], fields["permitted"])
        self.assertFalse(legality["nonempty"])
        self.assertIsNone(legality["illegal_fraction"])
        # An empty design keeps the street band fully open, which is exactly why
        # openness alone must never stand in for a quality result.
        self.assertEqual(ground_openness(fields["material"], fields["existing"],
                                         fields["protected"])["open_fraction"], 1)
        reach = endpoint_connectivity(fields["material"], fields["endpoints"], "E_ground")
        self.assertFalse(reach["source_open"])
        self.assertFalse(reach["all_connected"])


class ReferenceSetTests(unittest.TestCase):
    def test_frozen_set_loads_and_covers_the_intended_cases(self):
        scenes = load_reference_set()
        self.assertEqual(len(scenes), 6)
        self.assertIn("ref-02-facade-pair-and-ground", scenes)
        self.assertIn("ref-05-sealed-partition", scenes)
        for scene_id, scene in scenes.items():
            self.assertEqual(scene["scene_id"], scene_id)
            self.assertEqual(scene["street_levels"], 6)
            self.assertEqual(scene["grid_size"], 32)
            self.assertIsNone(scene["ceiling_z"])

    def test_sealed_partition_has_no_legal_route_between_its_entrances(self):
        scene = load_reference_set()["ref-05-sealed-partition"]
        existing = declared_existing(scene)
        anchors = np.zeros_like(existing)
        permitted = permitted_region(existing, anchors, scene["street_levels"])
        result = endpoint_connectivity(permitted, entrance_masks(scene), "E_facade_west")
        self.assertFalse(result["all_connected"])
        self.assertEqual(result["fraction_reached"], 0)

    def test_an_edited_frozen_scene_fails_loudly(self):
        scenes = load_reference_set()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scene = scenes["ref-06-minimal-smoke"]
            path = directory / "ref-06-minimal-smoke.json"
            path.write_bytes(canonical_json(scene))
            manifest = build_manifest(directory, "test")
            (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
            self.assertEqual(len(load_reference_set(directory)), 1)

            tampered = deepcopy(scene)
            tampered["entrances"][0]["x"] = 12
            path.write_bytes(canonical_json(tampered))
            with self.assertRaises(ValueError) as caught:
                load_reference_set(directory)
            self.assertIn("changed on disk", str(caught.exception))

            # A manifest updated to match the new bytes but not the canonical
            # hash must still be refused.
            entry = manifest["scenes"][0]
            entry["sha256"] = sha256(path.read_bytes()).hexdigest()
            (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
            with self.assertRaises(ValueError) as caught:
                load_reference_set(directory)
            self.assertIn("scene hash mismatch", str(caught.exception))

    def test_an_unlisted_scene_file_fails_loudly(self):
        scenes = load_reference_set()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for scene_id in ("ref-06-minimal-smoke", "ref-01-ground-pair"):
                (directory / f"{scene_id}.json").write_bytes(canonical_json(scenes[scene_id]))
            manifest = build_manifest(directory, "test")
            manifest["scenes"] = [e for e in manifest["scenes"]
                                  if e["scene_id"] == "ref-06-minimal-smoke"]
            (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
            with self.assertRaises(ValueError) as caught:
                load_reference_set(directory)
            self.assertIn("unlisted", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
