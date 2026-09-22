"""Verify the legacy sampler against the notebook generator as the oracle.

The historical ``UrbanSceneGenerator`` is extracted from
``notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb`` and executed here in
isolation so that it, rather than a reading of it, decides whether the
transcription in ``nca/legacy_scenes.py`` is faithful. The notebook file is read
only and never modified.
"""
import json
import random
import re
import unittest
from pathlib import Path

import numpy as np
import torch

from nca.contract import (RELAXATIONS, declared_existing, entrance_masks,
                          load_reference_set, to_generator_params, validate_scene,
                          verify_state_matches_scene)
from nca.legacy_scenes import (EASY_PARAMS, LEGACY_SET_VERSION, legacy_seed_state,
                               sample_legacy_easy, sample_validated)
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanSceneGenerator

NOTEBOOK = (Path(__file__).resolve().parents[1] / "notebooks" / "model_c"
            / "NB02_AllConstraints_v3_1_C.ipynb")


def load_historical_generator(config):
    """Execute only the notebook's scene-generator class, in its own namespace."""
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in document["cells"]
               if cell["cell_type"] == "code"]
    matches = [re.search(r"class UrbanSceneGenerator.*?(?=\nprint\(|\Z)", source, re.S)
               for source in sources]
    found = [match.group(0) for match in matches if match]
    if len(found) != 1:
        raise AssertionError(f"Expected one historical generator definition, found {len(found)}")
    namespace = {"torch": torch, "random": random, "Tuple": tuple}
    exec(compile(found[0], str(NOTEBOOK), "exec"), namespace)   # noqa: S102 - read-only oracle
    return namespace["UrbanSceneGenerator"](config)


class TranscriptionFidelityTests(unittest.TestCase):
    """The sampler must draw from the same stream and place the same voxels."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config, _, _ = load_model_c()
        cls.historical = load_historical_generator(cls.config)
        cls.deployed = UrbanSceneGenerator(cls.config)

    def test_the_oracle_is_the_notebook_generator(self):
        self.assertTrue(hasattr(self.historical, "_get_difficulty_params"))
        params = self.historical._get_difficulty_params("easy")
        self.assertEqual(params["height_range"], EASY_PARAMS["height_range"])
        self.assertEqual(params["width_range"], EASY_PARAMS["width_range"])
        self.assertFalse(params["height_variance"])
        self.assertEqual(params["n_ground_access"], EASY_PARAMS["n_ground_access"])
        self.assertEqual(params["n_elevated_access"], EASY_PARAMS["n_elevated_access"])

    def test_every_access_point_was_typed_facade_by_the_historical_generator(self):
        for seed in range(12):
            random.seed(seed)
            _, metadata = self.historical.generate("easy", device="cpu")
            kinds = {point["type"] for point in metadata["access_points"]}
            self.assertEqual(kinds, {"facade"},
                             "the ground-anchor branch could only run for type 'ground'")

    def test_legacy_seed_state_reproduces_the_notebook_voxel_for_voxel(self):
        checked = 0
        for seed in range(40):
            scene = sample_legacy_easy(random.Random(seed))
            try:
                scene = validate_scene(scene)
            except ValueError:
                continue                      # rejected sample; covered separately
            random.seed(seed)
            historical_state, metadata = self.historical.generate("easy", device="cpu")
            replayed = legacy_seed_state(scene, self.config)
            with self.subTest(seed=seed):
                self.assertTrue(torch.equal(historical_state, replayed),
                                f"seed {seed}: legacy seed state differs from the notebook")
                self.assertEqual(metadata["gap_width"],
                                 int(re.search(r"gap width (\d+)",
                                               scene["description"]).group(1)))
                self.assertEqual(verify_state_matches_scene(replayed, self.config,
                                                            scene), [])
            checked += 1
        self.assertGreaterEqual(checked, 25, "too few accepted samples to be meaningful")

    def test_the_deployed_generator_diverges_on_anchors_below_the_street_band(self):
        """A deployment change to the anchor rule alters what the model may grow."""
        diverged = agreed = 0
        for seed in range(40):
            scene, reason = sample_validated(seed)
            if scene is None:
                continue
            historical = legacy_seed_state(scene, self.config)
            deployed, _ = self.deployed.generate(to_generator_params(scene), device="cpu")
            index = self.config["ch_anchors"]
            same = torch.equal(historical[:, index], deployed[:, index])
            below = any(entrance["z"] < scene["street_levels"]
                        for entrance in scene["entrances"])
            with self.subTest(seed=seed):
                # Every other channel must still agree; only anchors are affected.
                for channel in ("ch_existing", "ch_access", "ch_ground"):
                    other = self.config[channel]
                    self.assertTrue(torch.equal(historical[:, other], deployed[:, other]))
                self.assertEqual(same, not below,
                                 "anchors diverge exactly when an entrance sits below "
                                 "the street band")
            if below:
                diverged += 1
                extra = int((deployed[0, index] > 0.5).sum() - (historical[0, index] > 0.5).sum())
                self.assertGreater(extra, 0, "the deployed rule only ever adds anchors")
            else:
                agreed += 1
        self.assertGreater(diverged, 0)
        self.assertGreater(agreed, 0)

    def test_the_divergence_widens_the_permitted_region(self):
        from nca.contract import permitted_region
        for seed in range(40):
            scene, _ = sample_validated(seed)
            if scene is None or not scene["legacy_relaxations"]:
                continue
            historical = legacy_seed_state(scene, self.config)
            deployed, _ = self.deployed.generate(to_generator_params(scene), device="cpu")
            index, existing_index = self.config["ch_anchors"], self.config["ch_existing"]
            historical_permitted = permitted_region(
                historical[0, existing_index].numpy() > 0.5,
                historical[0, index].numpy() > 0.5, scene["street_levels"])
            deployed_permitted = permitted_region(
                deployed[0, existing_index].numpy() > 0.5,
                deployed[0, index].numpy() > 0.5, scene["street_levels"])
            # Growth that training forbade at street level becomes legal.
            self.assertTrue(np.all(deployed_permitted >= historical_permitted))
            self.assertGreater(int(deployed_permitted.sum()),
                               int(historical_permitted.sum()))
            return
        self.skipTest("no relaxed scene available in the sampled range")

    def test_anchor_zones_come_only_from_gap_facades_in_this_distribution(self):
        # A consequence of facade-only typing: anchors are the two facade strips,
        # never the wider ground footprint the deployed interface can produce.
        for seed in (0, 1, 2, 3, 4):
            scene, reason = sample_validated(seed)
            if scene is None:
                continue
            state = legacy_seed_state(scene, self.config)
            anchors = state[0, self.config["ch_anchors"]].numpy() > 0.5
            self.assertTrue(anchors.any())
            columns = {int(x) for x in np.unique(np.argwhere(anchors)[:, 2])}
            with self.subTest(seed=seed):
                # Two one-voxel-wide strips, one per building facade.
                self.assertLessEqual(len(columns), 2, columns)


class RelaxationDisciplineTests(unittest.TestCase):
    def test_relaxation_is_declared_only_when_the_scene_needs_it(self):
        needed = unneeded = 0
        for seed in range(40):
            scene = sample_legacy_easy(random.Random(seed))
            below = any(entrance["z"] < scene["street_levels"]
                        for entrance in scene["entrances"])
            self.assertEqual(scene["legacy_relaxations"],
                             ["facade_below_street_band"] if below else [])
            needed += bool(below)
            unneeded += (not below)
        self.assertGreater(needed, 0, "expected some in-distribution scenes below the band")
        self.assertGreater(unneeded, 0, "expected some above it too")

    def test_a_scene_below_the_band_is_refused_without_the_declaration(self):
        scene = None
        for seed in range(40):
            candidate = sample_legacy_easy(random.Random(seed))
            if candidate["legacy_relaxations"]:
                scene = candidate
                break
        self.assertIsNotNone(scene)
        validate_scene(scene)                       # accepted as declared
        stripped = {**scene, "legacy_relaxations": []}
        with self.assertRaises(ValueError) as caught:
            validate_scene(stripped)
        self.assertIn("facade_below_street_band", str(caught.exception))

    def test_unknown_or_repeated_relaxations_are_refused(self):
        scene = sample_legacy_easy(random.Random(0))
        for bad in (["make_it_pass"], ["facade_below_street_band"] * 2, "not-a-list"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    validate_scene({**scene, "legacy_relaxations": bad})

    def test_every_relaxation_is_documented(self):
        for name, explanation in RELAXATIONS.items():
            self.assertTrue(explanation.strip())
            self.assertIn("still", explanation,
                          "a relaxation must say what it does not relax")

    def test_the_designed_reference_set_declares_no_relaxations(self):
        for scene_id, scene in load_reference_set().items():
            with self.subTest(scene=scene_id):
                self.assertEqual(scene["legacy_relaxations"], [])


class RejectionRecordingTests(unittest.TestCase):
    def test_rejections_are_reported_with_a_reason_rather_than_resampled(self):
        reasons = {}
        for seed in range(80):
            scene, reason = sample_validated(seed)
            if scene is None:
                reasons[seed] = reason
        for seed, reason in reasons.items():
            with self.subTest(seed=seed):
                self.assertTrue(reason)
                self.assertNotIn("Traceback", reason)
        # The historical generator only forbade equal z values, so blocks one
        # voxel apart could overlap. If that ever stops happening, the sampler or
        # the contract changed and this test should be revisited rather than deleted.
        self.assertTrue(any("overlap" in reason for reason in reasons.values()),
                        f"expected at least one overlap rejection; saw {reasons}")

    def test_the_frozen_legacy_set_matches_a_fresh_draw_from_its_seeds(self):
        from nca.contract import REFERENCE_SET_DIR, scene_hash
        directory = REFERENCE_SET_DIR.parent / LEGACY_SET_VERSION
        if not (directory / "manifest.json").is_file():
            self.skipTest("legacy set has not been built in this checkout")
        frozen = load_reference_set(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        sampling = manifest["sampling"]
        self.assertEqual(manifest["set_version"], LEGACY_SET_VERSION)
        self.assertEqual(sampling["difficulty"], "easy")
        self.assertEqual(len(frozen), len(sampling["accepted_seeds"]))
        for seed in sampling["accepted_seeds"]:
            scene_id = f"legacy-easy-seed-{seed:03d}"
            with self.subTest(seed=seed):
                self.assertIn(scene_id, frozen)
                fresh, reason = sample_validated(
                    seed, grid_size=sampling["grid_size"],
                    street_levels=sampling["street_levels"],
                    voxel_size_m=sampling["voxel_size_m"], scene_id=scene_id)
                self.assertIsNone(reason)
                self.assertEqual(scene_hash(fresh), scene_hash(frozen[scene_id]))
        self.assertTrue(any(scene["legacy_relaxations"] for scene in frozen.values()))
        self.assertTrue(any(not scene["legacy_relaxations"] for scene in frozen.values()))

    def test_an_accepted_scene_carries_its_provenance_and_set_version(self):
        scene, _ = sample_validated(0)
        if scene is None:
            self.skipTest("seed 0 is rejected in this distribution")
        self.assertTrue(any("NB02" in note for note in scene["notes"]))
        self.assertEqual(LEGACY_SET_VERSION, "legacy_easy_v1")
        self.assertTrue(declared_existing(scene).any())
        self.assertEqual(len(entrance_masks(scene)), 2)


if __name__ == "__main__":
    unittest.main()
