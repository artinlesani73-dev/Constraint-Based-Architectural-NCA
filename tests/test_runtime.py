"""Real PyTorch and API regressions; these are not model-quality benchmarks."""
import tempfile
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient
from deploy.checkpoints import load_model_c
from deploy.model_utils import (LocalLegalityLoss, UrbanPavilionNCA,
                                UrbanSceneGenerator)
from deploy import server
from nca.contract import (fields_from_state, load_reference_set, permitted_region,
                          to_generator_params, verify_state_matches_scene)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config, cls.weights, _ = load_model_c()

    def test_real_checkpoint_and_frozen_context(self):
        model = UrbanPavilionNCA(self.config)
        model.load_state_dict(self.weights, strict=True)
        state, _ = UrbanSceneGenerator(self.config).generate({
            "buildings": [{"x": [1, 9], "y": [0, 12], "z": [0, 14],
                           "gap_facing_x": None, "side": None}],
            "access_points": [{"x": 10, "y": 4, "z": 0, "type": "ground"}],
        })
        frozen = state[:, :self.config["n_frozen"]].clone()
        output = model.grow(state, steps=3)
        self.assertEqual(output.shape, state.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[:, :self.config["n_frozen"]], frozen))
        self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_checkpoint_config_wins_and_missing_assets_fail(self):
        self.assertEqual(self.config["street_levels"], 6)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_model_c(directory)

    def test_preview_accepts_ui_building_without_optional_facade_metadata(self):
        with TestClient(server.app) as client:
            response = client.post("/preview", json={"buildings": [
                {"x": [1, 9], "y": [0, 12], "z": [0, 14]}], "access_points": []})
            self.assertEqual(response.status_code, 200, response.text[:300])
            self.assertIn("anchors", response.json())


class ContractAgainstHistoricalSceneTests(unittest.TestCase):
    """The contract must describe the scene the historical generator actually builds."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config, cls.weights, _ = load_model_c()
        cls.scenes = load_reference_set()

    def test_every_reference_scene_is_realised_exactly(self):
        generator = UrbanSceneGenerator(self.config)
        for scene_id, scene in sorted(self.scenes.items()):
            with self.subTest(scene=scene_id):
                self.assertEqual(scene["street_levels"], self.config["street_levels"])
                state, _ = generator.generate(to_generator_params(scene), device="cpu")
                self.assertEqual(verify_state_matches_scene(state, self.config, scene), [])

    def test_boolean_permitted_region_equals_the_legacy_legality_field(self):
        generator = UrbanSceneGenerator(self.config)
        legality = LocalLegalityLoss(self.config)
        for scene_id, scene in sorted(self.scenes.items()):
            with self.subTest(scene=scene_id):
                state, _ = generator.generate(to_generator_params(scene), device="cpu")
                legacy = legality.compute_legality_field(state)[0].numpy()
                # The historical field is already binary; thresholding it loses nothing.
                self.assertTrue(np.all((legacy == 0) | (legacy == 1)))
                fields = fields_from_state(state, self.config, scene)
                self.assertTrue(np.array_equal(fields["permitted"], legacy > 0.5))

    def test_fields_read_from_a_grown_state_keep_the_declared_context(self):
        scene = self.scenes["ref-06-minimal-smoke"]
        model = UrbanPavilionNCA(self.config)
        model.load_state_dict(self.weights, strict=True)
        state, _ = UrbanSceneGenerator(self.config).generate(
            to_generator_params(scene), device="cpu")
        torch.manual_seed(0)
        grown = model.grow(state, steps=3)
        self.assertEqual(verify_state_matches_scene(grown, self.config, scene), [])
        fields = fields_from_state(grown, self.config, scene, threshold=0.5)
        self.assertEqual(fields["threshold"], 0.5)
        self.assertFalse(np.any(fields["material"] & fields["existing"]))
        self.assertEqual(sorted(fields["endpoints"]), ["E_ground_a", "E_ground_b"])
        # Structure outside the permitted region would mean the online legality
        # mask did not hold; this is a masking check, not a quality result.
        self.assertFalse(np.any(fields["material"] & ~fields["permitted"]))


if __name__ == "__main__":
    unittest.main()
