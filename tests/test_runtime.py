"""Real PyTorch and API regressions; these are not model-quality benchmarks."""
import tempfile
import unittest
import torch
from fastapi.testclient import TestClient
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA, UrbanSceneGenerator
from deploy import server


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


if __name__ == "__main__":
    unittest.main()
