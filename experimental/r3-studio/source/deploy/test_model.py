"""Model C smoke test: required assets and invariant failures are fatal."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA, UrbanSceneGenerator


def test_inference():
    config, weights, checkpoint_path = load_model_c()
    model = UrbanPavilionNCA(config)
    model.load_state_dict(weights, strict=True)
    model.eval()
    seed, _ = UrbanSceneGenerator(config).generate({
        "buildings": [{"x": [1, 9], "y": [0, 12], "z": [0, 14]}],
        "access_points": [{"x": 10, "y": 4, "z": 0, "type": "ground"}],
    })
    frozen = seed[:, :config["n_frozen"]].clone()
    output = model.grow(seed, steps=3)
    assert output.shape == seed.shape, (output.shape, seed.shape)
    assert torch.isfinite(output).all(), "Nonfinite state"
    assert torch.equal(output[:, :config["n_frozen"]], frozen), "Frozen scene changed"
    assert ((output >= 0) & (output <= 1)).all(), "State bounds violated"
    print(f"PASS Model C checkpoint + 3-step smoke: {checkpoint_path.name}, shape={tuple(output.shape)}")
    print("Execution/invariant check only; no architectural-quality or rollout-parity claim.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_inference()
