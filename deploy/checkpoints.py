"""One authoritative historical checkpoint/config loader for serving and smoke tests."""
from pathlib import Path
import json
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPO_ROOT / "notebooks" / "model_c"


def load_model_c(directory=None, device="cpu"):
    directory = Path(directory) if directory is not None else MODEL_DIRECTORY
    config_path = directory / "config_step_b.json"
    checkpoint_path = directory / "v31_fixed_geometry.pth"
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Model C asset missing: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config.update(checkpoint["config"])
    for key, value in {"corridor_width": 1, "vertical_envelope": 1,
                       "corridor_seed_scale": 0.15, "ground_max_ratio": 0.05}.items():
        config.setdefault(key, value)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    return config, state_dict, checkpoint_path
