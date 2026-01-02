import torch
import os
import sys
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from deploy.model_utils import UrbanPavilionNCA, UrbanSceneGenerator
    print("Successfully imported model utilities.")
except ImportError as e:
    print(f"Failed to import model utilities: {e}")
    sys.exit(1)


def test_inference():
    base_dir = os.path.join("notebooks", "sel", "MODEL A")
    config_path = os.path.join(base_dir, "config_step_b.json")
    model_candidates = [
        os.path.join(base_dir, "v31_fixed_geometry.pth"),
        os.path.join(base_dir, "MODEL_A.pth"),
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)

    if model_path is None or not os.path.exists(config_path):
        print("Model or config not found in expected location. Skipping inference test.")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = UrbanPavilionNCA(config)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        print("Successfully loaded model and checkpoint.")

        generator = UrbanSceneGenerator(config)
        params = {
            "buildings": [
                {"x": [1, 9], "y": [0, 27], "z": [0, 14], "side": "left", "gap_facing_x": 9},
                {"x": [23, 32], "y": [0, 19], "z": [0, 14], "side": "right", "gap_facing_x": 23},
            ],
            "access_points": [
                {"x": 12, "y": 23, "z": 0, "type": "ground"},
                {"x": 9, "y": 1, "z": 11, "type": "elevated"},
            ]
        }

        seed_state, _ = generator.generate(params)
        print("Successfully generated seed state.")

        output = model.grow(seed_state, steps=10)
        print(f"Successfully ran inference. Output shape: {output.shape}")

        expected_shape = (
            1,
            config['n_channels'],
            config['grid_size'],
            config['grid_size'],
            config['grid_size'],
        )
        if output.shape == expected_shape:
            print("Output shape matches expectations.")
        else:
            print(f"Unexpected output shape: {output.shape} vs {expected_shape}")

    except Exception as e:
        print(f"Error during inference test: {e}")


if __name__ == "__main__":
    print("Running deployment smoke tests...")
    test_inference()
