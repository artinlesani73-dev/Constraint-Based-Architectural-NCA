import os
import json
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
try:
    from .model_utils import UrbanPavilionNCA, UrbanSceneGenerator, compute_corridor_target_v31
except ImportError:
    from model_utils import UrbanPavilionNCA, UrbanSceneGenerator, compute_corridor_target_v31

# Global variables for model and config
model = None
config = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, config, generator
    # Paths - Using MODEL C (deployment target)
    base_dir = os.path.join("notebooks", "sel", "MODEL C - Copy - no change")
    config_path = os.path.join(base_dir, "config_step_b.json")
    model_candidates = [
        os.path.join(base_dir, "v31_fixed_geometry.pth"),
    ]

    if not os.path.exists(config_path):
        config_path = "../notebooks/sel/MODEL C - Copy - no change/config_step_b.json"

    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if model_path is None:
        alt_candidates = [
            "../notebooks/sel/MODEL C - Copy - no change/v31_fixed_geometry.pth",
        ]
        model_path = next((p for p in alt_candidates if os.path.exists(p)), None)

    try:
        if model_path is None or not os.path.exists(config_path):
            raise FileNotFoundError("Model or config not found in expected locations.")

        with open(config_path, 'r') as f:
            config = json.load(f)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = {**config, **checkpoint['config']}
            print("Using checkpoint config overrides.")
        # Use checkpoint values (trained with corridor_width=1, vertical_envelope=1)
        config.setdefault('corridor_width', 1)
        config.setdefault('vertical_envelope', 1)
        config.setdefault('corridor_seed_scale', 0.15)
        config.setdefault('ground_max_ratio', 0.05)
        if 'corridor_z_margin' in config:
            print(f"Corridor z margin: {config.get('corridor_z_margin')}")
        model = UrbanPavilionNCA(config)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        generator = UrbanSceneGenerator(config)
        print("Model and generator loaded successfully.")
        print(f"Model path: {model_path}")
        print(f"Config path: {config_path}")
        print(f"Corridor width: {config.get('corridor_width')}")
        print(f"Vertical envelope: {config.get('vertical_envelope')}")
        print(f"Corridor seed scale: {config.get('corridor_seed_scale')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")
    yield

app = FastAPI(title="Constraint-Based Architectural NCA API", lifespan=lifespan)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi import Response
    return Response(status_code=204)

@app.get("/")
async def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)

class BuildingParams(BaseModel):
    x: Tuple[int, int]
    y: Tuple[int, int]
    z: Tuple[int, int]
    gap_facing_x: Optional[int] = None
    side: Optional[str] = None

class AccessPointParams(BaseModel):
    x: int
    y: int
    z: int
    type: str

class GenerateRequest(BaseModel):
    buildings: List[BuildingParams]
    access_points: List[AccessPointParams]
    steps: int = 50
    seed: Optional[int] = None
    noise_std: float = 0.02
    corridor_seed_scale: float = 0.005
    fire_rate: float = 1.0
    corridor_width: int = 1
    vertical_envelope: int = 1
    threshold: float = 0.5
    update_scale: float = 0.1

class PreviewRequest(BaseModel):
    buildings: List[BuildingParams]
    access_points: List[AccessPointParams]

@app.post("/preview")
async def preview(request: PreviewRequest):
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not loaded")
    
    params = {
        "buildings": [b.model_dump() for b in request.buildings],
        "access_points": [ap.model_dump() for ap in request.access_points]
    }
    
    try:
        seed_state, info = generator.generate(params, device='cpu')
        existing = seed_state[0, config['ch_existing']].numpy()
        access = seed_state[0, config['ch_access']].numpy()
        anchors = seed_state[0, config['ch_anchors']].numpy()
        
        return {
            "existing": existing.tolist(),
            "access": access.tolist(),
            "anchors": anchors.tolist(),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None or generator is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Handle seed (-1 or None means random)
    actual_seed = request.seed
    if request.seed is None or request.seed < 0:
        actual_seed = np.random.randint(0, 2**31)
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)

    print(f"Generate request: steps={request.steps}, seed={actual_seed}, noise={request.noise_std}, fire_rate={request.fire_rate}")
    print(f"Buildings: {[b.model_dump() for b in request.buildings]}")
    print(f"Access points: {[ap.model_dump() for ap in request.access_points]}")

    params = {
        "buildings": [b.model_dump() for b in request.buildings],
        "access_points": [ap.model_dump() for ap in request.access_points]
    }
    
    try:
        seed_state, info = generator.generate(params, device='cpu')
        corridor_target = compute_corridor_target_v31(
            seed_state,
            config,
            corridor_width=request.corridor_width,
            vertical_envelope=request.vertical_envelope,
        )
        print(
            "Corridor stats:",
            f"total={corridor_target.sum().item():.0f},",
            f"ground={corridor_target[:, :config['street_levels']].sum().item():.0f}",
        )
        # Use request corridor_seed_scale instead of config
        seed_scale = request.corridor_seed_scale
        if seed_scale > 0:
            struct_idx = config['ch_structure']
            seed_state[:, struct_idx] = torch.clamp(
                seed_state[:, struct_idx] + seed_scale * corridor_target, 0.0, 1.0
            )
        print(f"Seed scale applied: {seed_scale}")
        mask_epochs = config.get('corridor_mask_epochs', 0)
        mask_anneal = config.get('corridor_mask_anneal', 0)
        print(f"Inference mask: epochs={mask_epochs}, anneal={mask_anneal}")

        # Override update_scale if provided
        original_update_scale = config.get('update_scale', 0.1)
        config['update_scale'] = request.update_scale

        with torch.no_grad():
            state = seed_state
            G = config['grid_size']
            for step in range(request.steps):
                # Add noise for randomness
                if request.noise_std > 0:
                    noise = torch.randn_like(state[:, config['n_frozen']:]) * request.noise_std
                    state[:, config['n_frozen']:] = torch.clamp(
                        state[:, config['n_frozen']:] + noise, 0.0, 1.0
                    )

                # Stochastic firing (if fire_rate < 1.0)
                if request.fire_rate < 1.0:
                    fire_mask = (torch.rand(1, 1, G, G, G) < request.fire_rate).float()
                    old_grown = state[:, config['n_frozen']:].clone()
                    state = model._step(state)
                    new_grown = state[:, config['n_frozen']:]
                    state[:, config['n_frozen']:] = old_grown * (1 - fire_mask) + new_grown * fire_mask
                else:
                    state = model._step(state)

                mask_strength = 0.0
                if step < mask_epochs:
                    mask_strength = 1.0
                elif mask_anneal > 0 and step < mask_epochs + mask_anneal:
                    mask_strength = 1.0 - (step - mask_epochs) / float(mask_anneal)
                if mask_strength > 0:
                    struct_idx = config['ch_structure']
                    state[:, struct_idx] = (
                        state[:, struct_idx] * (1.0 - mask_strength)
                        + state[:, struct_idx] * corridor_target * mask_strength
                    )
            final_state = state

        # Restore original update_scale
        config['update_scale'] = original_update_scale

        # Apply threshold for binarization
        structure = (final_state[0, config['ch_structure']] > request.threshold).float().numpy()
        existing = final_state[0, config['ch_existing']].numpy()
        access = final_state[0, config['ch_access']].numpy()
        anchors = final_state[0, config['ch_anchors']].numpy()
        
        return {
            "structure": structure.tolist(),
            "existing": existing.tolist(),
            "access": access.tolist(),
            "anchors": anchors.tolist(),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
