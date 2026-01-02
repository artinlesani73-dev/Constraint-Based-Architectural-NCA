# Technical Specification

## Constraint-Based Architectural NCA

**Version:** 2.0
**Status:** Deployed
**Date:** January 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Structures](#3-data-structures)
4. [Loss Functions](#4-loss-functions)
5. [Training Pipeline](#5-training-pipeline)
6. [Web Deployment](#6-web-deployment)
7. [Fine-tuning](#7-fine-tuning)
8. [Evaluation Results](#8-evaluation-results)

---

## 1. Overview

### 1.1 System Status

The Constraint-Based Architectural NCA system is fully implemented and deployed. The trained model (v3.1) successfully generates volumetric pavilion structures satisfying multiple architectural constraints.

### 1.2 Components

| Component | Status | Location |
|-----------|--------|----------|
| NCA Model | Deployed | `deploy/model_utils.py` |
| Web Server | Deployed | `deploy/server.py` |
| Web UI | Deployed | `deploy/index.html` |
| Trained Weights | Available | `notebooks/sel/MODEL C*/v31_fixed_geometry.pth` |
| Fine-tuning | Available | `notebooks/NB_Finetune_Porosity_Colab.ipynb` |

### 1.3 Key Achievements

| Metric | Target | Achieved |
|--------|--------|----------|
| Coverage | >70% | ~75% |
| Spill | <25% | ~18% |
| Ground Openness | >80% | ~85% |
| Thickness Compliance | >90% | ~92% |
| Legality | 100% | 100% |

---

## 2. System Architecture

### 2.1 NCA Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UrbanPavilionNCA                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: State S [B, 8, D, H, W]                                │
│         ├── Frozen [4]: ground, existing, access, anchors       │
│         └── Grown [4]: structure, surface, alive, hidden        │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Perceive3D  │ -> │ UpdateNet   │ -> │ Apply Delta │         │
│  │ (Sobel)     │    │ (MLP 1x1x1) │    │ (Masked)    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  Output: Updated State S' [B, 8, D, H, W]                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Configuration (Trained Model)

```python
CONFIG = {
    # Grid
    'grid_size': 32,
    'n_channels': 8,
    'n_frozen': 4,
    'n_grown': 4,

    # Channel indices
    'ch_ground': 0,
    'ch_existing': 1,
    'ch_access': 2,
    'ch_anchors': 3,
    'ch_structure': 4,
    'ch_surface': 5,
    'ch_alive': 6,
    'ch_hidden': 7,

    # Network
    'hidden_dim': 96,
    'update_scale': 0.1,

    # Corridor
    'corridor_width': 1,
    'vertical_envelope': 1,
    'corridor_seed_scale': 0.15,
    'street_levels': 2,

    # Constraints
    'max_thickness': 2,
    'max_facade_contact': 0.15,
}
```

### 2.3 Channel Specification

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | ground | Frozen | Ground plane (z=0) |
| 1 | existing | Frozen | Existing building volumes |
| 2 | access | Frozen | Access point locations |
| 3 | anchors | Frozen | Legal ground contact zones |
| 4 | structure | Grown | **Output: Generated pavilion** |
| 5 | surface | Grown | Surface detection |
| 6 | alive | Grown | Cell aliveness mask |
| 7 | hidden | Grown | Hidden state for temporal memory |

---

## 3. Data Structures

### 3.1 State Tensor

```python
# Shape: [B, C, D, H, W]
# B = batch size
# C = 8 channels
# D = depth (Z axis, vertical)
# H = height (Y axis)
# W = width (X axis)

state = torch.zeros(1, 8, 32, 32, 32)
```

### 3.2 Scene Parameters

```python
# Building definition
building = {
    'x': [start, end],      # X range (0-32)
    'y': [start, end],      # Y range (0-32)
    'z': [start, end],      # Z range (0-32)
    'side': 'left'|'right', # Which side of gap
    'gap_facing_x': int,    # X coordinate facing the gap
}

# Access point definition
access_point = {
    'x': int,               # X position
    'y': int,               # Y position
    'z': int,               # Z position (0 = ground)
    'type': 'ground'|'elevated'
}
```

---

## 4. Loss Functions

### 4.1 Trained Model Weights (v3.1)

```python
weights = {
    'legality': 30.0,       # No structure in buildings
    'coverage': 25.0,       # Fill corridor zone
    'spill': 25.0,          # Don't grow outside corridor
    'ground': 35.0,         # Keep ground open
    'thickness': 30.0,      # No bulky masses
    'sparsity': 30.0,       # Limit total volume
    'facade': 10.0,         # Limit building contact
    'access_conn': 15.0,    # Connect access points
    'loadpath': 8.0,        # Structural support
    'cantilever': 5.0,      # Limit overhangs
    'density': 3.0,         # Overall density
    'tv': 1.0,              # Smoothness
}
```

### 4.2 Core Loss Functions

#### LocalLegalityLoss
Prevents structure inside existing buildings.

#### CorridorCoverageLoss
Encourages filling the corridor zone between access points.

#### CorridorSpillLoss
Penalizes structure outside the corridor zone.

#### GroundOpennessLoss
Keeps street level open for pedestrian circulation.

#### ThicknessLoss
Penalizes bulky, blob-like structures.

#### SparsityLoss
Limits total fill ratio to 5-15%.

---

## 5. Training Pipeline

### 5.1 Training History

| Version | Approach | Outcome |
|---------|----------|---------|
| v1.x | Incremental constraints | Failed - degenerate solutions |
| v2.x | All constraints, fixed corridors | Partial success |
| **v3.1** | **All constraints, adaptive corridors** | **Success** |

### 5.2 Final Training Configuration

```python
# v3.1 training settings
epochs = 2000
batch_size = 4
lr_initial = 0.001
steps_range = (40, 60)
grad_clip = 1.0

# Corridor masking (early epochs)
corridor_mask_epochs = 100
corridor_mask_anneal = 50
```

### 5.3 Training Curriculum

All constraints active from epoch 0. Curriculum varies scene complexity:

| Phase | Gap Width | Buildings | Access Points |
|-------|-----------|-----------|---------------|
| Easy | 14-18 | 2 | 1-2 |
| Medium | 10-14 | 2 | 2-3 |
| Hard | 6-10 | 2-3 | 3-4 |

---

## 6. Web Deployment

### 6.1 Server Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Browser       │────>│   FastAPI       │────>│   PyTorch       │
│   (Three.js)    │<────│   Server        │<────│   Model         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 6.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web interface |
| `/preview` | POST | Preview scene (no generation) |
| `/generate` | POST | Run NCA generation |

### 6.3 Generation Request

```python
{
    "buildings": [...],
    "access_points": [...],
    "steps": 50,
    "seed": -1,           # -1 = random
    "noise_std": 0.02,
    "corridor_seed_scale": 0.005,
    "fire_rate": 1.0,
    "corridor_width": 1,
    "vertical_envelope": 1,
    "threshold": 0.5,
    "update_scale": 0.1
}
```

### 6.4 UI Features

- **Collapsible panels** for parameter organization
- **Interactive building controls** with size sliders
- **Visibility toggles** for each voxel type
- **Real-time preview** of scene configuration
- **Parameter hints** explaining each control

---

## 7. Fine-tuning

### 7.1 Porosity Fine-tuning

For lighter, more porous structures:

```python
# Modified weights
weights['thickness'] = 50.0   # Increased from 30
weights['sparsity'] = 45.0    # Increased from 30
weights['density'] = 8.0      # Increased from 3
weights['tv'] = 0.5           # Reduced from 1

# New losses
weights['porosity'] = 20.0    # Target 25% internal voids
weights['surface'] = 10.0     # Higher surface-to-volume ratio

# Modified constraints
max_thickness = 1             # Reduced from 2
```

### 7.2 Fine-tuning Resources

| Resource | Location |
|----------|----------|
| Plan document | `docs/FINETUNE_POROSITY_PLAN.md` |
| Colab notebook | `notebooks/NB_Finetune_Porosity_Colab.ipynb` |
| Local notebook | `notebooks/NB_Finetune_Porosity.ipynb` |
| CLI script | `scripts/finetune_porosity.py` |

---

## 8. Evaluation Results

### 8.1 Final Model Performance (v3.1)

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| Coverage | 82% | 76% | 68% |
| Spill | 12% | 18% | 24% |
| Ground Open | 91% | 86% | 79% |
| Thickness | 95% | 93% | 89% |
| Valid Rate | 89% | 78% | 65% |

### 8.2 Generalization

The model successfully generalizes to:
- Novel building configurations
- Varying gap widths
- Different access point arrangements
- Multiple buildings (2-4)

### 8.3 Limitations

- Performance degrades on very narrow gaps (<6 voxels)
- Complex multi-building scenes may produce suboptimal coverage
- Fine details limited by 32³ resolution

---

## 9. File Structure

```
Constraint-Based-Architectural-NCA/
├── deploy/
│   ├── server.py           # FastAPI backend
│   ├── model_utils.py      # NCA model & utilities
│   ├── index.html          # Web interface
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── requirements.txt
│
├── notebooks/
│   ├── sel/MODEL C*/       # Trained model
│   │   ├── v31_fixed_geometry.pth
│   │   └── config_step_b.json
│   ├── NB_Finetune_Porosity.ipynb
│   └── NB_Finetune_Porosity_Colab.ipynb
│
├── scripts/
│   └── finetune_porosity.py
│
├── docs/
│   └── FINETUNE_POROSITY_PLAN.md
│
└── assets/
    └── sample_output.png
```

---

*Technical Specification v2.0 - Deployed System*

*January 2025*
