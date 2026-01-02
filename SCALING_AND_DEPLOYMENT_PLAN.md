# Constraint-Based Architectural NCA: Scaling & Deployment Status

## Status: Complete

**Version:** 3.1
**Date:** December 2025

This document describes the completed scaling and deployment work for the Constraint-Based Architectural NCA project.

---

## Completed Phases

### Phase 0: v3.1 Training (Complete)

The model was successfully trained and validated on the easy difficulty setting.

**Target Metrics:**
| Metric | Target |
|--------|--------|
| Coverage | >70% |
| Spill | <20% |
| Ground Openness | >80% |
| Thickness | >90% |
| Fill Ratio | 3-12% |
| Legality | 100% |

**Training Configuration:**
- Epochs: 500
- Batch size: 4
- Learning rate: 7e-4 with decay
- Corridor seed scale: 0.15
- Steps range: 30-50

### Phase 1: Difficulty Curriculum (Partial)

The model is evaluated across difficulty levels in the v3.1 notebook. See `notebooks/model_c/v31_evaluation.json` for the latest metrics.

**Observations:**
- Performance degrades gracefully on harder scenes
- Model handles novel building configurations
- Narrow gaps (<6 voxels) remain challenging

### Phase 2: Resolution (Not Pursued)

The 32^3 resolution was retained for the deployed system. Higher resolution was not implemented as it was not necessary for the proof-of-concept goals.

**Rationale:**
- 32^3 provides sufficient detail for volumetric massing exploration
- Training time and memory requirements remain manageable
- Proof of concept successfully demonstrated

### Phase 3: Web Application (Complete)

The local web application is fully deployed and functional.

**Implementation:**

| Component | Technology | Location |
|-----------|------------|----------|
| Backend | FastAPI + PyTorch | `deploy/server.py` |
| Frontend | Three.js + HTML/CSS | `deploy/index.html` |
| Model Utils | Python | `deploy/model_utils.py` |
| Styles | CSS | `deploy/static/css/style.css` |
| App Logic | JavaScript | `deploy/static/js/app.js` |

**Features Implemented:**
- Add/remove/resize buildings interactively
- Add/remove access points (ground/elevated)
- Collapsible parameter panels
- Real-time preview
- Visibility toggles for voxel types
- Seed control (random or fixed)
- Adjustable generation parameters:
  - Steps (10-100)
  - Noise standard deviation
  - Fire rate
  - Corridor seed scale (0-0.01)
  - Corridor width
  - Vertical envelope
  - Threshold
  - Update scale
- 3D visualization with orbit controls
- Export capability (screenshot)

**Performance:**
- Generation time: <1s on CPU for 50 steps at 32^3
- GPU acceleration supported when available

---

## How to Run

```bash
cd Constraint-Based-Architectural-NCA
pip install -r deploy/requirements.txt
python deploy/server.py
# Open http://localhost:8000
```

---

## Future Extensions (Not Implemented)

These remain as potential future work:

1. **Higher Resolution (64^3, 128^3)**
   - Would require re-tuning of all loss parameters
   - Longer training times and higher memory requirements

2. **Multi-Building Complex Scenes**
   - Support for 4+ buildings
   - Irregular footprints

3. **Export Formats**
   - OBJ/STL mesh export
   - IFC for BIM integration

4. **Real Site Import**
   - GIS/OSM urban context
   - Terrain integration

5. **Multi-Objective Optimization UI**
   - Interactive constraint weight adjustment
   - Real-time loss visualization

---

## Lessons Learned

1. **Start simple** - Easy difficulty training provides stable foundation
2. **Fixed validation sets** - Essential for reliable evaluation
3. **Curriculum on scenes, not constraints** - All constraints from epoch 0
4. **Interactive deployment** - Enables exploration and validation
5. **Parameter exposure** - Users benefit from adjustable generation settings

---

*Scaling & Deployment Status v3.1 - Project Complete*

*December 2025*
