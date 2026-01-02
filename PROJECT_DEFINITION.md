# Project Definition

## Constraint-Based Architectural NCA: Neural Cellular Automata for Urban Pavilion Generation

**Version:** 2.0
**Status:** Completed & Deployed
**Date:** January 2025

---

## Executive Summary

This project demonstrates that Neural Cellular Automata (NCA) can serve as a generative architectural design tool. We successfully trained an NCA to create volumetric pavilion structures within urban settings while respecting architectural constraints including structural validity, corridor connectivity, ground-level openness, and massing distribution.

The system is now deployed as an interactive web application where users can configure urban contexts, adjust generation parameters, and produce constraint-satisfying architectural volumes in real-time.

---

## Project Status: Completed

| Milestone | Status |
|-----------|--------|
| Model Architecture | Complete |
| Constraint System | Complete |
| Training Pipeline | Complete |
| Trained Model (v3.1) | Complete |
| Web Deployment | Complete |
| Fine-tuning Support | Complete |

---

## What the System Does

The system takes as input:
- **Existing buildings** - Gray context volumes defining the urban setting
- **Access points** - Ground-level and elevated entry points to connect
- **Generation parameters** - Steps, noise, corridor settings

From this context, the NCA grows a volumetric pavilion structure (purple voxels) that:
- Connects all access points through walkable corridors
- Maintains open ground level for pedestrian circulation
- Avoids penetrating existing buildings
- Distributes mass efficiently (5-15% fill ratio)
- Remains structurally grounded (no floating elements)

---

## The Constraint Framework

The trained model satisfies five primary constraint categories:

### Constraint 1: Structural Legality
Generated structure must not penetrate existing buildings. Every voxel must be in legally available space.

### Constraint 2: Corridor Coverage
The structure must fill the corridor zone connecting access points, achieving >70% coverage of the computed corridor volume.

### Constraint 3: Spill Prevention
Structure should not grow excessively outside the corridor zone. Spill is limited to <25% of total structure volume.

### Constraint 4: Ground Openness
Street level must remain predominantly open for pedestrian movement, with >80% ground void outside anchor zones.

### Constraint 5: Thickness Control
No bulky masses allowed. Structure thickness is limited to prevent blob-like outputs, encouraging lighter, more articulated forms.

---

## Technical Implementation

### Model Architecture
- **Type:** Neural Cellular Automata (NCA)
- **Grid Size:** 32³ voxels
- **Channels:** 8 (4 frozen input, 4 grown output)
- **Perception:** 3D Sobel filters with replicate padding
- **Update Network:** 3-layer MLP as 1×1×1 convolutions
- **Hidden Dimension:** 96

### Training Approach
The final model (v3.1) was trained with all constraints active from epoch 0, using curriculum learning that varied scene complexity rather than constraint presence. Key innovations included:
- Corridor-based void protection
- Anchor zone system for controlled ground contact
- Multi-loss balancing with carefully tuned weights

### Deployment
- **Backend:** FastAPI server with PyTorch inference
- **Frontend:** Three.js-based 3D visualization
- **Interaction:** Real-time parameter adjustment and generation

---

## Research Questions: Answered

### RQ1: Can NCA learn to satisfy multiple architectural constraints simultaneously?
**Yes.** The trained model achieves >70% coverage, <25% spill, >80% ground openness, and >90% thickness compliance simultaneously.

### RQ2: Can NCA learn to protect void (absence) rather than just generate mass?
**Yes.** The corridor-based system successfully learns to keep ground level open while generating structure in appropriate zones.

### RQ3: Does the model generalize to novel building configurations?
**Yes.** The model handles various building placements, heights, and access point configurations not seen during training.

### RQ4: Can fine-tuning adjust aesthetic qualities while maintaining constraints?
**Yes.** The porosity fine-tuning approach demonstrates that aesthetic qualities (density, porosity) can be modified while preserving constraint satisfaction.

---

## System Capabilities

### Interactive Design
Users can:
- Add/remove/resize context buildings
- Place ground-level and elevated access points
- Adjust generation parameters (steps, noise, corridor settings)
- Toggle visibility of different voxel types
- Generate multiple variations with different seeds

### Fine-tuning
The model can be fine-tuned for different aesthetic goals:
- **Porosity:** Lighter, more lattice-like structures
- **Density:** More solid, continuous forms
- Custom constraint weight adjustments

---

## Outputs

The project delivers:

1. **Trained Model** - `v31_fixed_geometry.pth` checkpoint
2. **Web Application** - Interactive 3D design tool
3. **Fine-tuning Framework** - Notebooks and scripts for customization
4. **Documentation** - Complete technical specifications
5. **Research Findings** - Answers to core research questions

---

## Scope and Limitations

### What the System Does
- Volumetric massing at pavilion scale (10-30 meter spans)
- Constraint-satisfying form generation
- Interactive parameter exploration
- Real-time 3D visualization

### What the System Does Not Do
- Structural engineering analysis (load calculations, FEA)
- Detailed architectural elements (windows, doors, stairs)
- Construction documentation
- Building code compliance verification

### Resolution as Design Choice
The 32³ voxel resolution (~0.8m per voxel at 25m span) is intentional for volumetric concept generation. The output represents where mass should be distributed, not final construction geometry.

---

## Future Directions

Potential extensions include:
- Higher resolution (64³, 128³) for finer detail
- Additional constraint types (daylight, views, acoustic)
- Multi-objective optimization interface
- Export to architectural CAD formats
- Real-site urban context import

---

## Conclusion

This project successfully demonstrates Neural Cellular Automata as a viable generative architectural tool. The trained model produces constraint-satisfying pavilion structures, the web interface enables interactive exploration, and the fine-tuning framework allows aesthetic customization. The research questions have been answered affirmatively, establishing NCA as a promising approach for computational architectural design.

---

*Project Definition v2.0 - Completed System*

*January 2025*
