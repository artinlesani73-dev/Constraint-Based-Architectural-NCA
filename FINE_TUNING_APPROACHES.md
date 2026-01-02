# Fine-Tuning Approaches

## Status: Implemented

**Version:** 2.0
**Date:** January 2025

This document describes fine-tuning approaches for the Constraint-Based Architectural NCA. The porosity fine-tuning approach has been fully implemented with notebooks and scripts.

---

## Implemented: Porosity Fine-Tuning

A complete fine-tuning framework for achieving lighter, more porous structures has been implemented.

### Available Resources

| Resource | Location | Description |
|----------|----------|-------------|
| Plan Document | `docs/FINETUNE_POROSITY_PLAN.md` | Detailed strategy and implementation notes |
| Colab Notebook | `notebooks/NB_Finetune_Porosity_Colab.ipynb` | Google Colab-compatible (recommended) |
| Local Notebook | `notebooks/NB_Finetune_Porosity.ipynb` | Local Jupyter notebook |
| CLI Script | `scripts/finetune_porosity.py` | Command-line fine-tuning |

### How Porosity Fine-Tuning Works

**New Loss Functions:**

1. **PorosityLoss** - Encourages internal voids within structures
   ```python
   # Target 25% internal voids
   internal_void = (1 - structure) * dilate(structure)
   porosity = internal_void.sum() / (dilate(structure).sum() + eps)
   L_porosity = ReLU(target_porosity - porosity)
   ```

2. **SurfaceAreaLoss** - Rewards higher surface-to-volume ratio
   ```python
   surface = structure - erode(structure)
   ratio = surface.sum() / (structure.sum() + eps)
   L_surface = -log(ratio + eps)  # Higher ratio = lower loss
   ```

**Modified Weights:**

| Loss | Base Model | Porosity Fine-tune |
|------|------------|-------------------|
| Thickness | 30.0 | 50.0 |
| Sparsity | 30.0 | 45.0 |
| Density | 3.0 | 8.0 |
| TV | 1.0 | 0.5 |
| Porosity | - | 20.0 |
| Surface | - | 10.0 |

**Modified Constraints:**

| Parameter | Base Model | Porosity Fine-tune |
|-----------|------------|-------------------|
| max_thickness | 2 | 1 |

### Usage

**Google Colab (Recommended):**
1. Upload project to Google Drive
2. Open `notebooks/NB_Finetune_Porosity_Colab.ipynb`
3. Set `PROJECT_PATH` to your Drive location
4. Run all cells

**Local:**
```bash
python scripts/finetune_porosity.py --epochs 150 --lr 0.0005
```

---

## Other Fine-Tuning Approaches (Conceptual)

The following approaches remain conceptual options for future work:

### 1. Curriculum Fine-Tuning (Difficulty Ramp)

**Idea:** Continue training from best checkpoint with higher scene difficulty.

**When to use:**
- Generalization to medium/hard scenes needed

**How:**
- Easy -> Medium -> Hard progression
- Keep constraints unchanged; only vary scene complexity
- Optionally adjust corridor width per difficulty

### 2. Targeted Loss Rebalancing

**Idea:** Small weight adjustments to fix specific failures.

**When to use:**
- One or two metrics consistently fail while others pass

**How:**
- Increase only the failing constraint's weight
- Keep total weight range similar to baseline

### 3. Corridor Geometry Refinement

**Idea:** Adjust corridor computation parameters.

**When to use:**
- Coverage cannot exceed 70% due to corridor size issues

**How:**
- Monitor `corridor_fraction` (target 10-15%)
- Adjust width/envelope as needed

### 4. Density/Bulk Control

**Idea:** Encourage denser, more solid structures.

**When to use:**
- Structure is too fragmented or porous

**How:**
- Decrease thickness and sparsity weights
- Increase TV weight for smoother forms

### 5. Scale Fine-Tuning

**Idea:** Fine-tune for larger grids.

**When to use:**
- Higher resolution needed (48^3, 64^3)

**How:**
- Re-tune all loss parameters for new scale
- Significant compute increase required

---

## Practical Guidance

1. **Make one change at a time** - Evaluate on fixed validation scenes
2. **Prefer small adjustments** - Weight changes of +/-5
3. **Track corridor fraction** - Keep it at 10-15%
4. **Use fixed validation sets** - Avoid false positives from random scenes
5. **Save checkpoints frequently** - Enable rollback if changes destabilize

---

## Baseline Requirements

Before fine-tuning, ensure the base model:
- Passes on fixed validation set (easy difficulty)
- Maintains corridor fraction ~10-15%
- Shows consistent evaluation metrics

If baseline is unstable, prioritize stability over fine-tuning.

---

*Fine-Tuning Approaches v2.0 - Porosity Implemented, Others Conceptual*

*January 2025*
