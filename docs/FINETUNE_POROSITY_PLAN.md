# Fine-Tuning Plan: Lightweight & Porous Structures

## Goal
Fine-tune the trained v3.1 model to produce **more porous, lightweight** architectural structures while maintaining structural validity and corridor coverage.

---

## Current State Analysis

### Existing Loss Weights (v3.1 MODEL C)
| Loss | Weight | Purpose |
|------|--------|---------|
| legality | 30.0 | Prevent illegal placements |
| coverage | 25.0 | Fill corridor zones |
| spill | 25.0 | Don't grow outside corridors |
| ground | 35.0 | Keep ground level open |
| **thickness** | 30.0 | Penalize bulky cores |
| **sparsity** | 30.0 | Penalize dense fills |
| facade | 10.0 | Limit building contact |
| access_conn | 15.0 | Connect access points |
| loadpath | 8.0 | Structural load paths |
| cantilever | 5.0 | Limit overhangs |
| density | 3.0 | Overall density penalty |
| tv | 1.0 | Smoothness regularization |

### Current Constraints
- `max_thickness = 2` (max voxels before penalized)
- `fill_ratio_target = 0.05 - 0.15` (5-15% fill)
- No explicit porosity/void encouragement

---

## Fine-Tuning Strategy

### Phase 1: Weight Rebalancing
Increase penalties for bulk/density without breaking coverage:

| Loss | Current | New | Rationale |
|------|---------|-----|-----------|
| thickness | 30.0 | **50.0** | Stronger penalty for bulk |
| sparsity | 30.0 | **45.0** | More aggressive sparsity |
| density | 3.0 | **8.0** | Higher overall density penalty |
| tv | 1.0 | **0.5** | Reduce smoothness (allow holes) |

### Phase 2: Add Porosity Loss (New)
Introduce explicit void encouragement within corridor volumes:

```python
class PorosityLoss(nn.Module):
    """
    Encourage internal voids within the structure.
    Penalizes structures where interior voxels are surrounded by structure.
    """
    def __init__(self, target_porosity: float = 0.3):
        super().__init__()
        self.target_porosity = target_porosity  # 30% internal voids

    def forward(self, structure: torch.Tensor, corridor: torch.Tensor) -> torch.Tensor:
        # Only consider voxels inside corridor
        in_corridor = corridor > 0.5
        struct_in_corridor = structure * corridor

        # Compute "interior" by erosion
        eroded = -F.max_pool3d(-struct_in_corridor.unsqueeze(0).unsqueeze(0), 3, 1, 1)
        interior = eroded.squeeze() > 0.5

        # Porosity = 1 - (filled interior / total interior)
        interior_filled = (structure * interior.float()).sum()
        interior_total = interior.float().sum() + 1e-8

        porosity = 1.0 - (interior_filled / interior_total)

        # Penalize if porosity below target
        return F.relu(self.target_porosity - porosity)
```

### Phase 3: Reduce Max Thickness
```python
# Current
max_thickness = 2

# Fine-tuned
max_thickness = 1  # Force thinner walls
```

### Phase 4: Adjust Fill Ratio Target
```python
# Current
fill_ratio_target = (0.05, 0.15)

# Fine-tuned
fill_ratio_target = (0.03, 0.10)  # Lower target density
```

---

## Training Protocol

### Step 1: Load Pretrained Checkpoint
```python
checkpoint_path = "notebooks/sel/MODEL C - Copy - no change/v31_fixed_geometry.pth"
```

### Step 2: Modify Weights
Apply Phase 1 weight changes.

### Step 3: Add Porosity Loss
Add Phase 2 porosity loss with weight = 20.0.

### Step 4: Short Fine-Tune
- **Epochs**: 100-200 (not full training)
- **Learning Rate**: 0.0005 (half of original)
- **Batch Size**: 4
- **Validation**: Every 10 epochs on fixed scenes

### Step 5: Evaluate
Check metrics on validation set:
- Coverage > 60% (slightly relaxed)
- Spill < 25%
- Ground openness > 80%
- **Thickness > 95%** (stricter)
- **Fill ratio 3-10%** (lower)
- **Visual porosity check**

---

## Expected Outcomes

### Before Fine-Tuning
- Solid, continuous structures
- Fill ratio ~10-15%
- Minimal internal voids

### After Fine-Tuning
- Lattice-like, porous structures
- Fill ratio ~5-8%
- Visible internal voids and perforations
- Thinner structural members

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Structure becomes too sparse | Cap sparsity weight at 50, monitor coverage |
| Load paths break | Keep loadpath weight at 8.0 or increase |
| Disconnected fragments | Monitor access_conn loss |
| Training collapse | Use lower learning rate, early stopping |

---

## Files to Create

1. `notebooks/NB_Finetune_Porosity.ipynb` - Fine-tuning notebook
2. `deploy/models/v31_porous.pth` - Fine-tuned checkpoint (output)

---

## Quick Start

### Option 1: Google Colab (Recommended)

1. **Upload to Colab**: Upload `notebooks/NB_Finetune_Porosity_Colab.ipynb` to Google Colab

2. **Upload Project Files**: Either:
   - **Option A**: Upload entire project folder to Google Drive
   - **Option B**: Zip and upload, then unzip in Colab

3. **Set Path**: Edit this cell to match your folder location:
   ```python
   PROJECT_PATH = '/content/drive/MyDrive/Constraint-Based-Architectural-NCA'
   ```

4. **Run All Cells**: The notebook will:
   - Check GPU availability (T4/V100 recommended)
   - Load pretrained model
   - Fine-tune for ~150 epochs
   - Save checkpoints
   - Let you download the fine-tuned model

#### Expected Training Time (Colab)

| GPU | ~150 epochs |
|-----|-------------|
| T4 | ~30-45 min |
| V100 | ~15-20 min |
| CPU | ~3-4 hours |

### Option 2: Local Jupyter

```bash
jupyter notebook notebooks/NB_Finetune_Porosity.ipynb
```

### Option 3: Python Script

```bash
python scripts/finetune_porosity.py --epochs 150 --lr 0.0005
```

Script arguments:
- `--epochs 150` - Number of training epochs
- `--lr 0.0005` - Learning rate
- `--batch-size 4` - Batch size
- `--eval-every 10` - Evaluate every N epochs
- `--output-dir notebooks/finetuned_porous` - Output directory

---

## Validation Scenes

Use the same validation scenes as v3.1 for fair comparison:
- 2 buildings, 2 access points (easy)
- Varying gap widths
- Ground + elevated access combinations

---

*Plan created for fine-tuning MODEL C v3.1 toward porous, lightweight outputs.*
