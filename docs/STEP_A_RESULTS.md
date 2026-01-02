# Step A: Experimental Results and Failure Analysis

**Date:** December 2025
**Status:** Concluded
**Outcome:** Failure - Fundamental approach requires revision

---

## Executive Summary

Step A attempted to train an NCA using curriculum learning with constraints introduced incrementally across phases. The approach failed to produce architecturally valid results due to fundamental representation errors and flawed constraint formulation.

**Core Finding:** The system learned to satisfy mathematical objectives while violating architectural intent. Without all constraints present, the NCA exploits gaps in the loss landscape to produce degenerate solutions (ground carpets, solid blobs).

---

## What Was Attempted

### Curriculum Phases (Planned)

| Phase | Focus | Constraints |
|-------|-------|-------------|
| 1 | Core Structural | Connectivity, Cantilever, Sparsity, Access Reach |
| 2 | + Ground Openness | + Ground Void |
| 3 | + Massing | + Facade, Distribution |
| 4 | + Path Quality | + Path Connectivity |
| 5 | Integration | All constraints, harder scenes |

### Actual Progress

Only Phase 1 was completed. Results were architecturally unacceptable:
- Solid blobs instead of porous pavilion structures
- Ground completely occupied (no street circulation)
- "Walkable" surfaces appeared as scattered islands on blob surfaces
- Structure satisfied connectivity by spreading across the floor

---

## Failure Analysis

### 1. Fundamental Representation Error: Walkable

**The Misconception:**
```
Implemented: walkable = solid voxel channel grown on structure
Intended:    walkable = void space that must remain empty for circulation
```

This semantic inversion meant the system could never learn to protect street space. It was literally incapable of representing "street as void."

**Correct Understanding:**
- Walkable space = EMPTY volume near ground where people circulate
- It is the ABSENCE of structure, not a material
- Only exceptions: structural anchors and access point thickening

### 2. Ground Was Unprotected

No loss penalized structure touching the ground plane. The network treated the street as free real estate because:
- Connectivity rewards ground contact (easiest path to support)
- No penalty for blocking circulation
- Ground void was deferred to Phase 2

**Result:** The model learned ground-carpet solutions that are structurally valid but architecturally meaningless.

### 3. Curriculum Approach Created Degenerate Foundations

Training constraints separately allowed the model to find local optima that satisfy early constraints but are incompatible with later ones:

| Phase 1 Learns | Phase 2+ Needs |
|----------------|----------------|
| Spread across ground for connectivity | Keep ground empty |
| Fill available space up to 25% | Sparse, elevated structures |
| Grow walkable surfaces on mass | Protect void for walkable space |

The Phase 1 behavior is the opposite of what the full system needs.

### 4. Quality Losses Destroyed Cellular Behavior

`DensityPenalty` + `TotalVariation` pushed the NCA toward:
- Binary solid/void outputs
- Smooth, minimal surface area shapes
- Topology-optimized blobs

This eliminated the expected porous, cellular morphology of NCA growth.

### 5. Metrics Did Not Measure Architectural Quality

| Metric | What It Measured | What Was Needed |
|--------|------------------|-----------------|
| Fill ratio | Total volume | Volumetric distribution |
| Walkable coverage | Surface on structure | Street void preservation |
| Connectivity | Connected to support | Elegant elevated paths |
| Access reach | Near access points | Accessible from access points |

The system optimized for math, not architecture.

---

## Technical Bugs Encountered

### Bug 1: Perception Boundary Artifacts (x=0 Bias)

**Symptom:** Structure consistently grew heavier toward x=0 boundary.

**Cause:** Zero-padding in Sobel convolutions created systematic gradient artifacts:
- At x=0: gradient = +0.5 (from zero-padded neighbor)
- At x=31: gradient = -0.5 (from zero-padded neighbor)
- MLP learned to correlate positive gradients with growth

**Fix Applied:** Replicate padding instead of zero padding:
```python
x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
```

### Bug 2: Biased Scene Generation

**Symptom:** Access points clustered in specific regions.

**Cause:** Access point y-coordinates limited to `[0, G//3]` range.

**Fix Applied:** Full randomization across valid locations (facades facing gap, ground in gap).

### Bug 3: Axis Label Confusion

**Symptom:** Difficulty interpreting 3D visualizations.

**Cause:** Tensor dimensions [Z, Y, X] transposed for matplotlib without updating labels.

**Fix Applied:** Correct axis labels reflecting actual dimensions:
- X-axis (plot) = Y (depth)
- Y-axis (plot) = X (left-right)
- Z-axis (plot) = Z (height)

### Bug 4: Walkable Bias Misconfiguration

**Symptom:** Walkable channel not developing properly.

**Cause:** `walkable_bias` set to -0.5 (discouraging growth) instead of 0.1.

**Fix Applied:** Corrected bias in config.json.

### Bug 5: Structure Growing Inside Buildings

**Symptom:** Generated structure overlapping with existing buildings.

**Cause:** No exclusion mechanism in NCA update.

**Fix Applied:**
1. Added ExclusionLoss to penalize overlap
2. Added hard mask in NCA forward pass

### Bug 6: Solid Blob Outputs

**Symptom:** Fuzzy, blobby structures with intermediate voxel values.

**Cause:** No incentive for binary (0 or 1) outputs.

**Fix Applied:** Added DensityPenalty (SIMP) and TotalVariation losses.

**Note:** This fix may have contributed to the topology-optimization behavior identified by the reviewer.

---

## Conceptual Errors

### Error 1: Walkable as Material

Modeled walkable as a grown channel (solid) instead of protected void (empty).

### Error 2: Deferring Ground Protection

Assumed Phase 2 could fix ground occupation learned in Phase 1. This underestimated the difficulty of unlearning degenerate solutions.

### Error 3: Separate Constraint Training

Assumed constraints could be learned incrementally. In practice, the absence of later constraints allows exploitation of the loss landscape.

### Error 4: Insufficient Anchor Control

No budget limiting how much structure can touch ground. Connectivity loss encouraged maximum ground contact.

### Error 5: Quality Over Form

DensityPenalty and TotalVariation prioritized "clean" outputs over architectural validity.

---

## Versions Developed

| Notebook | Version | Key Changes |
|----------|---------|-------------|
| NB01_Foundation | v1.0 → v1.2 | Initial → Replicate padding, randomized access, axis labels |
| NB02_Phase1_Structural | v1.0 → v1.6 | Initial → ExclusionLoss, DensityPenalty, TV, padding fix |

---

## Key Lessons for Step B

1. **All constraints must be present from the start** - Curriculum should vary complexity/difficulty, not constraint presence

2. **Walkable = void, not solid** - Must be represented as protected empty space

3. **Ground is precious** - Touching ground must be expensive and budgeted

4. **Anchors need explicit control** - Define where and how much ground contact is allowed

5. **Quality losses need balance** - Binary outputs are good, but not at the cost of porosity

6. **Metrics must match intent** - Measure architectural validity, not just mathematical satisfaction

7. **Test with all constraints early** - Don't assume phases can be developed in isolation

---

## Files Produced

```
notebooks/
├── NB01_Foundation_v1.2.ipynb
├── NB02_Phase1_Structural_v1.6.ipynb
docs/
├── STEP_A_RESULTS.md (this file)
```

---

## Conclusion

Step A demonstrated that:

1. NCAs can satisfy mathematical constraints while producing architecturally invalid results
2. Curriculum learning with incremental constraints creates degenerate foundations
3. The representation of "walkable" as solid material was fundamentally wrong
4. Ground protection must be present from the beginning, not added later

Step B must address these issues with a revised approach where all constraints are present from epoch 0, with curriculum applied to scene complexity rather than constraint activation.

---

*Step A Concluded - December 2025*
