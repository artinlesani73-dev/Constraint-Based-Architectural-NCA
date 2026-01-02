# Constraint Specification Sheets

## Detailed Specifications for Each Architectural Constraint

**Version:** 3.1 (Fixed Loss Geometry)
**Date:** December 2025
**Status:** Reference Document (Experiment Snapshot)

> **Note:** This document captures the constraint development journey, including the evolution from v3.0 through v3.1. The v3.1 experiment uses the constraint formulations described in Part 5 (Fixed Loss Geometry) and matches the notebook `notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb`. For a summary of the deployed system, see `SPECIFICATION.md`.

---

## Revision History

| Version | Changes |
|---------|---------|
| 1.0 | Initial specification with curriculum phases |
| 2.0 | Step A failure analysis - walkable=void, all constraints from epoch 0 |
| 2.1 | Evidence-based revision linking changes to experimental observations |
| 3.0 | Ontology revision - local losses, causal encoding, aligned incentives |
| 4.0 | Architectural intent - path corridors, thickness limits, facade budgets |
| **3.1** | **Fixed loss geometry - Coverage+Spill split, ground openness, tighter thickness** |

---

## Part 1: Why v3.0 Failed (Intent Problem)

### The New Degenerate Attractor

v3.0 successfully eliminated ontology failures. The model respects legality (100%), maintains load paths (99%), and preserves access connectivity (100%).

However, a new attractor emerged:
- Facade-coating behavior
- Large connected blobs
- Low growth alignment (~15-30%)
- Excessive fill ratio (~25-30%)

### Root Cause: Missing Architectural Intent

| Problem | v3.0 Formulation | Why It Failed |
|---------|------------------|---------------|
| Load path encourages blobs | Only requires path existence | No penalty for thickness or redundancy |
| Growth not spatially bounded | Rewards target overlap | No penalty for mass OUTSIDE target |
| Point-based targets | Dilated access points | Creates bulbs, not corridors |
| Facades are free support | No contact cost | Infinite support manifold exploitation |
| Global fill ratio | Total volume constraint | Can fill anywhere convenient |

### The Key Insight

**v3.0 defined what is FORBIDDEN but not what is PREFERRED.**

The system knows where it cannot build, but not:
- Where it SHOULD build (path corridors)
- What SHAPE is preferred (thin, elongated)
- What to AVOID (blobs, facade coating)

---

## Part 2: Architectural Intent Definition

### What is a Pavilion?

A pavilion is a **connecting structure** between access points where you:
- Enter from one access point
- Exit from others
- Experience varied spaces along the path

### Architectural Requirements

1. **Connection** - Links access points through continuous paths
2. **Not blobs** - Thin, elongated volumes, not massive lumps
3. **Limited facade contact** - Does not coat entire building surfaces
4. **Ground preservation** - Leaves street level open except where necessary
5. **Spatial variety** - Compression (narrow) and decompression (wider) zones
6. **Purposeful mass** - Every voxel serves the connectivity goal

---

## Part 3: The v4.0 Architecture

### Core Principle

**Every loss must encode ARCHITECTURAL INTENT, not just constraint satisfaction.**

- Constraints define boundaries (where you CAN'T build)
- Intent defines goals (where you SHOULD build and what shape)

### Constraint Architecture

| Category | Loss | Formulation | Purpose |
|----------|------|-------------|---------|
| **Legality** | LocalLegality | Per-voxel field | Where you CAN'T build |
| **Intent** | PathConnectivity | Path-based corridors | Where you SHOULD build |
| **Intent** | OutsideTarget | Explicit penalty | Penalize off-path mass |
| **Shape** | Thickness | Erosion-based | Prevent blobs |
| **Shape** | FacadeContact | Contact budget | Limit facade coating |
| **Circulation** | AccessConnectivity | Access-seeded flood | Functional circulation |
| **Structure** | LoadPath | Path tracing | Structural validity |
| **Structure** | Cantilever | Local overhang | Local support |
| **Massing** | Sparsity | Volume bounds | Total volume |
| **Quality** | Density + TV | Binary + smooth | Output quality |

---

## Constraint 1: Local Legality (Unchanged from v3.0)

### Definition

Every voxel has a binary legality value. Structure in illegal voxels is directly penalized.

### Legality Field Rules

```
Legality(p) = 1 (legal) if:
  - p is not inside existing building, AND
  - (p.z >= street_levels) OR (p is in anchor zone)

Legality(p) = 0 (illegal) if:
  - p is inside existing building, OR
  - (p.z < street_levels) AND (p is not in anchor zone)
```

### Formula

```
legality_field = compute_legality(state)
illegal_structure = structure * (1 - legality_field)
L_legality = illegal_structure.sum() / (structure.sum() + epsilon)
```

### Weight: 30.0

---

## Constraint 2: Path Connectivity (NEW - Replaces AlignedGrowth)

### Definition

Growth target is defined by **shortest path corridors** between access points, not dilated spheres.

**Replaces:** AlignedGrowthLoss

### Why Path-Based Targets

| Point Dilation | Path Corridors |
|----------------|----------------|
| Creates spherical bulbs | Creates linear corridors |
| Encourages blob formation | Encourages bridge-like spans |
| No directionality | Connects specific endpoints |

### Formula

```
# Compute shortest paths between all access point pairs
access_points = find_access_locations(state)
paths = []
for i, j in pairs(access_points):
    path = shortest_path_3d(access_points[i], access_points[j], legal_space)
    paths.append(path)

# Create corridor target by dilating paths
corridor_target = zeros_like(structure)
for path in paths:
    corridor_target = max(corridor_target, dilate(path, radius=corridor_width))

# Reward structure IN corridor
in_corridor = (structure * corridor_target).sum()

# Dice toward corridor
intersection = (structure * corridor_target).sum()
dice = (2 * intersection + 1) / (structure.sum() + corridor_target.sum() + 1)

L_path = 1 - dice
```

### Corridor Width

- Base width: 3-4 voxels (allows passage)
- Can vary along path for compression/decompression effect

### Weight: 25.0

---

## Constraint 3: Outside Target Penalty (NEW)

### Definition

Explicitly penalize structure that exists OUTSIDE the path corridor target.

**This was the critical missing piece in v3.0.**

### Why This Matters

v3.0's AlignedGrowthLoss rewarded overlap with target but did NOT penalize mass elsewhere. The model satisfied alignment minimally, then expanded freely into safe zones.

### Formula

```
# Structure outside the corridor target
outside_corridor = structure * (1 - corridor_target) * legality_field

# Penalty = fraction of structure that is off-path
L_outside = outside_corridor.sum() / (structure.sum() + epsilon)
```

### Weight: 20.0

---

## Constraint 4: Thickness Limit (NEW)

### Definition

Penalize voxels that are "deep" inside structure (far from surface) to prevent blob formation.

### Architectural Motivation

Pavilions should have varied but bounded thickness:
- Thin passages (compression)
- Wider spaces (decompression)
- NOT solid blobs with deep interiors

### Formula

```
# Iteratively erode structure
# Voxels surviving N erosions are "core" (too thick)
core = structure
for i in range(max_thickness):
    core = erode_3d(core)

# Penalize remaining core
L_thickness = core.sum() / (structure.sum() + epsilon)
```

### Erosion Operation

```
erode_3d(x) = 1 - max_pool_3d(1 - x, kernel=3, padding=1)
```

### Max Thickness Parameter

- `max_thickness = 3-4` voxels allows corridors and small rooms
- Anything thicker is penalized as blob-like

### Weight: 15.0

---

## Constraint 5: Facade Contact Budget (NEW)

### Definition

Limit the amount of structure that directly touches existing building surfaces.

### Architectural Motivation

Facades provide free structural support, creating an incentive to coat them. A pavilion should:
- Touch buildings at connection points
- NOT coat entire facades
- Have some free-standing portions

### Formula

```
# Identify facade zone (voxels adjacent to buildings)
facade_zone = dilate(existing, radius=1) - existing

# Structure touching facades
contact = (structure * facade_zone).sum()
total_structure = structure.sum() + epsilon

# Contact ratio
contact_ratio = contact / total_structure

# Penalize if exceeding budget
L_facade = ReLU(contact_ratio - max_contact_ratio)
```

### Contact Budget

- `max_contact_ratio = 0.15-0.20` (15-20% of structure can touch facades)
- Forces some free-standing construction

### Weight: 10.0

---

## Constraint 6: Access Connectivity (Unchanged from v3.0)

### Definition

Void at street level must allow circulation BETWEEN access points.

### Formula

```
void_mask = (1 - structure) * (1 - existing) at z < street_levels
seed = dilate(access_points, radius=1) at z < street_levels
connected = flood_fill(void_mask, seed)

access_locations = access_points > 0.5 at z < street_levels
reachable = (connected * access_locations).sum()

L_access_conn = 1 - reachable / (access_locations.sum() + epsilon)
```

### Weight: 15.0

---

## Constraint 7: Load Path (Reduced Weight)

### Definition

Elevated structure must be connected to valid ground support through continuous structural paths.

### Note on Weight Reduction

LoadPath is still necessary for structural validity, but its weight is reduced because:
- It encourages blob formation (bigger = safer)
- PathConnectivity + Thickness now handle shape
- We want valid paths, not maximally robust paths

### Formula

```
support = existing OR (anchors at z < street_levels)
connected = flood_fill(structure, support)
elevated = structure at z >= street_levels
unsupported = elevated * (1 - connected)

L_loadpath = unsupported.sum() / (elevated.sum() + epsilon)
```

### Weight: 10.0 (reduced from 20.0)

---

## Constraint 8: Cantilever (Unchanged)

### Definition

Local horizontal overhangs limited to N voxels.

### Formula

```
For each voxel at height z:
  support_below = max(structure[z-N : z-1])
  has_support = dilate(support_below)

unsupported = structure * (1 - has_support) for z >= N

L_cantilever = unsupported.mean()
```

### Weight: 5.0

---

## Constraint 9: Sparsity (Unchanged)

### Definition

Total volume in 5-15% range.

### Formula

```
ratio = structure.sum() / available.sum()

L_sparsity = ReLU(ratio - 0.15) + 3.0 * ReLU(0.05 - ratio)
```

### Weight: 15.0

---

## Quality Losses (Unchanged)

### Density Penalty

```
L_density = mean(structure * (1 - structure))
```

Weight: 3.0

### Total Variation

```
L_tv = |grad(structure)|
```

Weight: 1.0

---

## Weight Summary (v4.0)

| Loss | Weight | Category | Change |
|------|--------|----------|--------|
| LocalLegality | 30.0 | Legality | Unchanged |
| PathConnectivity | 25.0 | Intent | NEW (replaces AlignedGrowth) |
| OutsideTarget | 20.0 | Intent | NEW |
| AccessConnectivity | 15.0 | Circulation | Unchanged |
| Thickness | 15.0 | Shape | NEW |
| Sparsity | 15.0 | Massing | Unchanged |
| FacadeContact | 10.0 | Shape | NEW |
| LoadPath | 10.0 | Structure | Reduced from 20.0 |
| Cantilever | 5.0 | Structure | Unchanged |
| Density | 3.0 | Quality | Unchanged |
| TV | 1.0 | Quality | Unchanged |
| **Total** | **149.0** | | |

---

## Comparison: v3.0 vs v4.0

| Aspect | v3.0 | v4.0 |
|--------|------|------|
| Growth target | Dilated access (bulbs) | Path corridors (linear) |
| Off-target mass | No penalty | Explicit penalty |
| Blob prevention | None | Thickness constraint |
| Facade exploitation | Unlimited | Contact budget |
| Architectural intent | Implicit | Explicit |
| Attractor type | Facade-coating blobs | Connecting corridors |

---

## Success Criteria (v4.0)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Legality | 100% | All structure in legal voxels |
| Path alignment | >70% | Structure in corridor target |
| Outside target | <20% | Structure off-path |
| Thickness compliance | >90% | No deep blob interiors |
| Facade contact | <20% | Limited facade touching |
| Access connectivity | >90% | Access points reachable |
| Load path | >95% | Elevated structure supported |
| Fill ratio | 5-15% | Total volume |

---

## Implementation Notes

### Path Computation

Shortest paths can be computed via:
1. **3D A*** - Exact but expensive
2. **3D Dijkstra** - Standard approach
3. **Differentiable approximation** - Soft path via iterative propagation

For training, use differentiable soft paths:
```
# Soft shortest path via iterative distance propagation
distance = infinity everywhere except access_points = 0
for iteration in range(max_iters):
    distance = min(distance, dilate(distance) + 1) * legal_mask

# Path = gradient descent on distance field
path = trace_gradient(distance, from=access_a, to=access_b)
```

### Erosion for Thickness

Use morphological erosion via min-pooling:
```
def erode_3d(x):
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
```

### Gradient Flow

All new losses are differentiable:
- Path targets computed from frozen channels
- Erosion uses soft thresholding
- Contact detection via differentiable dilation

---

## Part 4: Why v4.0 Failed (Loss Geometry Bug)

### Experimental Results

v4.0 (notebook v3.0) results:
- Legality: 100% PASS
- Path Align: 64% FAIL (target >70%)
- Outside: 36% FAIL (target <20%)
- Thickness: 69% FAIL (target >90%)
- Fill Ratio: 21% FAIL (target 5-15%)

### Root Cause: Duplicate Spill Losses

**Critical bug:** PathConnectivity and OutsideTarget were both "spill" losses.

| Loss | What it should do | What it actually did |
|------|-------------------|---------------------|
| PathConnectivity | Reward covering corridor (A) | Dice = weak spill penalty |
| OutsideTarget | Penalize outside corridor (B) | Explicit spill penalty |

**Result:** We had (B) + (B), not (A) + (B). No loss strongly rewarded filling the corridor.

### Additional Issues

| Problem | Consequence |
|---------|-------------|
| Corridor from current state | Unstable targets during training |
| max_thickness=4 | Too permissive for corridor_width=3 |
| Hinge sparsity | Not sharp enough beyond 15% |
| No ground openness | Street level filled unnecessarily |

---

## Part 5: The v3.1 Architecture (Fixed)

### Core Fix: Split Coverage and Spill

| Objective | Loss | Formula |
|-----------|------|---------|
| **(A) Coverage** | CorridorCoverage | `unfilled_corridor / corridor_target` |
| **(B) Spill** | CorridorSpill | `outside_corridor / structure` |

### All v3.1 Changes

| Change | v4.0 | v3.1 |
|--------|------|------|
| Intent losses | PathConnectivity + OutsideTarget | CorridorCoverage + CorridorSpill |
| Corridor source | Current state | Seed state (frozen) |
| max_thickness | 4 | 2 |
| corridor_width | 3 | 1 |
| Sparsity penalty | Hinge | Squared beyond threshold |
| Ground openness | None | Explicit GroundOpennessLoss |

### New Loss: CorridorCoverage

```
# (A) Coverage - penalize unfilled corridor
unfilled = corridor_target * (1 - structure)
L_coverage = unfilled.sum() / (corridor_target.sum() + eps)
```

This says: "fill the intended connector region."

### New Loss: CorridorSpill

```
# (B) Spill - penalize structure outside corridor
outside = structure * (1 - corridor_target) * legality_field
L_spill = outside.sum() / (structure.sum() + eps)
# Plus absolute term to prevent gaming via mass
L_spill += 0.01 * outside.sum() / (G^3)
```

This says: "whatever you build, keep it in the corridor."

### New Loss: GroundOpenness

```
# Ground structure outside corridor
ground_struct = structure[:, :street_levels]
ground_corridor = corridor_target[:, :street_levels]
unnecessary = ground_struct * (1 - ground_corridor)
L_ground = unnecessary.sum() / (ground_struct.sum() + eps)
# Extra cap on total ground mass (even inside corridor)
ground_mass = ground_struct.sum() / (ground_legal.sum() + eps)
L_ground += ReLU(ground_mass - ground_max_ratio)
```

This says: "don't block the street unless the corridor requires it."

### Improved: SparsityLoss with Squared Penalty

```
over = ReLU(ratio - max_ratio)
L_sparsity = over^2 * 100 + under_penalty
```

Squared penalty makes exceeding 15% much more costly.

### Improved: Corridor Computation

```
# Compute from SEED (frozen), not current state
# Add vertical envelope for elevated structures
corridor = compute_corridor_target(seed_state, vertical_envelope=1)
```

---

## Weight Summary (v3.1)

| Loss | Weight | Category | Change from v4.0 |
|------|--------|----------|------------------|
| LocalLegality | 30.0 | Legality | Unchanged |
| CorridorCoverage | 25.0 | Intent | NEW (replaces PathConnectivity) |
| CorridorSpill | 25.0 | Intent | Renamed from OutsideTarget |
| Thickness | 30.0 | Shape | Increased (was 15) |
| Sparsity | 30.0 | Massing | Increased (was 15) |
| GroundOpenness | 35.0 | Intent | NEW |
| AccessConnectivity | 15.0 | Circulation | Unchanged |
| FacadeContact | 10.0 | Shape | Unchanged |
| LoadPath | 8.0 | Structure | Reduced (was 10) |
| Cantilever | 5.0 | Structure | Unchanged |
| Density | 3.0 | Quality | Unchanged |
| TV | 1.0 | Quality | Unchanged |
| **Total** | **217.0** | | |

---

## Success Criteria (v3.1)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Legality | 100% | All structure in legal voxels |
| Coverage | >70% | Corridor target filled |
| Spill | <20% | Structure outside corridor |
| Ground openness | >80% | Ground structure in corridor |
| Thickness compliance | >90% | No deep blob interiors |
| Facade contact | <15% | Limited facade touching |
| Access connectivity | >90% | Access points reachable |
| Load path | >95% | Elevated structure supported |
| Fill ratio | 5-15% | Total volume |

---

## Configuration (v3.1)

```python
CONFIG = {
    'corridor_width': 1,         # Narrower for v3.1 experiment
    'max_thickness': 2,          # Stricter (was 4)
    'max_facade_contact': 0.15,  # Stricter (was 0.20)
    'vertical_envelope': 1,      # Reduced for v3.1 experiment
    'corridor_seed_scale': 0.15,
    'corridor_mask_epochs': 20,
    'corridor_mask_anneal': 40,
    'ground_max_ratio': 0.04,
    'street_levels': 6,
    'corridor_z_margin': 3,
}
```

---

*Constraint Specification v3.1 - Fixed Loss Geometry - December 2025*
