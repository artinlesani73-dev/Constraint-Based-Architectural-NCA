# Step D: Constraint Specification

**Date:** December 2025
**Status:** Active
**Goal:** Generate NCA-grown volumes that connect 3 access points between two buildings

---

## Scene Definition

### Buildings
- **Count:** 2 existing buildings
- **Minimum height:** 20 meters each
- **Gap:** Minimum 18 meters between buildings

### Gap Zone Layout

For an 18m gap:
```
[Building A] | 3m Ped | 12m No-Go | 3m Ped | [Building B]
             |<------------- 18m gap ------------->|
```

| Zone | Width | Description |
|------|-------|-------------|
| Pedestrian Zone A | 3m | Adjacent to Building A, allows ground access |
| No-Go Zone | 12m | Center of gap, forbidden at ground level |
| Pedestrian Zone B | 3m | Adjacent to Building B, allows ground access |

**Formula for variable gaps:**
- Pedestrian zones: 3m each (fixed)
- No-Go zone: gap_width - 6m

---

## Access Points

### Configuration
| ID | Location | Height Constraint |
|----|----------|-------------------|
| AP1 | Building A facade (facing gap) | H1 (variable) |
| AP2 | Building B facade (facing gap) | H2 ≠ H1 |
| AP3 | Ground level | In pedestrian zones only (not no-go) |

### Placement Rules
- AP1 and AP2 must be at different heights
- AP1 and AP2 only on facades facing the gap
- AP3 must be within the 3m pedestrian zones, never in 12m no-go center

---

## Constraints

### C1: Building Exclusion (Hard)
**Rule:** NCA-grown volume cannot penetrate existing buildings.

```
L_exclusion = (structure * existing_buildings).sum()
Target: 0 (absolute)
```

### C2: Ground Zone Protection (0-5m)

#### C2A: Gap Clearance
**Rule:** The entire gap area (all 18m width) must remain empty from ground to 5m height.

**Exception:** Limited volume around ground access point (AP3).

```
ground_zone = gap_area[:5m, :, :]  # 0-5m height over entire gap
L_ground = (structure * ground_zone * (1 - access_proximity)).sum()
Target: Minimal structure in ground zone except near AP3
```

#### C2B: No-Go Zone (0-5m)
**Rule:** The 12m center strip is strictly forbidden below 5m.

```
nogo_zone = center_12m[:5m, :, :]
L_nogo = (structure * nogo_zone).sum()
Target: 0 (absolute)
```

### C3: Height Ceiling (Hard)
**Rule:** Structure cannot exceed tallest building + 2m.

```
max_height = max(building_A_height, building_B_height) + 2m
L_ceiling = structure[:, above_max_height, :, :].sum()
Target: 0 (absolute)
```

### C4: Volume Budget
**Rule:** Total grown volume must be 20-40% of Building A's volume.

```
building_A_volume = height_A * width_A * depth_A
min_volume = 0.20 * building_A_volume
max_volume = 0.40 * building_A_volume

grown_volume = structure.sum()

L_volume_under = ReLU(min_volume - grown_volume) / building_A_volume
L_volume_over = ReLU(grown_volume - max_volume) / building_A_volume
```

### C5: Access Connectivity
**Rule:** All 3 access points must be connected by the grown structure.

```
# Flood fill from each access point through structure
connected_to_AP1 = flood_fill(structure, seed=AP1)
connected_to_AP2 = flood_fill(structure, seed=AP2)
connected_to_AP3 = flood_fill(structure, seed=AP3)

# All access points should be in same connected component
L_connectivity = 1 - (all_connected / 3)
Target: 0 (all connected)
```

### C6: Structural Support
**Rule:** All structure must have load path to support (buildings or ground anchors).

```
support = existing_buildings + ground_anchors
L_loadpath = unsupported_voxels / total_voxels
Target: 0
```

### C7: Anti-Flooding
**Rule:** Encourage sparse, corridor-like connections rather than massive volumes.

This is partially handled by C4 (volume budget), but additional measures:

```
# Thickness constraint - prevent blobs
L_thickness = eroded_core.sum() / structure.sum()

# Distribution - structure shouldn't concentrate in one area
L_distribution = variance_of_local_density
```

---

## Constraint Summary Table

| ID | Constraint | Type | Target | Weight |
|----|------------|------|--------|--------|
| C1 | Building Exclusion | Hard | 0 | 100.0 |
| C2A | Ground Zone (0-5m) | Soft | Minimal | 30.0 |
| C2B | No-Go Zone (0-5m) | Hard | 0 | 50.0 |
| C3 | Height Ceiling | Hard | 0 | 100.0 |
| C4 | Volume Budget (20-40%) | Soft | In range | 25.0 |
| C5 | Access Connectivity | Critical | All connected | 40.0 |
| C6 | Structural Support | Soft | >95% | 20.0 |
| C7 | Anti-Flooding | Soft | Sparse | 15.0 |

---

## Channel Architecture

| Channel | Type | Content |
|---------|------|---------|
| 0 | Frozen | Ground plane |
| 1 | Frozen | Existing buildings |
| 2 | Frozen | Access points (all 3) |
| 3 | Frozen | Zone masks (pedestrian, no-go, height limits) |
| 4 | Grown | Structure |
| 5 | Grown | Alive state |
| 6 | Grown | Hidden 1 |
| 7 | Grown | Hidden 2 |

---

## Loss Function

```python
total_loss = (
    # Hard constraints (high weights)
    w_exclusion * L_exclusion +      # C1: No building penetration
    w_nogo * L_nogo +                 # C2B: No-go zone
    w_ceiling * L_ceiling +           # C3: Height limit

    # Critical constraints
    w_connectivity * L_connectivity + # C5: Connect access points

    # Soft constraints
    w_ground * L_ground +             # C2A: Ground zone clearance
    w_volume * (L_vol_under + L_vol_over) +  # C4: Volume budget
    w_loadpath * L_loadpath +         # C6: Structural support
    w_thickness * L_thickness +       # C7: Anti-blob

    # Growth incentive
    w_growth * L_growth               # Encourage reaching access points
)
```

---

## Success Criteria

| Metric | Target | Priority |
|--------|--------|----------|
| Building exclusion | 100% | Critical |
| No-go zone compliance | 100% | Critical |
| Height ceiling compliance | 100% | Critical |
| Access connectivity | 100% (all 3 connected) | Critical |
| Ground zone clearance (0-5m) | >90% empty | High |
| Volume ratio | 20-40% of Building A | High |
| Structural support | >95% | Medium |

---

## Training Curriculum

### Phase 1: Simple Scenes
- Buildings: 20m tall, same height
- Gap: 18m
- Access heights: AP1=10m, AP2=10m (same level)

### Phase 2: Variable Heights
- Buildings: 20-25m, different heights
- Gap: 18-22m
- Access heights: AP1 and AP2 at different levels

### Phase 3: Complex Scenes
- Buildings: 20-30m, different heights
- Gap: 18-25m
- Access heights: Varied, challenging configurations

---

## Visualization

| Element | Color |
|---------|-------|
| Existing buildings | Gray |
| Structure (valid) | Blue |
| Structure (violation) | Red |
| Access points | Green |
| Ground zone (0-5m) | Light yellow (transparent) |
| No-go zone | Red outline |

---

*Step D Constraint Specification - December 2025*
