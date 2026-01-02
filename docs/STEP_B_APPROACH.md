# Step B: Revised Approach

**Date:** December 2025
**Status:** Planning
**Based On:** Step A Failure Analysis

---

## Fundamental Conceptual Corrections

### 1. Walkable = Protected Void

**Step A Error:** Walkable was a grown solid channel on top of structure.

**Step B Correction:**

```
Walkable space = EMPTY volume at/near ground level
             = ABSENCE of structure
             = Protected void for pedestrian/vehicle circulation
```

The street is not something we grow. It is something we PROTECT from growth.

**Exceptions (where structure MAY touch ground):**
1. Structural anchors (limited budget)
2. Architectural thickening around access points

### 2. Ground Is Precious

**Step A Error:** No penalty for structure touching ground.

**Step B Correction:**

```
Ground contact = expensive operation
              = must be justified by anchor need
              = budgeted and constrained
```

The default state of ground is PROTECTED. Structure must earn the right to touch it.

### 3. All Constraints From Epoch 0

**Step A Error:** Constraints introduced incrementally, allowing degenerate solutions.

**Step B Correction:**

```
All constraints active from the start
Curriculum varies SCENE COMPLEXITY, not constraint presence
```

| Phase | Scene Complexity | Constraints |
|-------|------------------|-------------|
| 1 | Easy (wide gaps, few access points) | ALL |
| 2 | Medium (narrower gaps, more access) | ALL |
| 3 | Hard (complex layouts, many access) | ALL |

---

## Revised Channel Architecture

### Step A Channels (8 total)

| Channel | Type | Purpose |
|---------|------|---------|
| 0 | Frozen | Ground plane |
| 1 | Frozen | Existing buildings |
| 2 | Frozen | Access points |
| 3 | Frozen | (unused) |
| 4 | Grown | Structure |
| 5 | Grown | Walkable (WRONG - was solid) |
| 6 | Grown | Alive state |
| 7 | Grown | Hidden |

### Step B Channels (8 total) - REVISED

| Channel | Type | Purpose |
|---------|------|---------|
| 0 | Frozen | Ground plane |
| 1 | Frozen | Existing buildings |
| 2 | Frozen | Access points |
| 3 | Frozen | **Anchor zones** (where ground contact allowed) |
| 4 | Grown | Structure |
| 5 | Grown | **Surface type** (floor/wall/ceiling indicator) |
| 6 | Grown | Alive state |
| 7 | Grown | Hidden |

**Key Change:** No "walkable" channel. Walkability is computed from the ABSENCE of structure at ground level.

---

## Revised Constraint Formulation

### C1: Structural Soundness

#### C1A: Connectivity (REVISED)

**Step A:** Connect to ground OR existing buildings.
**Step B:** Connect to existing buildings OR designated anchor zones. Penalize excessive ground contact.

```
connectivity_loss = disconnected_voxels / total_voxels
anchor_budget_loss = max(0, ground_contact - allowed_anchors)
```

#### C1B: Cantilever (UNCHANGED)

Limit horizontal overhangs to N voxels.

### C2: Street-Level Openness (CRITICAL - Was Phase 2, Now Phase 1)

#### C2A: Ground Void

**Definition:** The majority of ground-level space must remain EMPTY.

```
street_zone = ground_plane AND NOT existing_buildings AND NOT anchor_zones
street_void_loss = structure_in_street_zone.sum() / street_zone.sum()
```

**Target:** >70% of street zone remains empty.

#### C2B: Street Connectivity

**Definition:** Empty space at ground level must form a connected network.

```
void_at_ground = 1 - structure[:, 0:2, :, :]  # First 2 z-levels
street_connected = flood_fill(void_at_ground, from_edges)
street_connectivity_loss = 1 - (street_connected.sum() / void_at_ground.sum())
```

**Target:** >90% of ground void is connected.

### C3: Massing Distribution

#### C3A: Sparsity (REVISED)

**Step A:** 3-25% fill ratio (too permissive).
**Step B:** 5-15% fill ratio, encouraging more sparse structures.

#### C3B: Facade Clearance (UNCHANGED)

Structure must not block building facades.

#### C3C: Volume Distribution (UNCHANGED)

Mass should be distributed, not concentrated.

### C4: Access and Circulation

#### C4A: Access Reach (REVISED)

**Step A:** Structure near access points.
**Step B:** Structure provides PATHS TO access points, not just proximity.

```
access_zones = dilate(access_points, radius=3)
path_exists = pathfind(structure, from=ground, to=access_zones)
access_reach_loss = 1 - path_exists.mean()
```

#### C4B: Walkable Surfaces (REVISED - Computed, Not Grown)

**Step A:** Grown channel indicating walkable material.
**Step B:** Computed property of structure surfaces.

```
walkable_surface = top_faces_of_structure  # Horizontal surfaces you can walk on
walkable_coverage = walkable_surface.sum() / structure.sum()
```

This is a DERIVED metric, not a grown channel.

#### C4C: Ground Anchor Budget (NEW)

**Definition:** Limit how much structure can touch the ground.

```
ground_footprint = (structure[:, 0, :, :] > 0.5).sum()
anchor_budget_loss = relu(ground_footprint - max_allowed_anchors)
```

**Parameters:**
- `max_allowed_anchors`: e.g., 5% of ground area or N voxels

---

## Revised Loss Function

### All Losses Active From Epoch 0

```python
total_loss = (
    # Structural
    w_conn * connectivity_loss +
    w_cant * cantilever_loss +
    w_anchor * anchor_budget_loss +

    # Street Protection (CRITICAL)
    w_void * street_void_loss +
    w_street_conn * street_connectivity_loss +

    # Massing
    w_sparse * sparsity_loss +
    w_facade * facade_clearance_loss +
    w_dist * distribution_loss +

    # Access
    w_reach * access_reach_loss +

    # Growth Incentive
    w_dice * dice_loss +

    # Quality (REDUCED weights)
    w_density * density_penalty +
    w_tv * total_variation
)
```

### Suggested Weight Priorities

| Loss | Weight | Priority |
|------|--------|----------|
| Street void | 50.0 | CRITICAL |
| Anchor budget | 30.0 | HIGH |
| Connectivity | 10.0 | HIGH |
| Sparsity | 20.0 | HIGH |
| Access reach | 10.0 | MEDIUM |
| Cantilever | 5.0 | MEDIUM |
| Dice | 5.0 | MEDIUM |
| Density | 5.0 | LOW (reduced from 15) |
| TV | 1.0 | LOW (reduced from 2) |

**Key Change:** Street void and anchor budget are now the highest priority losses.

---

## Curriculum Design (Scene Complexity, Not Constraints)

### Phase 1: Easy Scenes

- Gap width: 14-18 voxels (wide)
- Buildings: 2, similar heights
- Access points: 1 ground, 1 elevated
- Anchor budget: 10% of ground area

### Phase 2: Medium Scenes

- Gap width: 10-14 voxels
- Buildings: 2, varying heights
- Access points: 2 ground, 2 elevated
- Anchor budget: 7% of ground area

### Phase 3: Hard Scenes

- Gap width: 6-10 voxels (narrow)
- Buildings: 2-4, complex arrangement
- Access points: 2-3 ground, 2-3 elevated
- Anchor budget: 5% of ground area

**Note:** ALL constraints are active in ALL phases. Only scene difficulty changes.

---

## Anchor Zone Definition

### What Are Anchor Zones?

Designated areas where structure is PERMITTED to touch the ground:

1. **Access projections:** Small areas directly beneath/around access points
2. **Building adjacencies:** Narrow strips along building facades facing the gap

### Implementation

```python
def create_anchor_zones(access_points, buildings, config):
    anchor_mask = torch.zeros_like(ground)

    # Access projections (2x2 footprint per access point)
    for ap in access_points:
        anchor_mask[:, 0, ap.y-1:ap.y+2, ap.x-1:ap.x+2] = 1.0

    # Building adjacencies (1-voxel strip along gap-facing facades)
    for b in buildings:
        if b.faces_gap:
            anchor_mask[:, 0, b.y_range, b.gap_x] = 1.0

    return anchor_mask
```

---

## Visualization Updates

### What to Show

| Element | Color | Meaning |
|---------|-------|---------|
| Existing buildings | Gray | Context |
| Structure | Blue | Generated pavilion |
| Access points | Green | Entry/exit locations |
| **Street void** | **Transparent/White** | **Protected circulation** |
| **Ground violations** | **Red** | **Structure in street zone** |
| Anchor zones | Yellow outline | Allowed ground contact |

### New Metrics to Display

1. Street void ratio (target: >70%)
2. Ground footprint (target: within budget)
3. Street connectivity (target: >90%)
4. Path to access (target: >80%)

---

## Implementation Plan

### Notebook Structure (Revised)

| Notebook | Purpose |
|----------|---------|
| NB01_Foundation_v2.0 | Revised channels, anchor zones, void computation |
| NB02_AllConstraints_v1.0 | All constraints from epoch 0 |
| NB03_EasyScenes | Phase 1: Easy scenes with all constraints |
| NB04_MediumScenes | Phase 2: Medium complexity |
| NB05_HardScenes | Phase 3: Hard complexity |
| NB06_Evaluation | Final evaluation and ablation |

### Key Implementation Changes

1. **Remove walkable channel** - Walkability is computed, not grown
2. **Add anchor zone channel** - Frozen channel defining allowed ground contact
3. **Add street void loss** - Critical loss protecting ground circulation
4. **Add anchor budget loss** - Limit ground footprint
5. **Reduce quality loss weights** - Allow more cellular forms
6. **Add street connectivity loss** - Ensure void is navigable

---

## Success Criteria (Revised)

| Metric | Target | Priority |
|--------|--------|----------|
| Street void ratio | >70% | CRITICAL |
| Street connectivity | >90% | CRITICAL |
| Anchor budget compliance | 100% | HIGH |
| Structural connectivity | >95% | HIGH |
| Cantilever compliance | >90% | MEDIUM |
| Access path existence | >80% | MEDIUM |
| Fill ratio | 5-15% | MEDIUM |

---

## Risk Mitigation

### Risk 1: Model Cannot Learn With All Constraints

**Mitigation:** Start with high weights on street void and anchor budget; these define the problem. Other constraints can have lower initial weights.

### Risk 2: No Structure Grows At All

**Mitigation:** Dice loss provides growth incentive. Anchor zones permit limited ground contact.

### Risk 3: Structure Only Grows in Anchor Zones

**Mitigation:** Access reach loss encourages elevated growth toward access points. Connectivity to buildings provides alternative support.

---

## Summary of Changes from Step A

| Aspect | Step A | Step B |
|--------|--------|--------|
| Walkable | Grown solid channel | Computed void property |
| Ground contact | Free | Expensive, budgeted |
| Street protection | Phase 2 | Epoch 0 |
| Constraints | Incremental | All from start |
| Curriculum | Constraint presence | Scene complexity |
| Quality losses | High weights | Reduced weights |
| Anchor control | None | Explicit zones + budget |

---

*Step B Planning - December 2025*
