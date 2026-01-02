# Step B v3.0: Architectural Intent

**Date:** December 2025
**Status:** Proposed Architecture
**Goal:** Transform constraint-satisfying blobs into architecturally meaningful pavilions

---

## Why v2.0 Produced Blobs

v2.0 (notebook v2.0, specs v3.0) successfully fixed the ontology problems:
- Legality: 100% (no illegal structure)
- Load Path: 99% (structure properly supported)
- Access Reach: 100% (access points connected)

But it produced **facade-coating blobs** because:

| Missing Element | Consequence |
|-----------------|-------------|
| No path-based target | Point dilation creates bulbs, not corridors |
| No outside-target penalty | Mass can expand freely after minimal alignment |
| No thickness limit | Blobs are structurally optimal |
| No facade contact limit | Facades provide infinite free support |

**v2.0 was ontologically sound but architecturally undefined.**

---

## What is a Pavilion?

### Architectural Definition

A pavilion is a **connecting structure** between access points:
- Enter from one access point
- Exit from others
- Experience varied spaces along the path

### Key Characteristics

1. **Connectivity** - Primary purpose is linking access points
2. **Linearity** - Generally elongated, not spherical
3. **Bounded thickness** - Corridors and rooms, not solid masses
4. **Ground preservation** - Leaves circulation space open
5. **Spatial variety** - Compression (narrow) and decompression (wider)
6. **Limited facade contact** - Not coating existing buildings

### What a Pavilion is NOT

- A blob glued to facades
- A solid mass filling available space
- A thin perimeter shell
- Random scattered volumes

---

## Mathematical Translation

### 1. Connectivity → Path-Based Targets

**Architectural:** Links access points through continuous paths

**Mathematical:**
```python
# Instead of dilating access points (creates bulbs)
target = dilate(access_points, radius=5)

# Compute shortest paths between access pairs (creates corridors)
paths = shortest_paths_between_all_access_pairs(state)
corridor_target = dilate(paths, radius=corridor_width)
```

**Effect:** Target geometry is linear (corridors) instead of spherical (bulbs)

### 2. Not Blobs → Thickness Constraint

**Architectural:** Bounded thickness with variety

**Mathematical:**
```python
# Erode structure iteratively
# Surviving voxels are "core" (too thick)
core = structure
for _ in range(max_thickness):
    core = erode_3d(core)

# Penalize thick cores
L_thickness = core.sum() / structure.sum()
```

**Effect:** Prevents solid blobs, allows corridors and rooms

### 3. Purposeful Mass → Outside Target Penalty

**Architectural:** Every voxel serves connectivity

**Mathematical:**
```python
# Structure outside corridor target
outside = structure * (1 - corridor_target)

# Explicit penalty (this was missing in v2.0!)
L_outside = outside.sum() / structure.sum()
```

**Effect:** Cannot expand freely after minimal alignment

### 4. Limited Facade Contact → Contact Budget

**Architectural:** Touch buildings at connection points, not everywhere

**Mathematical:**
```python
# Facade zone = adjacent to buildings
facade_zone = dilate(existing) - existing

# Contact ratio
contact_ratio = (structure * facade_zone).sum() / structure.sum()

# Budget-based penalty
L_facade = ReLU(contact_ratio - max_contact)
```

**Effect:** Forces some free-standing construction

### 5. Spatial Variety → Thickness Variance (Optional)

**Architectural:** Compression and decompression zones

**Mathematical:**
```python
# Allow some thickness variation
# Don't penalize uniform thickness, penalize extremes
local_thickness = compute_local_thickness(structure)
variance = local_thickness.var()

# Reward moderate variance (not too uniform, not too extreme)
L_variety = (variance - target_variance).abs()
```

**Effect:** Encourages spatial rhythm, not monotonous tubes

---

## New Loss Architecture

### Losses Added in v3.0

| Loss | Purpose | Weight |
|------|---------|--------|
| PathConnectivityLoss | Corridor-based targets | 25.0 |
| OutsideTargetLoss | Penalize off-path mass | 20.0 |
| ThicknessLoss | Prevent blobs | 15.0 |
| FacadeContactLoss | Limit facade coating | 10.0 |

### Losses Modified

| Loss | Change | New Weight |
|------|--------|------------|
| LoadPathLoss | Reduced (shape handled elsewhere) | 10.0 |

### Losses Unchanged

| Loss | Weight |
|------|--------|
| LocalLegalityLoss | 30.0 |
| AccessConnectivityLoss | 15.0 |
| SparsityLoss | 15.0 |
| CantileverLoss | 5.0 |
| DensityPenalty | 3.0 |
| TotalVariation3D | 1.0 |

---

## Implementation Challenges

### 1. Differentiable Path Computation

Shortest paths are typically computed via graph algorithms (Dijkstra, A*), which are not differentiable.

**Solution:** Soft path computation via distance propagation

```python
def soft_shortest_path(start, end, legal_mask, iterations=50):
    """Differentiable shortest path approximation."""
    # Initialize distance field
    distance = torch.full_like(legal_mask, float('inf'))
    distance[start] = 0

    # Propagate distances
    for _ in range(iterations):
        # Expand frontier
        expanded = F.max_pool3d(-distance, 3, 1, 1)
        expanded = -expanded + 1  # Add step cost

        # Update distances (only in legal space)
        distance = torch.min(distance, expanded) * legal_mask +
                   float('inf') * (1 - legal_mask)

    # Extract path via gradient descent on distance field
    path = trace_path(distance, start, end)
    return path
```

### 2. Multiple Access Point Pairs

With N access points, there are N*(N-1)/2 pairs.

**Solution:** Compute paths for all pairs, union into single target

```python
def compute_corridor_target(access_points, legal_mask):
    corridor = torch.zeros_like(legal_mask)

    for i in range(len(access_points)):
        for j in range(i+1, len(access_points)):
            path = soft_shortest_path(access_points[i], access_points[j], legal_mask)
            corridor = torch.max(corridor, dilate(path, corridor_width))

    return corridor
```

### 3. Efficient Erosion

Morphological erosion via min-pooling:

```python
def erode_3d(x, iterations=1):
    """Erode binary/soft volume."""
    result = x
    for _ in range(iterations):
        # Min-pool = erode
        result = -F.max_pool3d(-result, 3, 1, 1)
    return result
```

---

## Expected Behavior Change

### v2.0 Attractor (Blobs)
- Coat facades for free support
- Fill legal space near access minimally
- Expand into safe zones freely
- Form massive connected blobs

### v3.0 Expected Behavior (Corridors)
- Follow paths between access points
- Maintain bounded thickness
- Touch facades only at connections
- Create bridge-like spans

---

## Success Criteria

| Metric | v2.0 Result | v3.0 Target |
|--------|-------------|-------------|
| Legality | 100% | 100% |
| Path alignment | 15-30% | >70% |
| Outside target | ~70% | <20% |
| Thickness compliance | ~50% | >90% |
| Facade contact | ~40% | <20% |
| Fill ratio | 25-30% | 5-15% |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Path computation too expensive | Use soft differentiable approximation |
| Over-constrained (no growth) | Balance positive (path) and negative (outside) incentives |
| Fragmented structure | LoadPath still enforces connectivity |
| New degenerate attractor | Monitor for unexpected equilibria |

---

*Step B v3.0 Architectural Intent - December 2025*
