# Step B v2.0: Ontology-Level Fixes

**Date:** December 2025
**Status:** Proposed Architecture
**Goal:** Eliminate degenerate attractor by fixing loss ontology

---

## The Four Conditions to Fix

| Condition | Current (Broken) | Required (Fixed) |
|-----------|------------------|------------------|
| 1. Legality scope | Global ratios | Local per-voxel |
| 2. Growth-legality alignment | Misaligned | Unified |
| 3. Connectivity definition | Boundary-seeded | Access-seeded |
| 4. Structural encoding | Mass statistics | Load path causality |

---

## Condition 1: Local Legality (Not Global Ratios)

### Problem

Current constraints are global scalars:
```python
# Street void - global ratio
void_ratio = void.sum() / street_zone.sum()

# Anchor compliance - normalized fraction
compliance = 1 - (violation.sum() / street_struct.sum())
```

These can be gamed by reshaping global statistics.

### Solution: Per-Voxel Legality Fields

Instead of computing global ratios, define **legality as a spatial field** and penalize each illegal voxel directly.

```python
class LocalLegalityLoss(nn.Module):
    """Per-voxel legality enforcement.

    Every voxel has a legality value. Illegal voxels are penalized directly,
    not through global ratios.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def compute_legality_field(self, state):
        """Compute per-voxel legality (1 = legal, 0 = illegal)."""
        cfg = self.config
        G = cfg['grid_size']
        street_levels = cfg['street_levels']

        existing = state[:, cfg['ch_existing']]
        anchors = state[:, cfg['ch_anchors']]

        # Initialize: everywhere is legal
        legality = torch.ones_like(existing)

        # Rule 1: Inside buildings is illegal
        legality = legality * (1 - existing)

        # Rule 2: Street level (z < 2) outside anchors is illegal
        street_mask = torch.zeros_like(legality)
        street_mask[:, :street_levels, :, :] = 1.0

        anchor_mask = anchors.clone()
        anchor_mask[:, street_levels:, :, :] = 1.0  # Above street is legal

        # At street level: only anchors are legal
        # Above street level: everywhere (except buildings) is legal
        legality = legality * (anchor_mask + (1 - street_mask))
        legality = torch.clamp(legality, 0, 1)

        return legality

    def forward(self, state):
        cfg = self.config
        structure = state[:, cfg['ch_structure']]

        legality = self.compute_legality_field(state)

        # Direct per-voxel penalty: structure * (1 - legality)
        illegal_structure = structure * (1 - legality)

        # Sum of illegal voxels (not a ratio!)
        return illegal_structure.sum() / (structure.sum() + 1e-8)
```

**Key change:** Legality is a spatial field, not a global statistic. Each voxel knows if it's legal.

### Alternative: Hard Legality Mask

For strict enforcement, multiply structure by legality field in the forward pass:

```python
# In NCA._step()
legality_field = self.compute_legality_field(state)
struct_new = grown_new[:, 0:1] * legality_field
```

This makes illegal growth impossible by construction.

---

## Condition 2: Growth-Legality Alignment

### Problem

Growth incentives point to different locations than legality allows:
- Dice says "grow near access" (often at ground level)
- Legality says "only grow in anchors at ground level"
- These conflict when access is outside anchors

### Solution: Legal Growth Target

Instead of rewarding growth near access unconditionally, reward growth in the **intersection** of access influence AND legal zones.

```python
class AlignedGrowthLoss(nn.Module):
    """Growth incentive aligned with legality.

    Only rewards growth in zones that are both:
    1. Near access points (functional)
    2. Legal (permitted by constraints)
    """

    def __init__(self, config, dilation_radius=5):
        super().__init__()
        self.config = config
        self.dilation_radius = dilation_radius

    def forward(self, state, legality_field):
        cfg = self.config
        structure = state[:, cfg['ch_structure']]
        access = state[:, cfg['ch_access']]

        # Dilate access points
        k = 2 * self.dilation_radius + 1
        access_zone = F.max_pool3d(access.unsqueeze(1), k, 1, self.dilation_radius).squeeze(1)

        # Legal growth target = access zone AND legal
        legal_target = access_zone * legality_field

        # Dice loss toward legal target only
        intersection = (structure * legal_target).sum()
        dice = (2 * intersection + 1) / (structure.sum() + legal_target.sum() + 1)

        return 1 - dice
```

**Key change:** The model is rewarded for growing where it's BOTH useful AND allowed.

### Elevated Access Emphasis

To encourage elevated structures, weight the legal target by height:

```python
# Height weighting: higher = more rewarding
z_coords = torch.arange(G, device=device).float()
height_weight = (z_coords / G).view(1, G, 1, 1).expand(1, G, G, G)

legal_target_weighted = legal_target * (1 + height_weight)
```

---

## Condition 3: Access-Seeded Connectivity

### Problem

Street connectivity is seeded from domain boundaries:
```python
connected[:, :, :, 0] = 1.0    # y=0 edge
connected[:, :, :, -1] = 1.0   # y=max edge
connected[:, :, 0, :] = 1.0    # x=0 edge
connected[:, :, -1, :] = 1.0   # x=max edge
```

This allows a thin perimeter corridor to achieve 100% score while internal circulation is blocked.

### Solution: Seed from Access Points

Connectivity should measure whether void allows movement **between functional access points**, not edge-to-edge traversal.

```python
class AccessConnectivityLoss(nn.Module):
    """Street connectivity seeded from access points.

    Measures whether all access points are mutually reachable
    through the void network, not just boundary connectivity.
    """

    def __init__(self, config, iterations=32):
        super().__init__()
        self.config = config
        self.iterations = iterations

    def forward(self, state):
        cfg = self.config
        street_levels = cfg['street_levels']

        structure = state[:, cfg['ch_structure'], :street_levels, :, :]
        existing = state[:, cfg['ch_existing'], :street_levels, :, :]
        access = state[:, cfg['ch_access'], :street_levels, :, :]

        # Void mask
        void_mask = (1 - structure) * (1 - existing)

        # Seed from ACCESS POINTS (not boundaries)
        # Dilate access slightly to ensure seed is in void
        access_seed = F.max_pool3d(access.unsqueeze(1), 3, 1, 1).squeeze(1)
        connected = access_seed * void_mask

        # Flood fill
        for _ in range(self.iterations):
            dilated = F.max_pool3d(connected.unsqueeze(1), 3, 1, 1).squeeze(1)
            new_connected = torch.max(connected, dilated * void_mask)
            if torch.allclose(connected, new_connected, atol=1e-5):
                break
            connected = new_connected

        # Check: can we reach ALL access points from each other?
        # Every access point location should be in connected region
        access_locations = access > 0.5
        reachable = (connected * access_locations).sum()
        total_access = access_locations.sum() + 1e-8

        # Loss = fraction of access points NOT reachable
        return 1 - (reachable / total_access)
```

**Key change:** Connectivity is functional (access-to-access), not topological (edge-to-edge).

---

## Condition 4: Structural Load Path Encoding

### Problem

Support ratio measures mass distribution, not whether elevated mass is actually supported:
```python
ratio = elevated_mass / ground_contact
```

The model can have 1000 voxels floating above 100 ground voxels with no actual load path between them.

### Solution: Load Path Connectivity

Elevated structure must be **reachable** from ground support through continuous structure.

```python
class LoadPathLoss(nn.Module):
    """Structural load path connectivity.

    Elevated mass must be connected to ground support (anchors + buildings)
    through continuous structure - not just co-existing.
    """

    def __init__(self, config, iterations=32):
        super().__init__()
        self.config = config
        self.iterations = iterations

    def forward(self, state):
        cfg = self.config
        street_levels = cfg['street_levels']

        structure = state[:, cfg['ch_structure']]
        existing = state[:, cfg['ch_existing']]
        anchors = state[:, cfg['ch_anchors']]

        # Valid support = buildings + anchors (at ground level only)
        support = existing.clone()
        support[:, :street_levels, :, :] = torch.max(
            support[:, :street_levels, :, :],
            anchors[:, :street_levels, :, :]
        )

        # Flood fill through structure from support
        connected = support.clone()
        struct_soft = torch.sigmoid(10 * (structure - 0.3))

        for _ in range(self.iterations):
            dilated = F.max_pool3d(connected.unsqueeze(1), 3, 1, 1).squeeze(1)
            new_connected = torch.max(connected, dilated * struct_soft)
            if torch.allclose(connected, new_connected, atol=1e-5):
                break
            connected = new_connected

        # Elevated structure = structure at z >= street_levels
        elevated = structure[:, street_levels:, :, :]
        elevated_connected = connected[:, street_levels:, :, :]

        # Loss = elevated structure NOT connected to support
        unsupported = elevated * (1 - elevated_connected)

        return unsupported.sum() / (elevated.sum() + 1e-8)
```

**Key change:** Load path is traced through structure, not just mass counted.

### Combined with Cantilever

The existing cantilever loss handles local overhangs. LoadPathLoss handles global support connectivity. Both are needed:

- **Cantilever:** "Each voxel has support within N voxels below"
- **LoadPath:** "There exists a path from this voxel to valid ground support"

---

## Revised Loss Architecture (v2.0)

### Loss Categories

| Category | Loss | Type | Weight |
|----------|------|------|--------|
| **Legality** | LocalLegality | Per-voxel | 30.0 |
| **Growth** | AlignedGrowth | Local, legal-aligned | 25.0 |
| **Circulation** | AccessConnectivity | Access-seeded | 15.0 |
| **Structure** | LoadPath | Causal | 20.0 |
| **Structure** | Cantilever | Local | 5.0 |
| **Massing** | Sparsity | Range-bounded | 15.0 |
| **Quality** | Density | Binary incentive | 3.0 |
| **Quality** | TV | Smoothness | 1.0 |

### Removed/Replaced Losses

| Old Loss | Problem | Replacement |
|----------|---------|-------------|
| StreetVoidLoss | Global ratio, gameable | LocalLegality |
| AnchorBudgetLoss | Normalized, gameable | LocalLegality |
| DiceLoss | Not aligned with legality | AlignedGrowth |
| StreetConnectivityLoss | Boundary-seeded | AccessConnectivity |
| SupportRatioLoss | Mass-based, not causal | LoadPath |
| ConnectivityLoss | Includes ground as support | LoadPath (anchors+buildings) |
| ElevatedBonus | Reward without constraint | Implicit in AlignedGrowth |

---

## Expected Behavior Change

### Old Attractor (v1.1)
- Grow at ground level near access
- Violate anchors (penalty is weak)
- Maintain boundary connectivity
- Build tall on illegal foundation

### New Behavior (v2.0)
- Growth near access is only rewarded if LEGAL
- Illegal voxels directly penalized (not ratio-gamed)
- Connectivity measured between access points
- Elevated structure must trace load path to support

### Why This Eliminates the Attractor

| Old Exploit | Why It Fails in v2.0 |
|-------------|---------------------|
| Grow at ground outside anchors | Direct per-voxel penalty, no denominator to game |
| Perimeter corridor for connectivity | Connectivity requires access-to-access paths |
| Tall structure on illegal foundation | LoadPath requires trace to valid support |
| Game global statistics | No global statistics to game |

---

## Implementation Priority

1. **LocalLegalityLoss** - Fixes anchor/void gaming
2. **AlignedGrowthLoss** - Aligns incentives with rules
3. **AccessConnectivityLoss** - Fixes boundary exploit
4. **LoadPathLoss** - Fixes structural causality

Each fix addresses one of the four conditions. All four are needed to eliminate the attractor.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| "Do nothing" returns | AlignedGrowth provides local incentive in legal zones |
| Over-constrained | Legality field has valid elevated zones |
| Slow convergence | Losses are differentiable, gradients flow |
| New exploits | Monitor for unexpected attractors |

---

## Experimental Plan

### Phase 1: Implement and Test Individually
- Test LocalLegalityLoss alone
- Test AlignedGrowthLoss alone
- Verify each addresses its target condition

### Phase 2: Combine and Tune
- Start with equal weights
- Monitor for attractor formation
- Adjust based on which constraints bind

### Phase 3: Difficulty Progression
- Easy scenes first
- Validate no degenerate attractors
- Progress to medium/hard

---

*Step B v2.0 Ontology Fixes - December 2025*
