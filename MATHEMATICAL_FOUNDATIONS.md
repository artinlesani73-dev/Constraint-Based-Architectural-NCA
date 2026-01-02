# Mathematical Foundations

## Constraint-Based Architectural NCA: A Systematic Study of Volumetric Design Constraints

**Version:** 2.2 (MODEL C Alignment)
**Date:** January 2026
**Status:** Reference Document (Project Complete)
**Purpose:** Mathematical formulation grounded in experimental findings

> **Note:** This document captures the mathematical framework developed during the research phase. The formulations here informed the final v3.1 model that is now deployed. For current system specifications, see `SPECIFICATION.md`.

---

## Revision History

| Version | Changes |
|---------|---------|
| 1.0 | Initial formulation with curriculum phases |
| 2.0 | Draft revision based on failure analysis |
| **2.1** | **Evidence-based revision linking math to observed failures** |
| **2.2** | **Aligned with MODEL C notebook (v3.1_C)** |

---

## Part 1: Experimental Context

### What Step A Revealed

The following mathematical problems were identified through experimentation:

| Mathematical Issue | Observed Symptom | Experimental Evidence |
|-------------------|------------------|----------------------|
| Support = Ground + Buildings | Ground carpet solution | Connectivity ~98% via floor spreading |
| No street void loss | 0% street protection | NCA filled entire gap at z=0,1 |
| Walkable as grown channel | Semantic inversion | Scattered islands on blob exterior |
| Quality weights too high | Topology-optimized blobs | DensityPenalty=15, TV=2 destroyed porosity |
| Sparsity range 3-25% | Dense blob formation | Actual: 15-22% concentrated in blobs |

### Mathematical Corrections Required

1. **Redefine support set** - Exclude free ground from valid support
2. **Add street void loss** - Penalize any structure at z < 2
3. **Remove walkable channel** - Compute from void, not as grown material
4. **Reduce quality weights** - Preserve NCA cellular morphology
5. **Add anchor constraint** - Restrict where ground contact is allowed

---

## 1. Notation and Preliminaries

### 1.1 Basic Notation

| Symbol | Description | Domain |
|--------|-------------|--------|
| G | Grid size | Positive integer (default 32) |
| D | Grid depth (Z-axis, height) | = G |
| H | Grid height (Y-axis, depth into scene) | = G |
| W | Grid width (X-axis, left-right) | = G |
| C | Number of channels | 8 |
| S | State tensor | R^(C x D x H x W) |
| s(z,y,x) | Voxel value at position | [0, 1] |

### 1.2 Index Conventions

- **z**: Vertical axis (0 = ground level, increasing upward)
- **y**: Depth axis (into scene)
- **x**: Horizontal axis (left-right, street direction)
- **Batch dimension B** omitted for clarity

### 1.3 Set Notation

```
V = {(z, y, x) : 0 <= z < D, 0 <= y < H, 0 <= x < W}    (All voxels)

N_6(p) = {q in V : ||p - q||_1 = 1}                      (6-connectivity)

N_26(p) = {q in V : ||p - q||_inf = 1, q != p}          (26-connectivity)
```

### 1.4 Threshold Function

```
tau(s, theta) = sigmoid(k * (s - theta))

where:
  s     = soft voxel value in [0, 1]
  theta = threshold (typically 0.3 or 0.5)
  k     = steepness (typically 10)
```

---

## 2. State Space Definition

### 2.1 Channel Structure (Revised Based on Evidence)

**Step A had:** 8 channels with walkable as grown solid (ch_5)

**Problem observed:** Walkable channel never developed properly. NCA painted scattered islands on blob surfaces instead of protecting circulation space.

**Step B revision:**

```
S = [S_frozen | S_grown]

S_frozen in R^(4 x D x H x W):
  S[0] = G   (Ground plane)
  S[1] = E   (Existing buildings)
  S[2] = A   (Access points)
  S[3] = K   (Anchor zones - NEW)     <-- Defines allowed ground contact

S_grown in R^(4 x D x H x W):
  S[4] = P   (Pavilion structure)
  S[5] = T   (Surface type indicator) <-- Replaces walkable
  S[6] = L   (Alive state)
  S[7] = M   (Memory/hidden channel)
```

**Key change:** Walkable channel REMOVED. Walkability is computed from void.

### 2.2 Domain Definitions (Revised Based on Evidence)

**Step A problem:** Support included free ground, enabling ground carpet solution.

**Available space** (where pavilion can grow):

```
Omega = {p in V : E(p) = 0}
```

**Support region** (REVISED - NO free ground):

```
Sigma = {p in V : E(p) = 1 OR K(p) = 1}
      = Existing buildings OR Anchor zones

NOTE: Ground plane G is NOT support unless anchor zone exists.
```

**Evidence for change:** Step A connectivity was ~98% but achieved via ground spreading. By excluding free ground, structure must connect to buildings or designated anchors.

**Street zone** (ground level available space):

```
Z_street = {p in V : z(p) < z_street AND E(p) = 0}

where z_street = 6 (MODEL C: keep z <= 5 open)
```

**Anchor zones** (allowed ground contact):

```
K(p) = 1 if p is in designated anchor zone, 0 otherwise

K_set = {p in Z_street : K(p) = 1}
```

---

## 3. Neural Cellular Automata Dynamics

### 3.1 Perception Function

**Step A bug found:** Zero-padding caused systematic x=0 growth bias.

**Fix applied:** 3D Sobel filters with REPLICATE padding:

```
Phi(S) = [S, nabla_x S, nabla_y S, nabla_z S]

Implementation:
  x_padded = F.pad(S, (1,1,1,1,1,1), mode='replicate')
  nabla_x S = conv3d(x_padded, K_x, padding=0)
```

**Replicate padding eliminates boundary gradient artifacts** that caused systematic x=0 bias in Step A.

### 3.2 Update Network

```
Delta = f_theta(Phi(S))

f_theta: R^32 -> R^4

f_theta = Conv3D(32 -> 96) -> ReLU -> Conv3D(96 -> 96) -> ReLU -> Conv3D(96 -> 4)
```

### 3.3 Stochastic Update Rule

```
M ~ Bernoulli(rho)^(D x H x W)    where rho = 0.5

S_grown(t+1) = S_grown(t) + eta * Delta * M

where eta = 0.1 (update scale)
```

### 3.4 Hard Mask: Exclusion from Buildings

**Step A bug found:** Structure grew inside existing buildings.

**Fix applied:**

```
P_new = P_updated * (1 - E)

Structure cannot exist inside existing buildings.
```

---

## 4. Constraint 1: Structural Soundness

### 4.1 Sub-constraint 1A: Connectivity (REVISED)

**Step A problem:** Connectivity rewarded ANY ground contact. NCA achieved 98% connectivity by spreading across z=0 (ground carpet).

**Evidence:** Observed behavior was easiest path to satisfy loss, but architecturally invalid.

**Step B definition:** Structure must connect to existing buildings OR anchor zones. **NOT free ground.**

**Support Definition (Revised):**

```
Support = E OR K
        = Existing_buildings OR Anchor_zones

NOTE: Ground plane is NOT valid support unless anchored.
```

**Flood-Fill Connectivity:**

```
C^(0) = Support
C^(k+1) = max(C^(k), Dilate(C^(k)) * tau(P, theta))

After K iterations:

L_conn = sum[P * (1 - C^(K))] / [sum(P) + epsilon]
```

**Interpretation:** Fraction of structure disconnected from valid support.

**Why this works:** Prevents ground carpet solution by excluding free ground from support.

### 4.2 Sub-constraint 1B: Cantilever Limit

Unchanged from Step A. Worked correctly.

```
For voxel at height z:
  Support_below = max over [z-N : z-1] of Structure

Unsupported = P * (1 - dilate(Support_below))

L_cant = mean(Unsupported for z >= N)
```

---

## 5. Constraint 2: Street-Level Protection (NEW - CRITICAL)

**Why added:** Step A had no mechanism to protect ground. Observed 0% street void.

### 5.1 Sub-constraint 2A: Street Void Ratio (CRITICAL)

**Definition:** Street zone must remain mostly empty for circulation.

**This is the highest priority constraint based on Step A failure.**

**Street Zone:**

```
Z_street = {p in V : z(p) < z_street AND E(p) = 0}

where z_street = 6 (MODEL C)
```

**Street Occupation:**

```
rho_occupation = sum[P * Z_street] / |Z_street|
```

**Street Void Ratio:**

```
rho_void = 1 - rho_occupation
```

**Loss Function:**

```
L_void = ReLU(rho_occupation - (1 - rho_target)) + 0.1 * rho_occupation

where rho_target = 0.70 (target 70% void)
```

**Weight:** 50.0 (CRITICAL)

**Why weight 50:** Must dominate connectivity (10) to prevent preference for ground carpet solution.

### 5.2 Sub-constraint 2B: Street Connectivity (NEW)

**Definition:** Void at street level must form connected network.

**Rationale:** Step A produced fragmented spaces that don't allow circulation even if void ratio were high.

**Void Mask:**

```
V_street = (1 - P) * (1 - E) at z < z_street
```

**Flood-Fill from Edges:**

```
C_void^(0) = edges of scene at street level
C_void^(k+1) = max(C_void^(k), Dilate(C_void^(k)) * V_street)
```

**Loss:**

```
L_street_conn = 1 - sum(C_void^(K)) / sum(V_street)
```

**Weight:** 10.0

### 5.3 Corridor Target + Height Band (v3.1_C)

MODEL C defines a 3D corridor target computed from access points in the seed state,
then clamps that corridor to a limited vertical band around access heights.

**Access centroids (seed state):**

```
centroids = connected_components(A)  # 3D access voxels
```

**Path corridor (voxel volume):**

```
C = Dilate(Path(centroids), corridor_width)
```

**Vertical envelope (local thickening):**

```
C = C union Envelope(C, vertical_envelope)
```

**Z-band clamp (MODEL C):**

```
z_min = min(z_centroids) - corridor_z_margin
z_max = max(z_centroids) + corridor_z_margin
C = C * 1[z_min <= z < z_max]
```

**MODEL C parameters:**

```
corridor_width = 1
vertical_envelope = 1
corridor_z_margin = 3
```

**Intent:** Prevent full-height corridor fill by restricting the corridor target to
the access height band.

---

## 6. Constraint 3: Massing Control

### 6.1 Sub-constraint 3A: Sparsity (REVISED)

**Step A problem:** Range 3-25% allowed dense blobs. Observed 15-22% concentrated in solid masses.

**Step B revision:** Tightened to 5-15%.

```
rho_vol = sum(P) / |Omega|

L_sparse = ReLU(rho_vol - 0.15) + 0.5 * ReLU(0.05 - rho_vol)
```

**Range:** 5-15% fill (tighter than Step A's 3-25%)

**Weight:** 20.0 (increased from 5.0)

### 6.2 Sub-constraint 3B: Anchor Budget (NEW - Critical)

**Step A problem:** No restriction on where structure touches ground. Result: ground carpet.

**Definition:** Ground-level structure may ONLY exist in anchor zones.

**Anchor Zones:**

```
K = Anchor zones (frozen channel 3)
```

**Structure in Street Zone:**

```
P_street = P[:, :z_street, :, :] * (1 - E[:, :z_street, :, :])
```

**Violation (outside anchors):**

```
Violation = P_street * (1 - K[:, :z_street, :, :])
```

**Loss:**

```
L_anchor = sum(Violation) / [sum(P_street) + epsilon]
```

**Interpretation:** Fraction of street-level structure outside anchor zones.

**Weight:** 30.0 (HIGH)

**Anchor Zone Generation:**

```
K(p) = 1 if:
  - p is within radius r of ground-level access point, OR
  - p is on building facade strip facing the gap

Anchor budget: |K_set| / |Z_street| <= budget_ratio

budget_ratio by difficulty:
  Easy:   10%
  Medium: 7%
  Hard:   5%
```

### 6.3 Sub-constraint 3C: Distribution (Optional)

```
CV = std(regional_densities) / mean(regional_densities)

L_dist = CV
```

**Weight:** 2.0

---

## 7. Constraint 4: Access Connectivity

### 7.1 Access Reach (UNCHANGED)

```
A_dilated = Dilate(A, radius=3)
Target = A_dilated * Omega

Reach = sum(P * Target) / sum(Target)

L_reach = 1 - Reach
```

**Weight:** 10.0

**MODEL C note:** Access points are typically placed on facades at z >= 3 and
the corridor target is computed from those access centroids (see 5.3).

### 7.2 Walkable Surfaces (REMOVED)

**Step A had:**
```
W = walkable channel (grown)
coverage = sum(W) / sum(P)
```

**Problem observed:** Walkable modeled as solid material. NCA painted isolated islands on blob surface.

**Step B:** Walkable is COMPUTED from void, not grown:

```
Walkable_space = 1 - P at z < z_street
               = Street void (where people walk)
```

**No loss function for walkable.** Street void loss (C2A) handles this.

---

## 8. Combined Loss Function

### 8.1 All Constraints Active from Epoch 0

**Step A approach (failed):** Phased introduction of constraints.

**Problem:** Phase 1 found local optimum (ground carpet) incompatible with Phase 2 goals. Degenerate solution persisted.

**Step B approach:** All losses active from start:

```
L_total = L_void_street * w_void
        + L_anchor * w_anchor
        + L_sparse * w_sparse
        + L_conn * w_conn
        + L_street_conn * w_street_conn
        + L_reach * w_reach
        + L_cant * w_cant
        + L_dice * w_dice
        + L_density * w_density
        + L_tv * w_tv
```

### 8.2 Loss Weights (Evidence-Based)

| Loss | Weight | Step A Weight | Change Rationale |
|------|--------|---------------|------------------|
| Street Void | 50.0 | N/A | New; must dominate connectivity |
| Anchor Budget | 30.0 | N/A | New; ground must be precious |
| Sparsity | 20.0 | 5.0 | Increased to enforce tighter range |
| Connectivity | 10.0 | 10.0 | Unchanged weight, but redefined support |
| Street Connectivity | 10.0 | N/A | New; ensures void network |
| Access Reach | 10.0 | 10.0 | Unchanged |
| Cantilever | 5.0 | 5.0 | Unchanged |
| Dice | 5.0 | 5.0 | Unchanged |
| Density Penalty | 5.0 | **15.0** | **Reduced** - was destroying porosity |
| Total Variation | 1.0 | **2.0** | **Reduced** - was minimizing surface area |

### 8.3 Dice Loss for Growth

```
Target = Dilate(A, radius=R) * Omega

Dice = (2 * sum(P * Target) + 1) / (sum(P) + sum(Target) + 1)

L_dice = 1 - Dice
```

### 8.4 Quality Losses (REDUCED)

**Step A problem:** DensityPenalty=15 + TotalVariation=2 caused topology-optimization behavior. NCA produced smooth, minimal surface area blobs instead of porous architecture.

**Step B revision:**

```
L_density = mean(P * (1 - P))         # Weight: 5 (was 15)

L_tv = |dP/dz| + |dP/dy| + |dP/dx|    # Weight: 1 (was 2)
```

**Weights reduced** to preserve cellular NCA behavior.

---

## 9. Evaluation Metrics

### 9.1 Primary Metrics (Based on Step A Failures)

| Metric | Step A Result | Step B Target | Evidence |
|--------|---------------|---------------|----------|
| Street Void | ~0% | >70% | Ground carpet failure |
| Anchor Compliance | N/A | 100% | New constraint |
| Street Connectivity | N/A | >90% | New constraint |
| Structural Connectivity | ~98% | >95% | Was working, keep it |
| Fill Ratio | 15-22% | 5-15% | Blob formation |

### 9.2 Validity Definition (Step B)

```
Valid = (M_void > 0.70)
    AND (M_anchor == 1.00)
    AND (M_street_conn > 0.90)
    AND (M_struct_conn > 0.95)
    AND (0.05 < M_fill < 0.15)
```

---

## Appendix A: Step A vs Step B Summary

| Aspect | Step A | Step B | Evidence for Change |
|--------|--------|--------|---------------------|
| Walkable | Grown channel | Computed void | Semantic inversion observed |
| Ground contact | Free | Anchor zones only | Ground carpet observed |
| Support | Ground + buildings | Buildings + anchors | 98% connectivity via floor |
| Street protection | Phase 2 | Epoch 0, weight 50 | 0% void in Phase 1 |
| Curriculum | Constraint phases | Scene complexity | Degenerate local optima |
| Fill ratio | 3-25% | 5-15% | Blobs at 15-22% |
| Density weight | 15 | 5 | Topology optimization |
| TV weight | 2 | 1 | Topology optimization |

---

## Appendix B: Technical Bug Fixes

| Bug | Mathematical Impact | Fix |
|-----|---------------------|-----|
| Zero-padding Sobel | Gradient asymmetry causing x=0 bias | Replicate padding |
| No building exclusion | Structure inside buildings counted as connected | Hard mask P*(1-E) |
| Walkable bias = -0.5 | Channel never developed | Bias = 0.1 (now removed) |

---

*Mathematical Foundations v2.1 - Evidence-Based Revision - December 2025*
