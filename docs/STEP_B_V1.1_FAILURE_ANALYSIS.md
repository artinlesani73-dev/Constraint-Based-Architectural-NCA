# Step B v1.1 Failure Analysis

**Date:** December 2025
**Status:** Critical - Ontology Problem Identified
**Source:** External Review + Internal Analysis

---

## Executive Summary

Step B v1.1 produces a **stable degenerate attractor** that is mathematically optimal but architecturally meaningless. This is not a tuning problem - it is a fundamental flaw in the loss ontology.

The system converges to the same configuration every run because the objective landscape makes this configuration globally optimal.

---

## Observed Attractor Characteristics

| Metric | Observed | Target | Pattern |
|--------|----------|--------|---------|
| Street Void | 65-69% | >70% | Plateau just below threshold |
| Anchor Compliance | 20-35% | 100% | Collapse after growth onset |
| Street Connectivity | ~100% | >90% | Saturated (exploited) |
| Support Ratio | 11:1 | <4:1 | Explosion |
| Fill Ratio | 15-17% | 5-15% | Slightly over |

This pattern appears in every run because it IS the optimum.

---

## Root Cause: Causal Structure Misalignment

### The Fundamental Problem

**Local growth incentives** are mixed with **global scalar constraints** without aligning their causal structure.

| Category | Losses | Formulation | Effect |
|----------|--------|-------------|--------|
| Growth | Dice, AccessReach, Sparsity-under | Local, spatial | Strong, targeted |
| Protection | Void, Anchor, Connectivity | Global, statistical | Weak, gameable |

The NCA is rewarded to grow in exactly the zones that constraints are trying to protect, and penalties are formulated in ways that can be gamed by reshaping global statistics rather than obeying local rules.

### The Resulting Exploit

The model discovers a stable compromise:

1. **Grow aggressively** around access points (satisfies Dice, AccessReach)
2. **Violate anchors** at ground level (penalty is normalized, gameable)
3. **Preserve boundary void** (thin perimeter satisfies connectivity)
4. **Offload mass upward** (elevated bonus, but on illegal foundation)

This is mathematically rational given the objective.

---

## Specific Structural Pathologies

### 1. Street Void Ceiling (~69%)

**Problem:** Void is enforced as a global scalar ratio.

**Mechanism:** Once the system reaches a configuration where removing more street mass would break boundary-seeded connectivity, the optimizer hits a Pareto wall.

**Why it plateaus:** The model can't reduce street occupation further without losing connectivity score, so it stops at ~69%.

### 2. Anchor Compliance Collapse (~30%)

**Problem:** AnchorBudgetLoss is normalized by total street structure.

```python
L_anchor = violation.sum() / (struct_in_street.sum() + epsilon)
```

**Mechanism:** The system can change the denominator (add more street structure) instead of reducing violations. This makes anchor placement negotiable rather than mandatory.

**Why it collapses:** Adding illegal ground structure reduces the loss if it grows the denominator faster than the numerator.

### 3. Support Ratio Explosion (11:1)

**Problem:** SupportRatioLoss measures ground contact mass, not structural load paths.

**Mechanism:** The system can satisfy connectivity with minimal tendons and dump mass aloft. There's no requirement that elevated mass actually be supported by what's below it.

**Why it explodes:** The loss doesn't encode structural causality - it just counts voxels.

### 4. Street Connectivity Saturation (~100%)

**Problem:** Connectivity is seeded from domain boundaries, not functional access points.

**Mechanism:** A thin perimeter corridor keeps the metric perfect regardless of internal street obstruction.

**Why it saturates:** The model only needs edge-to-edge void connectivity, not access-to-access circulation.

---

## Why Weight Tuning Cannot Fix This

| Proposed Fix | Why It Fails |
|--------------|--------------|
| Increase anchor weight | Still global ratio, still gameable via denominator |
| Increase void weight | Creates "do nothing" attractor again |
| Add hard mask | Removes learning, doesn't fix incentives |
| Adjust support ratio | Still measures mass, not load paths |

The attractor will shift but not disappear. A new plateau will form at different metric values.

---

## The Four Conditions Creating This Attractor

Per reviewer analysis, the attractor is mathematically unavoidable while:

1. **Legality is global and weak** - Scalar ratios instead of per-voxel rules
2. **Growth incentives are local and strong** - Spatially targeted rewards
3. **Functional connectivity is incorrectly defined** - Boundary seeds, not access seeds
4. **Structural realism is not causally encoded** - Mass counting, not load paths

Until ALL FOUR conditions are corrected, the system will find a similar exploit.

---

## Training Curve Evidence

### Anchor Compliance Timeline

```
Epoch 0-100:   ~100% (no structure exists)
Epoch 100-150: Sharp drop to 20-40% (growth onset)
Epoch 150-400: Oscillates at low values (stable attractor)
```

The drop at epoch 100-150 marks when the model "discovered" the exploit. It found that growing at ground level (even illegally) is rewarded more than it's punished.

### Fill Ratio Timeline

```
Epoch 0-100:   ~0% (no growth)
Epoch 100-150: Jumps to 5-10%
Epoch 150-400: Continues to 15-17%
```

Growth correlates with anchor compliance collapse - they're the same event.

---

## Conclusion

This is an **ontology problem**, not a tuning problem.

The loss formulation allows the model to trade legality for growth because the incentive structures are misaligned. The NCA is behaving rationally within a poorly-specified objective.

**The specification is wrong.**

---

## Next Steps

See `STEP_B_V2_ONTOLOGY_FIXES.md` for proposed restructuring of the loss ontology.

---

*Step B v1.1 Failure Analysis - December 2025*
