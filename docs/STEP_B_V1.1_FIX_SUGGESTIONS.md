# Step B v1.1 Fix Suggestions

**Date:** December 2025
**Status:** Proposed (Pre-Review)
**Context:** Model grows but creates ground carpet violations

---

## Observed Failure Mode

| Metric | Result | Target |
|--------|--------|--------|
| Street Void | 69.4% | >70% |
| Anchor Compliance | 31.9% | 100% |
| Fill Ratio | 17.1% | 5-15% |
| Support Ratio | 11.2:1 | <4:1 |

The model grows (good) but in wrong places (bad). Structure spreads at ground level outside anchor zones.

---

## Initial Fix Suggestions (Weight-Based)

### Option 1: Increase Anchor Weight

```python
'anchor_budget': 50.0,  # Was 20, make violations very expensive
```

**Rationale:** Anchor compliance dropped because growth incentives (Dice=25) overwhelmed anchor penalty (20).

### Option 2: Reformulate Support Ratio

Current formula counts ANY ground contact:
```python
ground_contact = structure[:, :2, :, :].sum()
elevated = structure[:, 2:, :, :].sum()
ratio = elevated / ground_contact
```

Problem: Encourages ANY ground contact to reduce ratio, including illegal contact.

Proposed: Only count ANCHOR-BASED contact:
```python
anchor_contact = (structure[:, :2, :, :] * anchors[:, :2, :, :]).sum()
elevated = structure[:, 2:, :, :].sum()
ratio = elevated / anchor_contact
```

**Rationale:** This removes incentive to grow outside anchors.

### Option 3: Hard Mask at Ground Level

Force structure at z<2 to ONLY exist in anchors:
```python
# In NCA forward pass
ground_struct = grown_new[:, 0:1, :street_levels, :, :]
anchors = state[:, ch_anchors:ch_anchors+1, :street_levels, :, :]
ground_struct_masked = ground_struct * anchors  # Hard constraint
```

**Rationale:** Makes anchor compliance 100% by construction, not optimization.

---

## Analysis of Why v1.1 Failed

### Training Curve Pattern

- Epoch 0-100: No growth, 100% anchor compliance
- Epoch 100-150: Growth starts, anchor compliance DROPS to 20-40%
- Epoch 150-400: Stabilizes at low compliance

### The Exploitation Strategy

The model discovered:
1. Grow at ground level (easiest connectivity path)
2. Build tall from ground (elevated bonus)
3. Accept anchor violations (penalty not strong enough)

Result: Tall structures on illegal ground footprint.

### Weight Imbalance

| Force Type | Losses | Combined Weight |
|------------|--------|-----------------|
| Growth | Dice (25) + Sparsity-under (20×3) + Elevated (5) | ~90 |
| Protection | Void (25) + Anchor (20) | ~45 |

Growth forces are 2x stronger than protection.

---

## Recommended Implementation

Combine Options 1 + 2:

1. **Increase anchor weight:** 20 → 50
2. **Reformulate support ratio:** Only anchor contact counts

This makes illegal ground growth expensive AND removes incentive to add more illegal contact.

---

*Note: These are surface-level weight/formula fixes. Deeper architectural issues may exist.*
