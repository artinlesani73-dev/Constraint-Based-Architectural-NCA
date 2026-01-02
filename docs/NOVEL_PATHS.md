# Novel Research Paths

## Future Directions Aligned with Project Definition

**Status:** Supplementary Documentation (Not Core)
**Date:** December 2025

---

## Purpose

This document outlines potential future research directions that extend the core project. These paths are documented for reference but are **not part of the initial implementation**. Paths are included only if they:

1. Do not increase initial complexity
2. Are fully aligned with the project definition
3. Build naturally on the core framework

---

## Path A: Constraint Complexity Scaling

### Description

Progressively increase constraint sophistication within the existing framework.

### Alignment with Core

This path directly extends the defined constraints without changing the architecture or approach.

### Implementation Levels

**Level 1: Binary Constraints (Current)**
- Constraints are either satisfied or violated
- Single threshold per constraint
- Current implementation

**Level 2: Soft Constraints**
- Degrees of satisfaction (0-100%)
- Weighted satisfaction scores
- Graceful degradation near thresholds

```python
# Example: Soft void ratio
def soft_void_satisfaction(void_ratio, min_target=0.7, ideal_target=0.8):
    if void_ratio >= ideal_target:
        return 1.0
    elif void_ratio >= min_target:
        return 0.5 + 0.5 * (void_ratio - min_target) / (ideal_target - min_target)
    else:
        return 0.5 * void_ratio / min_target
```

**Level 3: Parameterized Constraints**
- User-adjustable thresholds at inference time
- Condition NCA on constraint parameters
- Same model, different constraint tightness

```python
# Conditioning on constraint parameters
constraint_params = torch.tensor([
    0.7,   # min void ratio
    0.25,  # max volume ratio
    3,     # max cantilever
])
grown = model.grow(seed, steps=70, constraints=constraint_params)
```

### Complexity Assessment

- Level 2: Low additional complexity
- Level 3: Medium complexity (requires conditioning mechanism)

### Recommendation

Implement Level 2 after core is stable. Level 3 is future work.

---

## Path B: Multi-Solution Exploration

### Description

Generate multiple valid solutions for the same input, exploring the design space.

### Alignment with Core

Directly supports the "generative tool" aspect of the project definition. Same constraints, diverse outputs.

### Implementation

```python
def explore_solutions(model, scene, n_solutions=10):
    """Generate diverse solutions through stochastic variation."""
    solutions = []

    for i in range(n_solutions):
        # Different random seed affects stochastic updates
        torch.manual_seed(i * 42)

        with torch.no_grad():
            solution = model.grow(scene.clone(), steps=70)

        # Verify validity
        if is_valid(solution):
            solutions.append({
                'structure': solution,
                'seed': i,
                'metrics': evaluate(solution)
            })

    return solutions


def measure_diversity(solutions):
    """Quantify how different solutions are from each other."""
    n = len(solutions)
    pairwise_iou = []

    for i in range(n):
        for j in range(i+1, n):
            iou = compute_iou(solutions[i]['structure'], solutions[j]['structure'])
            pairwise_iou.append(iou)

    return {
        'mean_similarity': np.mean(pairwise_iou),
        'diversity_score': 1 - np.mean(pairwise_iou),
        'n_valid_solutions': n,
    }
```

### Complexity Assessment

Low additional complexity. Uses existing trained model.

### Recommendation

Implement after core training is complete. Natural extension for demonstration.

---

## Path C: Constraint Transfer Learning

### Description

Study what knowledge transfers between constraints when pre-training and fine-tuning.

### Alignment with Core

Directly addresses RQ2 (constraint interactions) from a learning transfer perspective.

### Experimental Design

| Experiment | Pre-train | Fine-tune | Question |
|------------|-----------|-----------|----------|
| T1 | C1 (Structural) | C4 (Access) | Does structural help access? |
| T2 | C2 (Openness) | C3 (Massing) | Does openness help massing? |
| T3 | Easy scenes | Hard scenes | Does difficulty transfer? |
| T4 | Few constraints | All constraints | Does partial help full? |

### Metrics

```python
def measure_transfer(pretrained_model, finetuned_model, scratch_model, test_scenes):
    """
    Measure transfer learning benefit.

    transfer_ratio > 1.0 means positive transfer
    transfer_ratio < 1.0 means negative transfer
    """
    pretrain_perf = evaluate(pretrained_model, test_scenes)
    finetune_perf = evaluate(finetuned_model, test_scenes)
    scratch_perf = evaluate(scratch_model, test_scenes)

    # How much faster/better is finetuning vs scratch?
    transfer_ratio = finetune_perf / scratch_perf

    return transfer_ratio
```

### Complexity Assessment

Medium complexity. Requires multiple training runs.

### Recommendation

Document findings in ablation study. Not required for core.

---

## Path D: Interactive Constraint Editing (Future Work)

### Description

Real-time constraint manipulation with immediate NCA response.

### Alignment with Core

Supports "generative tool" aspect but requires interface development.

### Conceptual Design

```
User Interface:
┌─────────────────────────────────────────┐
│  [3D Viewport]                          │
│                                         │
│  Buildings: [editable]                  │
│  Access Points: [draggable]             │
│                                         │
├─────────────────────────────────────────┤
│  Constraints:                           │
│  [ ] Void Ratio: [===|====] 70%        │
│  [ ] Max Volume: [====|===] 25%        │
│  [ ] Cantilever: [==|=====] 3          │
│                                         │
│  [Regenerate] [Export]                  │
└─────────────────────────────────────────┘
```

### Complexity Assessment

High complexity. Requires real-time inference, UI development.

### Recommendation

Future work after core publication. Document concept only.

---

## Path E: Resolution Scaling

### Description

Increase output resolution beyond 32³ for more detailed results.

### Alignment with Core

Same constraints at higher fidelity.

### Approaches

**Approach 1: Direct Training at Higher Resolution**
- Train at 48³ or 64³
- Higher memory requirements
- Longer training time

**Approach 2: Hierarchical Generation**
```
8³ NCA   → Global massing (coarse)
   ↓ upsample + condition
32³ NCA  → Structural elements (medium)
   ↓ upsample + condition
128³ NCA → Details (fine)
```

**Approach 3: Super-Resolution Post-Processing**
- Train at 32³
- Apply learned upsampling network
- Separate model for refinement

### Complexity Assessment

- Approach 1: Low complexity, high compute
- Approach 2: High complexity
- Approach 3: Medium complexity

### Recommendation

Start with Approach 1 (48³) after core is validated. Others are future work.

---

## Path F: Simplified Structural Feedback

### Description

Add basic structural analysis feedback without full FEM.

### Alignment with Core

Extends Constraint 1 (Structural Soundness) with physics awareness.

### Simplified Load Path Loss

```python
def load_path_loss(structure):
    """
    Encourage vertical load paths without full FEM.

    Idea: Structure should have more mass at lower levels
    to support upper mass.
    """
    D = structure.shape[0]

    # Mass per layer
    mass_per_layer = structure.sum(dim=(1, 2))

    # Cumulative mass from top
    cumulative_mass = torch.cumsum(mass_per_layer.flip(0), dim=0).flip(0)

    # Each layer should have enough local mass to support above
    # Simplified: lower layers should be denser than upper
    layer_weights = torch.arange(D, 0, -1, device=structure.device).float() / D

    support_ratio = mass_per_layer / (cumulative_mass + 1e-8)
    loss = (layer_weights * (1 - support_ratio)).mean()

    return loss
```

### Complexity Assessment

Low additional complexity for simplified version.

### Recommendation

Consider as optional enhancement to C1 after core works.

---

## Summary Table

| Path | Description | Complexity | Alignment | Recommendation |
|------|-------------|------------|-----------|----------------|
| A | Constraint scaling | Low-Medium | High | Level 2 post-core |
| B | Multi-solution | Low | High | Post-core demo |
| C | Transfer learning | Medium | High | Ablation study |
| D | Interactive editing | High | Medium | Future work |
| E | Resolution scaling | Medium-High | High | 48³ post-core |
| F | Structural feedback | Low | High | Optional C1 enhancement |

---

## Implementation Priority

**Phase 1 (Core):** None of these paths

**Phase 2 (Post-Core):**
1. Path B: Multi-solution exploration
2. Path A Level 2: Soft constraints
3. Path E: 48³ resolution

**Phase 3 (Future Work):**
1. Path C: Transfer learning study
2. Path F: Simplified structural feedback
3. Path A Level 3: Parameterized constraints

**Beyond Scope:**
1. Path D: Interactive editing

---

*Novel research paths documentation*

*December 2025*
