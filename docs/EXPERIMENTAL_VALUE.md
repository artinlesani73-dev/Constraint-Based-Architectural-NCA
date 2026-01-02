# Experimental Value Enhancement

## Additional Methods to Increase Scientific Rigor

**Status:** Supplementary Documentation (Not Core)
**Date:** December 2025

---

## Purpose

This document outlines additional experimental methodologies that can enhance the scientific value of the project. These are **not required for the core implementation** but provide paths to stronger empirical validation and publishable results.

---

## 1. Quantitative Metrics Framework

### 1.1 Per-Constraint Metrics

Each constraint should have clearly defined, measurable metrics:

| Constraint | Primary Metric | Secondary Metric | Formula |
|------------|---------------|------------------|---------|
| Floating Voxels | `float_ratio` | `max_disconnect` | `disconnected / total` |
| Cantilever | `max_overhang` | `overhang_count` | Max unsupported span |
| Ground Openness | `void_ratio_0` | `min_clearance` | `1 - mean(structure[0:3])` |
| Massing | `volume_ratio` | `density_variance` | `structure.sum() / available.sum()` |
| Facade Coverage | `facade_blocked` | `facade_distribution` | `(structure * facade).sum()` |
| Access Connectivity | `path_coverage` | `path_length` | Flood-fill coverage |

### 1.2 Aggregate Metrics

```python
def compute_validity_score(metrics):
    """Single score combining all constraints."""
    scores = [
        1.0 if metrics['connectivity_rate'] > 0.95 else metrics['connectivity_rate'],
        1.0 if metrics['cantilever_ok'] else 0.5,
        metrics['ground_void_ratio'] / 0.7,  # Normalize to target
        1.0 - (metrics['volume_ratio'] / 0.25),  # Invert
        1.0 - (metrics['facade_coverage'] / 0.4),  # Invert
        metrics['path_connectivity'],
    ]
    return sum(scores) / len(scores)
```

---

## 2. Ablation Studies

### 2.1 Constraint Ablation Matrix

Systematically test each constraint combination:

| Experiment | C1A | C1B | C2A | C3A | C3B | C4A | Purpose |
|------------|-----|-----|-----|-----|-----|-----|---------|
| Base | X | | | | | | Connectivity only |
| +Cant | X | X | | | | | Full structural |
| +Open | X | X | X | | | | Add openness |
| +Mass | X | X | | X | X | | Add massing (no open) |
| +Access | X | X | | | | X | Add access (no mass) |
| Full | X | X | X | X | X | X | All constraints |

### 2.2 Interaction Analysis

For each constraint pair, measure:

```python
def measure_interaction(model_A, model_B, model_AB, test_scenes):
    """
    Measure if constraints A and B interact positively or negatively.

    Returns:
        synergy_score: > 0 means synergy, < 0 means conflict
    """
    metric_A = evaluate(model_A, test_scenes)  # Trained with A only
    metric_B = evaluate(model_B, test_scenes)  # Trained with B only
    metric_AB = evaluate(model_AB, test_scenes)  # Trained with A+B

    expected_AB = (metric_A + metric_B) / 2
    actual_AB = metric_AB

    synergy_score = actual_AB - expected_AB
    return synergy_score
```

### 2.3 Curriculum Order Study

Test different constraint introduction orders:

| Order | Sequence | Hypothesis |
|-------|----------|------------|
| Original | Struct → Open → Mass → Access | Structural foundation first |
| Access-first | Struct → Access → Open → Mass | Connectivity early |
| Mass-first | Struct → Mass → Open → Access | Volume control early |
| All-at-once | All simultaneously | Baseline comparison |

---

## 3. Baseline Comparisons

### 3.1 Required Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Random | Random voxels in available space | Lower bound |
| Rule-based | Straight bridge between access | Simple heuristic |
| Greedy | Greedy flood-fill from access | Algorithmic baseline |

### 3.2 Implementation

```python
class RandomBaseline:
    """Random voxel generation in available space."""

    def generate(self, scene, fill_ratio=0.15):
        available = 1 - scene['existing']
        random_fill = torch.rand_like(available) < fill_ratio
        return random_fill.float() * available


class RuleBasedBridge:
    """Connect access points with straight bridges."""

    def generate(self, scene):
        access = scene['access']
        access_positions = torch.where(access > 0.5)

        # Find bounding box of access points
        z_range = (access_positions[0].min(), access_positions[0].max())
        y_range = (access_positions[1].min(), access_positions[1].max())
        x_range = (access_positions[2].min(), access_positions[2].max())

        # Create bridge
        structure = torch.zeros_like(access)
        z = z_range[0]  # Single level
        structure[z, y_range[0]:y_range[1]+1, x_range[0]:x_range[1]+1] = 1

        return structure


class GreedyFloodFill:
    """Greedy expansion from access points."""

    def generate(self, scene, max_volume=0.2):
        access = scene['access']
        available = 1 - scene['existing']

        structure = access.clone()
        target_volume = available.sum() * max_volume

        while structure.sum() < target_volume:
            dilated = F.max_pool3d(structure.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze()
            candidates = dilated * available * (1 - structure)

            if candidates.sum() < 1:
                break

            # Add layer of candidates
            structure = torch.max(structure, candidates)

        return structure
```

### 3.3 Baseline Comparison Table

Report format for publication:

| Method | Connectivity | Void Ratio | Volume | Path Conn | Valid% |
|--------|-------------|------------|--------|-----------|--------|
| Random | -- | -- | -- | -- | -- |
| Rule-based | -- | -- | -- | -- | -- |
| Greedy | -- | -- | -- | -- | -- |
| **NCA (Ours)** | -- | -- | -- | -- | -- |

---

## 4. Generalization Testing

### 4.1 Test Categories

| Category | Description | Samples |
|----------|-------------|---------|
| In-distribution | Same difficulty as training | 100 |
| Novel easy | New easy configurations | 50 |
| Novel hard | New hard configurations | 50 |
| Extreme | Edge cases (very narrow, very tall) | 25 |
| Random | Completely random configurations | 100 |

### 4.2 Generalization Metrics

```python
def compute_generalization_gap(train_results, test_results):
    """
    Measure how much performance degrades on unseen data.
    """
    train_valid = np.mean([r['valid'] for r in train_results])
    test_valid = np.mean([r['valid'] for r in test_results])

    gap = train_valid - test_valid
    relative_gap = gap / train_valid

    return {
        'absolute_gap': gap,
        'relative_gap': relative_gap,
        'generalizes_well': relative_gap < 0.10  # Within 10%
    }
```

---

## 5. Statistical Significance

### 5.1 Multiple Runs

Run each experiment multiple times with different seeds:

```python
N_RUNS = 5
SEEDS = [42, 123, 456, 789, 1011]

results_per_seed = []
for seed in SEEDS:
    torch.manual_seed(seed)
    model = train_model(seed)
    results = evaluate(model)
    results_per_seed.append(results)

mean_metric = np.mean(results_per_seed)
std_metric = np.std(results_per_seed)
```

### 5.2 Confidence Intervals

Report metrics with 95% confidence intervals:

```
Connectivity: 96.2% ± 1.3%
Valid Rate: 73.5% ± 2.8%
```

### 5.3 Statistical Tests

For comparing methods:

```python
from scipy import stats

# Compare NCA vs baseline
nca_results = [...]
baseline_results = [...]

t_stat, p_value = stats.ttest_ind(nca_results, baseline_results)
significant = p_value < 0.05
```

---

## 6. Diversity Analysis

### 6.1 Multi-Solution Generation

Generate multiple solutions for same input:

```python
def generate_diverse_solutions(model, scene, n_solutions=10):
    """Generate multiple solutions with different random seeds."""
    solutions = []
    for seed in range(n_solutions):
        torch.manual_seed(seed)
        with torch.no_grad():
            solution = model.grow(scene, steps=70)
        solutions.append(solution)
    return solutions
```

### 6.2 Diversity Metrics

```python
def compute_diversity(solutions):
    """Measure diversity among solutions."""
    n = len(solutions)

    # Pairwise IoU
    ious = []
    for i in range(n):
        for j in range(i+1, n):
            iou = compute_iou(solutions[i], solutions[j])
            ious.append(iou)

    mean_iou = np.mean(ious)
    diversity = 1 - mean_iou  # Higher = more diverse

    return {
        'mean_iou': mean_iou,
        'diversity': diversity,
        'min_iou': min(ious),
        'max_iou': max(ious),
    }
```

---

## 7. Visualization Requirements

### 7.1 Training Curves

For each phase, plot:
- Total loss over epochs
- Individual constraint losses
- Validation metrics

### 7.2 Qualitative Results

For each difficulty level, show:
- Input scene (existing buildings, access points)
- Generated pavilion (3D view)
- Plan view
- Constraint satisfaction overlay

### 7.3 Comparison Figures

Side-by-side comparison:
- NCA vs baselines
- Different curriculum orders
- Multiple solutions for same input

---

## 8. Reproducibility

### 8.1 Experiment Configuration

Document all hyperparameters:

```yaml
experiment:
  name: "constraint_ablation_01"
  seed: 42

model:
  grid_size: 32
  hidden_dim: 96
  fire_rate: 0.5

training:
  epochs_per_phase: 400
  batch_size: 4
  lr: 0.002

constraints:
  connectivity_weight: 10.0
  cantilever_weight: 5.0
  # ... etc
```

### 8.2 Environment

Record:
- PyTorch version
- CUDA version
- GPU model
- Training time per phase

---

## Summary

These experimental enhancements increase scientific value through:

1. **Rigorous metrics** - Quantifiable, comparable measures
2. **Ablation studies** - Understanding component contributions
3. **Baselines** - Context for performance claims
4. **Generalization testing** - Validation beyond training distribution
5. **Statistical significance** - Confidence in results
6. **Diversity analysis** - Understanding solution space
7. **Reproducibility** - Enabling verification

These are recommended but not required for the core implementation.

---

*Experimental value enhancement documentation*

*December 2025*
