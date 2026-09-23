# H1 growth stability: pilot

Run `20260923T145912Z_17b1d18f5b10`. No optimizer updates.

Original, F1 and F2 models on two development scenes. Six fixed growth durations and three firing seeds in the full study. Each horizon restarts the same seed, so neighboring samples share their random prefix. Pilot uses two F2 models and firing seed2 only. F1/original gradients reuse verified A2 evidence.

## Growth and budget across firing seeds

| Model | Growth steps | Cases | Connected | In budget | Joint | Mass range |
|---|---:|---:|---:|---:|---:|---|
| F2-mass_3-r0 | 16 | 1 | 1 | 0 | 0 | 22.181%–22.181% |
| F2-mass_3-r0 | 24 | 1 | 1 | 0 | 0 | 28.641%–28.641% |
| F2-mass_3-r0 | 32 | 1 | 1 | 0 | 0 | 30.526%–30.526% |
| F2-mass_3-r0 | 40 | 1 | 1 | 0 | 0 | 31.307%–31.307% |
| F2-mass_3-r0 | 50 | 1 | 1 | 0 | 0 | 31.633%–31.633% |
| F2-mass_3-r0 | 64 | 1 | 1 | 0 | 0 | 31.779%–31.779% |
| F2-mass_3-r1 | 16 | 1 | 1 | 0 | 0 | 22.885%–22.885% |
| F2-mass_3-r1 | 24 | 1 | 1 | 0 | 0 | 30.179%–30.179% |
| F2-mass_3-r1 | 32 | 1 | 1 | 0 | 0 | 32.173%–32.173% |
| F2-mass_3-r1 | 40 | 1 | 1 | 0 | 0 | 32.979%–32.979% |
| F2-mass_3-r1 | 50 | 1 | 1 | 0 | 0 | 33.423%–33.423% |
| F2-mass_3-r1 | 64 | 1 | 1 | 0 | 0 | 33.902%–33.902% |

Strict material>0.5 connectivity; continuous3%-12% mass/envelope budget, tolerance1e-6. Consecutive sampled successes would not prove stability between samples or beyond64steps. No fresh holdout scenes or statistical independence claim. All metrics, losses, transitions and individual failures are in the JSON and raw evidence.

## Actual parameter-gradient tradeoffs

| Model | Growth | Reused | Access value | Access norm x15 | Coverage norm x25 | Sparsity weighted norm | Access/sparsity cosine | Coverage/sparsity | Total/sparsity |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| F2-mass_3-r0 | 16 | False | 0.349134 | 116.097 | 135.385 | 194.636 | -0.894826 | -0.888384 | -0.332802 |
| F2-mass_3-r0 | 50 | False | 0 | 0 | 2.17117 | 365.693 | undefined (zero norm) | -0.574844 | 0.999658 |
| original-r0 | 16 | True | 1 | 0 | 53.988 | 1793.18 | undefined (zero norm) | -0.60447 | 0.998706 |
| original-r0 | 50 | True | 1 | 0 | 3.89846 | 38282.4 | undefined (zero norm) | -0.438619 | 0.999998 |
| original-r1 | 16 | True | 1 | 0 | 66.8012 | 146.601 | undefined (zero norm) | -0.671545 | 0.901707 |
| original-r1 | 50 | True | 1 | 0 | 3.09305 | 8958.11 | undefined (zero norm) | -0.491788 | 0.999999 |
| F1-mapped_30-r0 | 16 | True | 0.914526 | 173.242 | 130.225 | 137.208 | -0.826537 | -0.873512 | -0.73145 |
| F1-mapped_30-r0 | 50 | True | 1 | 0 | 1.92157 | 8119.51 | undefined (zero norm) | 0.0140262 | 1 |
| F1-mapped_30-r1 | 16 | True | 0.788892 | 280.418 | 111.782 | 224.341 | -0.772044 | -0.890251 | -0.415786 |
| F1-mapped_30-r1 | 50 | True | 0 | 0 | 2.31221 | 1799.04 | undefined (zero norm) | -0.621306 | 0.999999 |
| F1-mass_3-r0 | 16 | True | 0.560787 | 110.558 | 125.347 | 118.954 | -0.906823 | -0.898569 | -0.828173 |
| F1-mass_3-r0 | 50 | True | 0 | 0 | 3.39968 | 113.618 | undefined (zero norm) | -0.190989 | 0.999564 |
| F1-mass_3-r1 | 16 | True | 0.508366 | 173.653 | 114.363 | 131.8 | -0.850819 | -0.900735 | -0.753803 |
| F1-mass_3-r1 | 50 | True | 0 | 0 | 3.25535 | 88.983 | undefined (zero norm) | -0.00683023 | 0.99927 |

Negative cosine means the two local gradient-descent directions conflict. Above the upper budget, sparsity is monotone in continuous mass; this does not forecast an Adam update or prove the cause of learned growth. Weights, gradients and raw-field derivatives are distinct. Zero gradients are retained.

## Verification

```json
{
  "source_hashes_verified": 31,
  "growth_fields_rescored": 12,
  "sampled_transitions_verified": 10,
  "exact_historical_anchors": 4,
  "new_gradient_cases": 2,
  "reused_gradient_cases": 12,
  "parameter_vectors_verified": 84,
  "last_raw_vectors_verified": 84,
  "cosines_verified": 504,
  "gradient_forward_fields_exact": 14,
  "frozen_model_weight_checks": 2,
  "elapsed_caps_met": true,
  "optimizer_updates": 0,
  "admission": {
    "estimated_seconds": 989.4872613011103,
    "safety_factor": 1.5,
    "cap_seconds": 1500,
    "max_growth_worker_seconds": 12.506458500021836,
    "max_gradient_worker_seconds": 35.55805240001064,
    "admitted": true
  },
  "scope": "Shared formulas rescore all fields; independent BFS evaluates connectivity. Backprop vectors recorded; norms/cosines recomputed, not independent differentiation."
}
```

No model, stopping rule or objective is promoted from this diagnostic. No production, paid-compute or Drive action.
