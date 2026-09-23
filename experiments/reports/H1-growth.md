# H1 growth stability: study

Run `20260923T150119Z_f7b304516723`. No optimizer updates.

Original, F1 and F2 models on two development scenes. Six fixed growth durations and three firing seeds in the full study. Each horizon restarts the same seed, so neighboring samples share their random prefix. Pilot uses two F2 models and firing seed2 only. F1/original gradients reuse verified A2 evidence.

## Growth and budget across firing seeds

| Model | Growth steps | Cases | Connected | In budget | Joint | Mass range |
|---|---:|---:|---:|---:|---:|---|
| F1-mapped_30-r0 | 16 | 3 | 0 | 0 | 0 | 12.989%–13.293% |
| F1-mapped_30-r0 | 24 | 3 | 0 | 0 | 0 | 15.202%–15.466% |
| F1-mapped_30-r0 | 32 | 3 | 0 | 0 | 0 | 17.393%–17.451% |
| F1-mapped_30-r0 | 40 | 3 | 0 | 0 | 0 | 19.062%–19.188% |
| F1-mapped_30-r0 | 50 | 3 | 0 | 0 | 0 | 19.839%–20.160% |
| F1-mapped_30-r0 | 64 | 3 | 0 | 0 | 0 | 20.093%–20.488% |
| F1-mapped_30-r1 | 16 | 3 | 0 | 0 | 0 | 12.681%–13.260% |
| F1-mapped_30-r1 | 24 | 3 | 0 | 0 | 0 | 17.437%–17.943% |
| F1-mapped_30-r1 | 32 | 3 | 1 | 0 | 0 | 22.323%–22.806% |
| F1-mapped_30-r1 | 40 | 3 | 3 | 0 | 0 | 24.676%–25.079% |
| F1-mapped_30-r1 | 50 | 3 | 3 | 0 | 0 | 25.128%–25.465% |
| F1-mapped_30-r1 | 64 | 3 | 3 | 0 | 0 | 25.134%–25.714% |
| F1-mass_3-r0 | 16 | 3 | 0 | 0 | 0 | 18.613%–18.835% |
| F1-mass_3-r0 | 24 | 3 | 3 | 0 | 0 | 25.055%–25.291% |
| F1-mass_3-r0 | 32 | 3 | 3 | 0 | 0 | 27.703%–28.006% |
| F1-mass_3-r0 | 40 | 3 | 3 | 0 | 0 | 28.378%–28.448% |
| F1-mass_3-r0 | 50 | 3 | 3 | 0 | 0 | 28.682%–28.725% |
| F1-mass_3-r0 | 64 | 3 | 3 | 0 | 0 | 28.895%–28.979% |
| F1-mass_3-r1 | 16 | 3 | 0 | 0 | 0 | 17.325%–18.086% |
| F1-mass_3-r1 | 24 | 3 | 3 | 0 | 0 | 25.522%–26.011% |
| F1-mass_3-r1 | 32 | 3 | 3 | 0 | 0 | 29.281%–29.543% |
| F1-mass_3-r1 | 40 | 3 | 3 | 0 | 0 | 29.989%–30.206% |
| F1-mass_3-r1 | 50 | 3 | 3 | 0 | 0 | 30.377%–30.597% |
| F1-mass_3-r1 | 64 | 3 | 3 | 0 | 0 | 30.642%–30.914% |
| F2-mapped_30-r0 | 16 | 3 | 0 | 0 | 0 | 13.884%–14.194% |
| F2-mapped_30-r0 | 24 | 3 | 0 | 0 | 0 | 17.384%–17.600% |
| F2-mapped_30-r0 | 32 | 3 | 1 | 0 | 0 | 20.573%–21.074% |
| F2-mapped_30-r0 | 40 | 3 | 3 | 0 | 0 | 22.319%–22.424% |
| F2-mapped_30-r0 | 50 | 3 | 3 | 0 | 0 | 23.118%–23.543% |
| F2-mapped_30-r0 | 64 | 3 | 3 | 0 | 0 | 23.477%–23.676% |
| F2-mapped_30-r1 | 16 | 3 | 0 | 0 | 0 | 12.857%–13.392% |
| F2-mapped_30-r1 | 24 | 3 | 1 | 0 | 0 | 18.132%–18.630% |
| F2-mapped_30-r1 | 32 | 3 | 3 | 0 | 0 | 23.166%–23.735% |
| F2-mapped_30-r1 | 40 | 3 | 3 | 0 | 0 | 25.363%–25.534% |
| F2-mapped_30-r1 | 50 | 3 | 3 | 0 | 0 | 25.716%–25.916% |
| F2-mapped_30-r1 | 64 | 3 | 3 | 0 | 0 | 25.731%–25.952% |
| F2-mass_3-r0 | 16 | 3 | 3 | 0 | 0 | 22.181%–22.271% |
| F2-mass_3-r0 | 24 | 3 | 3 | 0 | 0 | 28.524%–28.860% |
| F2-mass_3-r0 | 32 | 3 | 3 | 0 | 0 | 30.526%–30.644% |
| F2-mass_3-r0 | 40 | 3 | 3 | 0 | 0 | 31.307%–31.566% |
| F2-mass_3-r0 | 50 | 3 | 3 | 0 | 0 | 31.633%–31.779% |
| F2-mass_3-r0 | 64 | 3 | 3 | 0 | 0 | 31.670%–31.779% |
| F2-mass_3-r1 | 16 | 3 | 3 | 0 | 0 | 21.809%–22.885% |
| F2-mass_3-r1 | 24 | 3 | 3 | 0 | 0 | 30.179%–30.271% |
| F2-mass_3-r1 | 32 | 3 | 3 | 0 | 0 | 32.173%–32.432% |
| F2-mass_3-r1 | 40 | 3 | 3 | 0 | 0 | 32.979%–33.039% |
| F2-mass_3-r1 | 50 | 3 | 3 | 0 | 0 | 33.238%–33.423% |
| F2-mass_3-r1 | 64 | 3 | 3 | 0 | 0 | 33.519%–33.902% |
| original-r0 | 16 | 3 | 0 | 0 | 0 | 16.088%–16.564% |
| original-r0 | 24 | 3 | 0 | 0 | 0 | 28.270%–29.405% |
| original-r0 | 32 | 3 | 0 | 0 | 0 | 42.997%–44.185% |
| original-r0 | 40 | 3 | 0 | 0 | 0 | 48.102%–48.713% |
| original-r0 | 50 | 3 | 0 | 0 | 0 | 49.027%–49.344% |
| original-r0 | 64 | 3 | 0 | 0 | 0 | 49.336%–49.453% |
| original-r1 | 16 | 3 | 0 | 1 | 0 | 11.928%–12.421% |
| original-r1 | 24 | 3 | 0 | 0 | 0 | 20.962%–21.500% |
| original-r1 | 32 | 3 | 0 | 0 | 0 | 31.978%–32.245% |
| original-r1 | 40 | 3 | 0 | 0 | 0 | 36.241%–36.345% |
| original-r1 | 50 | 3 | 0 | 0 | 0 | 36.950%–37.167% |
| original-r1 | 64 | 3 | 0 | 0 | 0 | 37.097%–37.276% |

Strict material>0.5 connectivity; continuous3%-12% mass/envelope budget, tolerance1e-6. Consecutive sampled successes would not prove stability between samples or beyond64steps. No fresh holdout scenes or statistical independence claim. All metrics, losses, transitions and individual failures are in the JSON and raw evidence.

## Actual parameter-gradient tradeoffs

| Model | Growth | Reused | Access value | Access norm x15 | Coverage norm x25 | Sparsity weighted norm | Access/sparsity cosine | Coverage/sparsity | Total/sparsity |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| F2-mapped_30-r0 | 16 | False | 0.795073 | 221.074 | 128.789 | 304.308 | -0.77415 | -0.86698 | 0.128429 |
| F2-mapped_30-r0 | 50 | False | 0 | 0 | 705.902 | 6929.95 | undefined (zero norm) | -0.810671 | 0.997895 |
| F2-mapped_30-r1 | 16 | False | 0.720272 | 145.532 | 122.816 | 258.076 | -0.921473 | -0.882265 | 0.129598 |
| F2-mapped_30-r1 | 50 | False | 0 | 0 | 3.06462 | 702.091 | undefined (zero norm) | -0.530342 | 0.999984 |
| F2-mass_3-r0 | 16 | False | 0.349134 | 116.097 | 135.385 | 194.636 | -0.894826 | -0.888384 | -0.332802 |
| F2-mass_3-r0 | 50 | False | 0 | 0 | 2.17117 | 365.693 | undefined (zero norm) | -0.574844 | 0.999658 |
| F2-mass_3-r1 | 16 | False | 0.294863 | 144.179 | 141.929 | 227.737 | -0.847145 | -0.907569 | -0.237864 |
| F2-mass_3-r1 | 50 | False | 0 | 0 | 585.157 | 1354.68 | undefined (zero norm) | -0.563341 | 0.904672 |
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
  "growth_fields_rescored": 180,
  "sampled_transitions_verified": 150,
  "exact_historical_anchors": 20,
  "new_gradient_cases": 8,
  "reused_gradient_cases": 12,
  "parameter_vectors_verified": 120,
  "last_raw_vectors_verified": 120,
  "cosines_verified": 720,
  "gradient_forward_fields_exact": 20,
  "frozen_model_weight_checks": 10,
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
