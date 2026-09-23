# F2 pilot: access-only training

Run `20260923T135623Z_cc3691d8cc81`.

Only the access family changes. Same original checkpoint, two development scenes, one training seed, optimizer, 16-step rollout and coefficients as F1. Both access definitions and both common recipe totals are retained. F1 controls are reused only after exact short baseline parity; this is not a new 64-update old-objective run.

8 executed optimizer updates; 24 evaluations. All saved fields rescored and checkpoint metadata/counters checked. Final study checkpoints are replayed at both growth horizons. Formula rescoring shares the implementation; binary component connectivity uses independent BFS.

## Every evaluation and paired historical control

| Arm | Member | Update | Growth | Old connected | Component connected | Mass/envelope | Joint component/budget | Old access | New access | Coverage | Sparsity | Old total30 | New total30 | Old total3 | New total3 |
|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F2 | mapped_30-r0 | 0 | 16 | False | False | 0.1608797 | False | 1 | 1 | 0.78134459 | 0.25067252 | 46.756882 | 46.756882 | 39.988724 | 39.988724 |
| F2 | mapped_30-r0 | 0 | 50 | False | False | 0.49027273 | False | 1 | 1 | 0.703013 | 20.565283 | 655.20087 | 655.20087 | 99.93824 | 99.93824 |
| F2 | mapped_30-r0 | 1 | 16 | False | False | 0.14891423 | False | 1 | 1 | 0.7869758 | 0.12540495 | 42.67255 | 42.67255 | 39.286617 | 39.286617 |
| F2 | mapped_30-r0 | 1 | 50 | False | False | 0.46858674 | False | 1 | 1 | 0.70331448 | 18.226906 | 585.04266 | 585.04266 | 92.916183 | 92.916183 |
| F2 | mapped_30-r0 | 2 | 16 | False | False | 0.13741516 | False | 1 | 1 | 0.79243386 | 0.045493163 | 39.845345 | 39.845345 | 38.617031 | 38.617031 |
| F2 | mapped_30-r0 | 2 | 50 | False | False | 0.44558382 | False | 1 | 1 | 0.76971596 | 15.900723 | 516.86603 | 516.86603 | 87.546562 | 87.546562 |
| F2 | mapped_30-r1 | 0 | 16 | False | False | 0.12421025 | False | 1 | 1 | 0.8752023 | 0.002658929 | 41.049126 | 41.049126 | 40.977337 | 40.977337 |
| F2 | mapped_30-r1 | 0 | 50 | False | False | 0.37166923 | False | 1 | 1 | 0.78166318 | 9.5006104 | 324.31155 | 324.31155 | 67.795052 | 67.795052 |
| F2 | mapped_30-r1 | 1 | 16 | False | False | 0.11849251 | False | 1 | 1 | 0.8737309 | 0 | 40.429104 | 40.429104 | 40.429104 | 40.429104 |
| F2 | mapped_30-r1 | 1 | 50 | False | False | 0.36143559 | False | 1 | 1 | 0.78161132 | 8.7436714 | 301.55908 | 301.55908 | 65.479935 | 65.479935 |
| F2 | mapped_30-r1 | 2 | 16 | False | False | 0.1128661 | False | 1 | 1 | 0.87222171 | 0 | 39.811451 | 39.811451 | 39.811451 | 39.811451 |
| F2 | mapped_30-r1 | 2 | 50 | False | False | 0.33641493 | False | 1 | 1 | 0.78154951 | 7.0253134 | 249.69405 | 249.69405 | 60.01059 | 60.01059 |
| F2 | mass_3-r0 | 0 | 16 | False | False | 0.1608797 | False | 1 | 1 | 0.78134459 | 0.25067252 | 46.756882 | 46.756882 | 39.988724 | 39.988724 |
| F2 | mass_3-r0 | 0 | 50 | False | False | 0.49027273 | False | 1 | 1 | 0.703013 | 20.565283 | 655.20087 | 655.20087 | 99.93824 | 99.93824 |
| F2 | mass_3-r0 | 1 | 16 | False | False | 0.15131471 | False | 1 | 1 | 0.78122497 | 0.14709164 | 43.124065 | 43.124065 | 39.152592 | 39.152592 |
| F2 | mass_3-r0 | 1 | 50 | False | False | 0.47703132 | False | 1 | 1 | 0.70299751 | 19.120703 | 611.78949 | 611.78949 | 95.530472 | 95.530472 |
| F2 | mass_3-r0 | 2 | 16 | False | False | 0.14192179 | False | 1 | 1 | 0.78111231 | 0.07208474 | 40.259724 | 40.259724 | 38.313435 | 38.313435 |
| F2 | mass_3-r0 | 2 | 50 | False | False | 0.45739627 | False | 1 | 1 | 0.70298058 | 17.075436 | 550.36432 | 550.36432 | 89.327637 | 89.327637 |
| F2 | mass_3-r1 | 0 | 16 | False | False | 0.12421025 | False | 1 | 1 | 0.8752023 | 0.002658929 | 41.049126 | 41.049126 | 40.977337 | 40.977337 |
| F2 | mass_3-r1 | 0 | 50 | False | False | 0.37166923 | False | 1 | 1 | 0.78166318 | 9.5006104 | 324.31155 | 324.31155 | 67.795052 | 67.795052 |
| F2 | mass_3-r1 | 1 | 16 | False | False | 0.11887208 | False | 1 | 1 | 0.87336749 | 0 | 40.426712 | 40.426712 | 40.426712 | 40.426712 |
| F2 | mass_3-r1 | 1 | 50 | False | False | 0.36307469 | False | 1 | 1 | 0.78158247 | 8.8627949 | 305.11496 | 305.11496 | 65.819481 | 65.819481 |
| F2 | mass_3-r1 | 2 | 16 | False | False | 0.11337794 | False | 1 | 1 | 0.87177712 | 0 | 39.81205 | 39.81205 | 39.81205 | 39.81205 |
| F2 | mass_3-r1 | 2 | 50 | False | False | 0.33705279 | False | 1 | 1 | 0.78150833 | 7.0667872 | 250.94044 | 250.94044 | 60.137188 | 60.137188 |
| F1 | mapped_30-r0 | 0 | 16 | False | False | 0.1608797 | False | 1 | 1 | 0.78134459 | 0.25067252 | 46.756882 | 46.756882 | 39.988724 | 39.988724 |
| F1 | mapped_30-r0 | 0 | 50 | False | False | 0.49027273 | False | 1 | 1 | 0.703013 | 20.565283 | 655.20087 | 655.20087 | 99.93824 | 99.93824 |
| F1 | mapped_30-r0 | 1 | 16 | False | False | 0.14891423 | False | 1 | 1 | 0.7869758 | 0.12540495 | 42.67255 | 42.67255 | 39.286617 | 39.286617 |
| F1 | mapped_30-r0 | 1 | 50 | False | False | 0.46858674 | False | 1 | 1 | 0.70331448 | 18.226906 | 585.04266 | 585.04266 | 92.916183 | 92.916183 |
| F1 | mapped_30-r0 | 3 | 16 | False | False | 0.12676497 | False | 1 | 1 | 0.79763341 | 0.0068647242 | 38.164131 | 38.164131 | 37.978783 | 37.978783 |
| F1 | mapped_30-r0 | 3 | 50 | False | False | 0.43086484 | False | 1 | 1 | 0.79759622 | 14.495542 | 475.42578 | 475.42578 | 84.04615 | 84.04615 |
| F1 | mapped_30-r0 | 8 | 16 | False | False | 0.092379056 | False | 1 | 1 | 0.80734086 | 0 | 35.708626 | 35.708626 | 35.708626 | 35.708626 |
| F1 | mapped_30-r0 | 8 | 50 | False | False | 0.10456944 | False | 1 | 1 | 0.82065994 | 0 | 35.54298 | 35.54298 | 35.54298 | 35.54298 |
| F1 | mapped_30-r0 | 16 | 16 | False | False | 0.098932393 | False | 1 | 1 | 0.78266549 | 0 | 35.028378 | 35.028378 | 35.028378 | 35.028378 |
| F1 | mapped_30-r0 | 16 | 50 | False | False | 0.1524072 | False | 1 | 1 | 0.70260715 | 0.15753402 | 37.302357 | 37.302357 | 33.048939 | 33.048939 |
| F1 | mapped_30-r0 | 32 | 16 | False | False | 0.12509046 | False | 1 | 1 | 0.72583312 | 0.0038869292 | 33.591774 | 33.591774 | 33.486828 | 33.486828 |
| F1 | mapped_30-r0 | 32 | 50 | False | False | 0.19409198 | False | 1 | 1 | 0.69855219 | 0.82344317 | 57.182182 | 57.182182 | 34.949219 | 34.949219 |
| F1 | mapped_30-r0 | 64 | 16 | False | False | 0.12988722 | False | 1 | 0.91452569 | 0.57271141 | 0.014663585 | 29.978769 | 28.696655 | 29.582853 | 28.300739 |
| F1 | mapped_30-r0 | 64 | 50 | False | False | 0.20108192 | False | 1 | 1 | 0.40356314 | 0.98614168 | 54.688263 | 54.688263 | 28.062435 | 28.062435 |
| F1 | mapped_30-r1 | 0 | 16 | False | False | 0.12421025 | False | 1 | 1 | 0.8752023 | 0.002658929 | 41.049126 | 41.049126 | 40.977337 | 40.977337 |
| F1 | mapped_30-r1 | 0 | 50 | False | False | 0.37166923 | False | 1 | 1 | 0.78166318 | 9.5006104 | 324.31155 | 324.31155 | 67.795052 | 67.795052 |
| F1 | mapped_30-r1 | 1 | 16 | False | False | 0.11849251 | False | 1 | 1 | 0.8737309 | 0 | 40.429104 | 40.429104 | 40.429104 | 40.429104 |
| F1 | mapped_30-r1 | 1 | 50 | False | False | 0.36143559 | False | 1 | 1 | 0.78161132 | 8.7436714 | 301.55908 | 301.55908 | 65.479935 | 65.479935 |
| F1 | mapped_30-r1 | 3 | 16 | False | False | 0.10781872 | False | 1 | 1 | 0.87068748 | 0 | 39.172089 | 39.172089 | 39.172089 | 39.172089 |
| F1 | mapped_30-r1 | 3 | 50 | False | False | 0.33185902 | False | 1 | 1 | 0.78148597 | 6.7326365 | 240.88437 | 240.88437 | 59.103188 | 59.103188 |
| F1 | mapped_30-r1 | 8 | 16 | False | False | 0.090065137 | False | 1 | 1 | 0.86321837 | 0 | 37.056427 | 37.056427 | 37.056427 | 37.056427 |
| F1 | mapped_30-r1 | 8 | 50 | False | False | 0.14940347 | False | 1 | 1 | 0.78117514 | 0.12968461 | 38.439705 | 38.439705 | 34.938217 | 34.938217 |
| F1 | mapped_30-r1 | 16 | 16 | False | False | 0.10808117 | False | 1 | 1 | 0.821531 | 0 | 35.936771 | 35.936771 | 35.936771 | 35.936771 |
| F1 | mapped_30-r1 | 16 | 50 | False | False | 0.17179698 | False | 1 | 1 | 0.77963364 | 0.40243906 | 46.575214 | 46.575214 | 35.709362 | 35.709362 |
| F1 | mapped_30-r1 | 32 | 16 | False | False | 0.12509578 | False | 1 | 1 | 0.78021365 | 0.0038950574 | 34.860683 | 34.860683 | 34.755512 | 34.755512 |
| F1 | mapped_30-r1 | 32 | 50 | False | False | 0.21285172 | False | 1 | 1 | 0.77740824 | 1.2932163 | 73.265327 | 73.265327 | 38.348488 | 38.348488 |
| F1 | mapped_30-r1 | 64 | 16 | False | False | 0.13143419 | False | 1 | 0.78889155 | 0.5916906 | 0.019611105 | 30.548134 | 27.381506 | 30.018633 | 26.852005 |
| F1 | mapped_30-r1 | 64 | 50 | True | True | 0.2546533 | False | 1 | 0 | 0.3259995 | 2.7197268 | 104.7702 | 89.770203 | 31.337582 | 16.337582 |
| F1 | mass_3-r0 | 0 | 16 | False | False | 0.1608797 | False | 1 | 1 | 0.78134459 | 0.25067252 | 46.756882 | 46.756882 | 39.988724 | 39.988724 |
| F1 | mass_3-r0 | 0 | 50 | False | False | 0.49027273 | False | 1 | 1 | 0.703013 | 20.565283 | 655.20087 | 655.20087 | 99.93824 | 99.93824 |
| F1 | mass_3-r0 | 1 | 16 | False | False | 0.15131471 | False | 1 | 1 | 0.78122497 | 0.14709164 | 43.124065 | 43.124065 | 39.152592 | 39.152592 |
| F1 | mass_3-r0 | 1 | 50 | False | False | 0.47703132 | False | 1 | 1 | 0.70299751 | 19.120703 | 611.78949 | 611.78949 | 95.530472 | 95.530472 |
| F1 | mass_3-r0 | 3 | 16 | False | False | 0.1331297 | False | 1 | 1 | 0.78086317 | 0.025858367 | 38.183346 | 38.183346 | 37.485172 | 37.485172 |
| F1 | mass_3-r0 | 3 | 50 | False | False | 0.44970497 | False | 1 | 1 | 0.70296019 | 16.305805 | 527.2561 | 527.2561 | 86.999352 | 86.999352 |
| F1 | mass_3-r0 | 8 | 16 | False | False | 0.10306095 | False | 1 | 1 | 0.77826679 | 0 | 34.928329 | 34.928329 | 34.928329 | 34.928329 |
| F1 | mass_3-r0 | 8 | 50 | False | False | 0.1562404 | False | 1 | 1 | 0.70274377 | 0.19700506 | 38.503643 | 38.503643 | 33.184505 | 33.184505 |
| F1 | mass_3-r0 | 16 | 16 | False | False | 0.11510575 | False | 1 | 1 | 0.75017107 | 0 | 34.162663 | 34.162663 | 34.162663 | 34.162663 |
| F1 | mass_3-r0 | 16 | 50 | False | False | 0.16833408 | False | 1 | 1 | 0.70074153 | 0.35042757 | 43.044437 | 43.044437 | 33.582897 | 33.582897 |
| F1 | mass_3-r0 | 32 | 16 | False | False | 0.14790595 | False | 1 | 1 | 0.69168162 | 0.11681129 | 36.087505 | 36.087505 | 32.933601 | 32.933601 |
| F1 | mass_3-r0 | 32 | 50 | False | False | 0.22935678 | False | 1 | 1 | 0.66781163 | 1.793836 | 85.531029 | 85.531029 | 37.097458 | 37.097458 |
| F1 | mass_3-r0 | 64 | 16 | False | False | 0.18638769 | False | 1 | 0.5607875 | 0.3757399 | 0.66109878 | 44.377689 | 37.789501 | 26.528021 | 19.939833 |
| F1 | mass_3-r0 | 64 | 50 | True | True | 0.28724986 | False | 1 | 0 | 0.28571561 | 4.1958776 | 148.03577 | 133.03577 | 34.747082 | 19.747086 |
| F1 | mass_3-r1 | 0 | 16 | False | False | 0.12421025 | False | 1 | 1 | 0.8752023 | 0.002658929 | 41.049126 | 41.049126 | 40.977337 | 40.977337 |
| F1 | mass_3-r1 | 0 | 50 | False | False | 0.37166923 | False | 1 | 1 | 0.78166318 | 9.5006104 | 324.31155 | 324.31155 | 67.795052 | 67.795052 |
| F1 | mass_3-r1 | 1 | 16 | False | False | 0.11887208 | False | 1 | 1 | 0.87336749 | 0 | 40.426712 | 40.426712 | 40.426712 | 40.426712 |
| F1 | mass_3-r1 | 1 | 50 | False | False | 0.36307469 | False | 1 | 1 | 0.78158247 | 8.8627949 | 305.11496 | 305.11496 | 65.819481 | 65.819481 |
| F1 | mass_3-r1 | 3 | 16 | False | False | 0.10835656 | False | 1 | 1 | 0.87023705 | 0 | 39.173542 | 39.173542 | 39.173542 | 39.173542 |
| F1 | mass_3-r1 | 3 | 50 | False | False | 0.33250245 | False | 1 | 1 | 0.78143847 | 6.7735934 | 242.10278 | 242.10278 | 59.215759 | 59.215759 |
| F1 | mass_3-r1 | 8 | 16 | False | False | 0.090319909 | False | 1 | 1 | 0.86306477 | 0 | 37.057941 | 37.057941 | 37.057941 | 37.057941 |
| F1 | mass_3-r1 | 8 | 50 | False | False | 0.15081717 | False | 1 | 1 | 0.78113019 | 0.14245473 | 38.839912 | 38.839912 | 34.993633 | 34.993633 |
| F1 | mass_3-r1 | 16 | 16 | False | False | 0.1079538 | False | 1 | 1 | 0.82178098 | 0 | 35.944855 | 35.944855 | 35.944855 | 35.944855 |
| F1 | mass_3-r1 | 16 | 50 | False | False | 0.1715506 | False | 1 | 1 | 0.77961332 | 0.39861971 | 46.46011 | 46.46011 | 35.69738 | 35.69738 |
| F1 | mass_3-r1 | 32 | 16 | False | False | 0.15445204 | False | 1 | 0.98804367 | 0.74356818 | 0.17804149 | 39.189129 | 39.009785 | 34.382011 | 34.202663 |
| F1 | mass_3-r1 | 32 | 50 | False | False | 0.26147392 | False | 1 | 1 | 0.61720133 | 3.0022304 | 120.51171 | 120.51171 | 39.451488 | 39.451488 |
| F1 | mass_3-r1 | 64 | 16 | False | False | 0.18086243 | False | 1 | 0.50836623 | 0.47244629 | 0.55563533 | 43.594696 | 36.220188 | 28.592541 | 21.218035 |
| F1 | mass_3-r1 | 64 | 50 | True | True | 0.30596921 | False | 1 | 0 | 0.32350975 | 5.1876817 | 178.7399 | 163.7399 | 38.672497 | 23.672499 |

r0 is ground-pair; r1 is minimal-smoke. Strict material>0.5 connectivity; continuous mass budget3%-12%, tolerance1e-6. Both can pass without satisfying all graded terms or usable/structurally safe architecture. All nine terms, three regularizers, binary metrics and candidate critical voxels are preserved in the machine-readable report.

## Every training update

| Member | Update | Loss before update | Gradient norm before clip | Access used | Seconds incl. evidence |
|---|---:|---:|---:|---:|---:|
| mapped_30-r0 | 1 | 48.519341 | 2299.8479 | 1 | 3.329 |
| mapped_30-r0 | 2 | 43.578056 | 1672.9475 | 1 | 2.824 |
| mapped_30-r1 | 1 | 40.577156 | 233.0123 | 1 | 2.754 |
| mapped_30-r1 | 2 | 40.779675 | 220.77342 | 1 | 2.735 |
| mass_3-r0 | 1 | 40.084175 | 402.05582 | 1 | 2.648 |
| mass_3-r0 | 2 | 39.454857 | 401.81186 | 1 | 2.461 |
| mass_3-r1 | 1 | 40.562778 | 189.51811 | 1 | 2.579 |
| mass_3-r1 | 2 | 40.779331 | 219.43849 | 1 | 2.540 |

Training fields precede the update; checkpoints follow it. Evaluation follows its named update. CPU recovery covers completed update boundaries, not GPU/AMP or abrupt writes. Three early recovery steps may not exercise a nonzero access parameter gradient; do not overstate that gate.

## Verification

```json
{
  "baseline_parity": true,
  "training_update_exact_F1_matches": 12,
  "evaluation_record_exact_F1_matches": 24,
  "prefix_checkpoint": true,
  "resumed_traces": true,
  "resumed_checkpoint": true,
  "resumed_fields": true,
  "repeat_traces": true,
  "repeat_checkpoint": true,
  "repeat_fields": true,
  "resumed_evaluation_traces": true,
  "resumed_evaluation_fields": true,
  "repeat_evaluation_traces": true,
  "repeat_evaluation_fields": true,
  "source_hashes_verified": 32,
  "saved_fields_rescored": 32,
  "training_checkpoints_verified": 8,
  "evaluation_metrics_verified": 24,
  "final_checkpoint_rollouts_replayed": 0,
  "F1_controls_rescored_both_definitions": 56,
  "initial_fields_exactly_match_F1": 8,
  "successful_workers": 4,
  "cost_admission": {
    "p90_update_seconds": 2.9753617700014727,
    "max_evaluation_pair_seconds": 3.853160400001798,
    "startup_allowance_seconds": 5.0,
    "safety_factor": 1.5,
    "estimated_member_seconds": 333.59291412016023,
    "estimated_total_seconds": 1334.371656480641,
    "admitted": true
  }
}
```

One seed/two development scenes cannot establish generalization. Semantic rescoring alone is not geometry improvement. No production model or coefficient is promoted by this diagnostic.
