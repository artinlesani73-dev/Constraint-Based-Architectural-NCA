# F1 pilot: repeated single-scene fitting

Run `20260923T112500Z_8e1a30c320b8`.

Unchanged K2 update rule and original checkpoint; independent models repeat one development scene. One training seed, weak scaffold initialization, 16 growth steps per optimizer update. The only paired coefficient change is sparsity30 versus3. Evaluations use firing seed2, separate from training.

8 executed optimizer updates; 24 evaluations. All saved objectives and evaluation metrics rescored; complete checkpoint metadata/counters checked. Verification uses the shared formulas, not an independent implementation. Final study rollouts are replayed from all four final checkpoints at both horizons.

## Every evaluation boundary

| Member | Update | Growth steps | Connected | Mass/envelope | Coverage | Access | Sparsity | Support | Total weight30 | Total weight3 | Guide raw<0 / count |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| mapped_30-r0 | 0 | 16 | False | 0.1608797 | 0.7813446 | 1 | 0.2506725 | 0.311887 | 46.75688 | 39.98872 | 16 / 36 |
| mapped_30-r0 | 0 | 50 | False | 0.4902727 | 0.703013 | 1 | 20.56528 | 0.002008129 | 655.2009 | 99.93824 | 17 / 36 |
| mapped_30-r0 | 1 | 16 | False | 0.1489142 | 0.7869758 | 1 | 0.125405 | 0.296877 | 42.67255 | 39.28662 | 16 / 36 |
| mapped_30-r0 | 1 | 50 | False | 0.4685867 | 0.7033145 | 1 | 18.22691 | 0.002611555 | 585.0427 | 92.91618 | 17 / 36 |
| mapped_30-r0 | 2 | 16 | False | 0.1374152 | 0.7924339 | 1 | 0.04549316 | 0.2763137 | 39.84534 | 38.61703 | 16 / 36 |
| mapped_30-r0 | 2 | 50 | False | 0.4455838 | 0.769716 | 1 | 15.90072 | 0.00141203 | 516.866 | 87.54656 | 18 / 36 |
| mapped_30-r1 | 0 | 16 | False | 0.1242102 | 0.8752023 | 1 | 0.002658929 | 0.2779198 | 41.04913 | 40.97734 | 18 / 32 |
| mapped_30-r1 | 0 | 50 | False | 0.3716692 | 0.7816632 | 1 | 9.50061 | 0.001576355 | 324.3116 | 67.79505 | 13 / 32 |
| mapped_30-r1 | 1 | 16 | False | 0.1184925 | 0.8737309 | 1 | 0 | 0.2592744 | 40.4291 | 40.4291 | 18 / 32 |
| mapped_30-r1 | 1 | 50 | False | 0.3614356 | 0.7816113 | 1 | 8.743671 | 0.002878451 | 301.5591 | 65.47993 | 13 / 32 |
| mapped_30-r1 | 2 | 16 | False | 0.1128661 | 0.8722217 | 1 | 0 | 0.235522 | 39.81145 | 39.81145 | 18 / 32 |
| mapped_30-r1 | 2 | 50 | False | 0.3364149 | 0.7815495 | 1 | 7.025313 | 0.002444139 | 249.694 | 60.01059 | 13 / 32 |
| mass_3-r0 | 0 | 16 | False | 0.1608797 | 0.7813446 | 1 | 0.2506725 | 0.311887 | 46.75688 | 39.98872 | 16 / 36 |
| mass_3-r0 | 0 | 50 | False | 0.4902727 | 0.703013 | 1 | 20.56528 | 0.002008129 | 655.2009 | 99.93824 | 17 / 36 |
| mass_3-r0 | 1 | 16 | False | 0.1513147 | 0.781225 | 1 | 0.1470916 | 0.2931893 | 43.12407 | 39.15259 | 16 / 36 |
| mass_3-r0 | 1 | 50 | False | 0.4770313 | 0.7029975 | 1 | 19.1207 | 0.002585045 | 611.7895 | 95.53047 | 17 / 36 |
| mass_3-r0 | 2 | 16 | False | 0.1419218 | 0.7811123 | 1 | 0.07208474 | 0.2688641 | 40.25972 | 38.31343 | 16 / 36 |
| mass_3-r0 | 2 | 50 | False | 0.4573963 | 0.7029806 | 1 | 17.07544 | 0.002084581 | 550.3643 | 89.32764 | 17 / 36 |
| mass_3-r1 | 0 | 16 | False | 0.1242102 | 0.8752023 | 1 | 0.002658929 | 0.2779198 | 41.04913 | 40.97734 | 18 / 32 |
| mass_3-r1 | 0 | 50 | False | 0.3716692 | 0.7816632 | 1 | 9.50061 | 0.001576355 | 324.3116 | 67.79505 | 13 / 32 |
| mass_3-r1 | 1 | 16 | False | 0.1188721 | 0.8733675 | 1 | 0 | 0.2595649 | 40.42671 | 40.42671 | 18 / 32 |
| mass_3-r1 | 1 | 50 | False | 0.3630747 | 0.7815825 | 1 | 8.862795 | 0.002922911 | 305.115 | 65.81948 | 13 / 32 |
| mass_3-r1 | 2 | 16 | False | 0.1133779 | 0.8717771 | 1 | 0 | 0.2360944 | 39.81205 | 39.81205 | 18 / 32 |
| mass_3-r1 | 2 | 50 | False | 0.3370528 | 0.7815083 | 1 | 7.066787 | 0.002208441 | 250.9404 | 60.13719 | 13 / 32 |

r0 is ground-pair; r1 is minimal-smoke. Connected uses material>0.5. Mass is continuous; budget3%-12%, tolerance1e-6. Connectivity and budget do not certify all nine constraints, usable architecture or mechanical safety. Raw saturation counts do not prove a blocked parameter gradient.

## Complete objective values

| Member | Update | Growth | access | coverage | facade | ground | legality | sparsity | spill | support | thickness | cantilever_boundary | density_binary | tv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mapped_30-r0 | 0 | 16 | 1 | 0.7813446 | 0.2138448 | 0 | 0 | 0.2506725 | 0.002068451 | 0.311887 | 0 | 0.000342664 | 0.001688452 | 0.01105864 |
| mapped_30-r0 | 0 | 50 | 1 | 0.703013 | 0.5313832 | 0 | 0 | 20.56528 | 0.01215305 | 0.002008129 | 0 | 0.0003662109 | 3.254303e-05 | 0.03141786 |
| mapped_30-r0 | 1 | 16 | 1 | 0.7869758 | 0.1801549 | 0 | 0 | 0.125405 | 0.001730701 | 0.296877 | 0 | 0.0003416571 | 0.001480034 | 0.01002461 |
| mapped_30-r0 | 1 | 50 | 1 | 0.7033145 | 0.530942 | 0 | 0 | 18.22691 | 0.01160066 | 0.002611555 | 0 | 0.0003662109 | 3.917049e-05 | 0.03033188 |
| mapped_30-r0 | 2 | 16 | 1 | 0.7924339 | 0.1409639 | 0 | 0 | 0.04549316 | 0.001401411 | 0.2763137 | 0 | 0.000340674 | 0.001264434 | 0.009023555 |
| mapped_30-r0 | 2 | 50 | 1 | 0.769716 | 0.5285583 | 0 | 0 | 15.90072 | 0.01095875 | 0.00141203 | 0 | 0.0003662109 | 4.633638e-05 | 0.02866678 |
| mapped_30-r1 | 0 | 16 | 1 | 0.8752023 | 0.1831633 | 0 | 0 | 0.002658929 | 0.0009467871 | 0.2779198 | 0 | 0 | 0.001071655 | 0.007423214 |
| mapped_30-r1 | 0 | 50 | 1 | 0.7816632 | 0.4581932 | 0 | 0 | 9.50061 | 0.005457914 | 0.001576355 | 0 | 0 | 2.270702e-05 | 0.02057938 |
| mapped_30-r1 | 1 | 16 | 1 | 0.8737309 | 0.1481816 | 0 | 0 | 0 | 0.000793699 | 0.2592744 | 0 | 0 | 0.0009926251 | 0.006998971 |
| mapped_30-r1 | 1 | 50 | 1 | 0.7816113 | 0.4534394 | 0 | 0 | 8.743671 | 0.00523624 | 0.002878451 | 0 | 0 | 3.187988e-05 | 0.0202151 |
| mapped_30-r1 | 2 | 16 | 1 | 0.8722217 | 0.109641 | 0 | 0 | 0 | 0.0006410442 | 0.235522 | 0 | 0 | 0.0009062055 | 0.006582707 |
| mapped_30-r1 | 2 | 50 | 1 | 0.7815495 | 0.4242301 | 0 | 0 | 7.025313 | 0.004605716 | 0.002444139 | 0 | 0 | 2.625367e-05 | 0.01883553 |
| mass_3-r0 | 0 | 16 | 1 | 0.7813446 | 0.2138448 | 0 | 0 | 0.2506725 | 0.002068451 | 0.311887 | 0 | 0.000342664 | 0.001688452 | 0.01105864 |
| mass_3-r0 | 0 | 50 | 1 | 0.703013 | 0.5313832 | 0 | 0 | 20.56528 | 0.01215305 | 0.002008129 | 0 | 0.0003662109 | 3.254303e-05 | 0.03141786 |
| mass_3-r0 | 1 | 16 | 1 | 0.781225 | 0.1775293 | 0 | 0 | 0.1470916 | 0.001735581 | 0.2931893 | 0 | 0.0003421802 | 0.001519508 | 0.01022751 |
| mass_3-r0 | 1 | 50 | 1 | 0.7029975 | 0.5247554 | 0 | 0 | 19.1207 | 0.01169948 | 0.002585045 | 0 | 0.0003662109 | 3.895021e-05 | 0.03076279 |
| mass_3-r0 | 2 | 16 | 1 | 0.7811123 | 0.136814 | 0 | 0 | 0.07208474 | 0.001407313 | 0.2688641 | 0 | 0.0003417229 | 0.00133979 | 0.009406873 |
| mass_3-r0 | 2 | 50 | 1 | 0.7029806 | 0.5201666 | 0 | 0 | 17.07544 | 0.01108778 | 0.002084581 | 0 | 0.0003662109 | 3.085231e-05 | 0.02935063 |
| mass_3-r1 | 0 | 16 | 1 | 0.8752023 | 0.1831633 | 0 | 0 | 0.002658929 | 0.0009467871 | 0.2779198 | 0 | 0 | 0.001071655 | 0.007423214 |
| mass_3-r1 | 0 | 50 | 1 | 0.7816632 | 0.4581932 | 0 | 0 | 9.50061 | 0.005457914 | 0.001576355 | 0 | 0 | 2.270702e-05 | 0.02057938 |
| mass_3-r1 | 1 | 16 | 1 | 0.8733675 | 0.1486051 | 0 | 0 | 0 | 0.0007972977 | 0.2595649 | 0 | 0 | 0.000997785 | 0.007026366 |
| mass_3-r1 | 1 | 50 | 1 | 0.7815825 | 0.4516682 | 0 | 0 | 8.862795 | 0.005243943 | 0.002922911 | 0 | 0 | 3.148931e-05 | 0.02027779 |
| mass_3-r1 | 2 | 16 | 1 | 0.8717771 | 0.1103365 | 0 | 0 | 0 | 0.0006457585 | 0.2360944 | 0 | 0 | 0.0009132969 | 0.006619196 |
| mass_3-r1 | 2 | 50 | 1 | 0.7815083 | 0.4247063 | 0 | 0 | 7.066787 | 0.004617748 | 0.002208441 | 0 | 0 | 2.352817e-05 | 0.01887294 |

## Saved controls on the same scenes

| Method | Scene | Growth steps | Connected | Mass/envelope | Total weight30 | Total weight3 |
|---|---|---:|---|---:|---:|---:|
| mapped_30-s0 | ref-01-ground-pair | 16 | False | 0.08697776 | 36.33255 | 36.33255 |
| mapped_30-s0 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59434 | 35.59434 |
| mapped_30-s0 | ref-06-minimal-smoke | 16 | False | 0.06402721 | 39.31152 | 39.31152 |
| mapped_30-s0 | ref-06-minimal-smoke | 50 | False | 0.07258064 | 37.85109 | 37.85109 |
| mapped_30-s1 | ref-01-ground-pair | 16 | False | 0.08691613 | 36.33854 | 36.33854 |
| mapped_30-s1 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59698 | 35.59698 |
| mapped_30-s1 | ref-06-minimal-smoke | 16 | False | 0.06393798 | 39.32574 | 39.32574 |
| mapped_30-s1 | ref-06-minimal-smoke | 50 | False | 0.07258064 | 37.8546 | 37.8546 |
| mass_3-s0 | ref-01-ground-pair | 16 | False | 0.1429801 | 40.10003 | 37.96129 |
| mass_3-s0 | ref-01-ground-pair | 50 | False | 0.4561582 | 546.5374 | 88.87797 |
| mass_3-s0 | ref-06-minimal-smoke | 16 | False | 0.1126303 | 39.45462 | 39.45462 |
| mass_3-s0 | ref-06-minimal-smoke | 50 | False | 0.3339843 | 244.9896 | 59.54298 |
| mass_3-s1 | ref-01-ground-pair | 16 | False | 0.1207294 | 36.65306 | 36.6509 |
| mass_3-s1 | ref-01-ground-pair | 50 | False | 0.4222696 | 449.2254 | 79.18938 |
| mass_3-s1 | ref-06-minimal-smoke | 16 | False | 0.09319687 | 38.66436 | 38.66436 |
| mass_3-s1 | ref-06-minimal-smoke | 50 | False | 0.3032693 | 190.0701 | 54.04012 |
| original_checkpoint | ref-01-ground-pair | 16 | False | 0.1608797 | 46.75688 | 39.98872 |
| original_checkpoint | ref-01-ground-pair | 50 | False | 0.4902727 | 655.2009 | 99.93824 |
| original_checkpoint | ref-06-minimal-smoke | 16 | False | 0.1242102 | 41.04913 | 40.97734 |
| original_checkpoint | ref-06-minimal-smoke | 50 | False | 0.3716692 | 324.3116 | 67.79505 |
| W1_procedural | ref-01-ground-pair | static | True | 0.03904555 | 0.004327589 | 0.004327589 |
| W1_procedural | ref-06-minimal-smoke | static | True | 0.04301075 | 0.003213205 | 0.003213205 |
| D1_mapped_30 | ref-01-ground-pair | static | True | 0.1072894 | 0.008973205 | 0.008973205 |
| D1_mass_3 | ref-01-ground-pair | static | True | 0.1045388 | 0.008848749 | 0.008848749 |
| D1_mapped_30 | ref-06-minimal-smoke | static | True | 0.08542801 | 0.006259237 | 0.006259237 |
| D1_mass_3 | ref-06-minimal-smoke | static | True | 0.08778885 | 0.006382785 | 0.006382785 |

K2 controls used17 shared-weight updates across17 scenes. D1 used32 per-scene raw-voxel updates at learning rate0.05. F1 uses64 per-scene NCA updates at0.0001. W1 is a static procedural witness. Degrees of freedom and effort differ; these are diagnostic comparisons, not matched-cost model rankings.

## Every training update

| Member | Update | Objective before update | Gradient norm before clip | Seconds incl. checkpoint/fields |
|---|---:|---:|---:|---:|
| mapped_30-r0 | 1 | 48.51934 | 2299.848 | 3.493 |
| mapped_30-r0 | 2 | 43.57806 | 1672.948 | 3.154 |
| mapped_30-r1 | 1 | 40.57716 | 233.0123 | 3.483 |
| mapped_30-r1 | 2 | 40.77967 | 220.7734 | 3.188 |
| mass_3-r0 | 1 | 40.08418 | 402.0558 | 3.313 |
| mass_3-r0 | 2 | 39.45486 | 401.8119 | 2.636 |
| mass_3-r1 | 1 | 40.56278 | 189.5181 | 2.670 |
| mass_3-r1 | 2 | 40.77933 | 219.4385 | 2.594 |

Training fields describe the forward pass BEFORE the update; named checkpoints are AFTER it. Evaluation fields are AFTER the named update. Recovery certifies completed CPU boundaries only, not CUDA or abrupt write interruption. Every failed/interrupted attempt must remain archived. No checkpoint is promoted automatically.

## Verification

```json
{
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
  "source_hashes_verified": 28,
  "saved_fields_rescored": 32,
  "training_checkpoints_verified": 8,
  "evaluation_metrics_verified": 24,
  "final_checkpoint_rollouts_replayed": 0,
  "controls_reused": 26,
  "cost_admission": {
    "p90_update_seconds": 3.4860454799956644,
    "max_evaluation_pair_seconds": 4.233992399997078,
    "startup_allowance_seconds": 5.0,
    "safety_factor": 1.5,
    "estimated_member_seconds": 386.6172862795531,
    "estimated_total_seconds": 1546.4691451182125,
    "admitted": true
  }
}
```

All per-case metrics (including binary legality, blocked ground, unsupported material, thickness proxy and threshold counts) are retained in the machine-readable evaluation report and original immutable records. One seed/two scenes cannot establish generalization or architecture failure. Timing includes local overhead; it is not deployment performance.
