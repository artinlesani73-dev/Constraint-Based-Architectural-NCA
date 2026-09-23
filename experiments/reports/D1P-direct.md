# D1 pilot direct-field control

Run `20260923T105120Z_4d1e4d2d20d6`. Per-scene raw voxel optimization, not NCA training.

Both recipes start from the same weak0.15 scaffold. No solved W1 initialization. Every case uses its own voxel parameters, Adam0.05 and the fixed nine-family/three-regularizer objective. Optimizer steps are not NCA growth steps and the compute budgets are not matched.

## Initial and final summary

| Recipe | State | Cases | Connected | In3%-12% budget | Connected AND in budget | Mean material/envelope | Illegal / blocked / unsupported voxels |
|---|---|---:|---:|---:|---:|---:|---|
| mapped_30 | initial | 3 | 0 | 2 | 0 | 0.03980724 | 0 / 0 / 0 |
| mass_3 | initial | 3 | 0 | 2 | 0 | 0.03980724 | 0 / 0 / 0 |
| mapped_30 | final | 3 | 1 | 2 | 1 | 0.06670799 | 0 / 0 / 1 |
| mass_3 | final | 3 | 2 | 2 | 1 | 0.06709339 | 0 / 0 / 1 |

Budget tolerance1e-6; connectivity uses material>0.5. Connected-and-in-budget is a limited conjunction, not a full nine-family or architectural success claim.

## Final per-family means

| Recipe | access | coverage | facade | ground | legality | sparsity | spill | support | thickness | Total mapped_30 | Total mass_3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mapped_30 | 0.5567975 | 0.5071744 | 0 | 0 | 0 | 0.0007245584 | 0 | 0.2289828 | 0 | 22.89069 | 22.87112 |
| mass_3 | 0.5138864 | 0.4790066 | 0.01403892 | 0 | 0 | 0.000339161 | 0 | 0.2011275 | 0 | 21.4488 | 21.43965 |

## Every optimized case

| Recipe | Scene | Updates | Connected | Mass/envelope | Coverage | Access | Sparsity | Total mapped_30 | Total mass_3 | Worker seconds |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mapped_30 | legacy-easy-seed-008 | 8 | False | 0.02782632 | 0.5616103 | 0.6227296 | 0.002173675 | 28.61539 | 28.5567 | 4.659 |
| mass_3 | legacy-easy-seed-008 | 8 | True | 0.02898252 | 0.4771068 | 0.4939963 | 0.001017483 | 24.28974 | 24.26227 | 4.496 |
| mapped_30 | ref-01-ground-pair | 8 | False | 0.08278462 | 0.4896402 | 0.5660177 | 0 | 20.9487 | 20.9487 | 4.553 |
| mass_3 | ref-01-ground-pair | 8 | False | 0.08278462 | 0.4896402 | 0.5660177 | 0 | 20.9487 | 20.9487 | 4.626 |
| mapped_30 | ref-06-minimal-smoke | 8 | True | 0.08951302 | 0.4702727 | 0.4816452 | 0 | 19.10797 | 19.10797 | 4.402 |
| mass_3 | ref-06-minimal-smoke | 8 | True | 0.08951302 | 0.4702727 | 0.4816452 | 0 | 19.10797 | 19.10797 | 4.438 |

## Preserved K2 and W1 controls on these same scenes

| Model | Scene | NCA growth steps | Connected | Mass/envelope | Total mapped_30 | Total mass_3 |
|---|---|---:|---|---:|---:|---:|
| mapped_30-s0 | legacy-easy-seed-008 | 16 | False | 0.1623398 | 33.28036 | 26.02009 |
| mapped_30-s0 | legacy-easy-seed-008 | 50 | False | 0.1853933 | 42.49464 | 25.17571 |
| mapped_30-s0 | ref-01-ground-pair | 16 | False | 0.08697776 | 36.33255 | 36.33255 |
| mapped_30-s0 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59434 | 35.59434 |
| mapped_30-s0 | ref-06-minimal-smoke | 16 | False | 0.06402721 | 39.31152 | 39.31152 |
| mapped_30-s0 | ref-06-minimal-smoke | 50 | False | 0.07258064 | 37.85109 | 37.85109 |
| mapped_30-s1 | legacy-easy-seed-008 | 16 | False | 0.1624816 | 33.28291 | 25.97394 |
| mapped_30-s1 | legacy-easy-seed-008 | 50 | False | 0.1853933 | 42.49786 | 25.17893 |
| mapped_30-s1 | ref-01-ground-pair | 16 | False | 0.08691613 | 36.33854 | 36.33854 |
| mapped_30-s1 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59698 | 35.59698 |
| mapped_30-s1 | ref-06-minimal-smoke | 16 | False | 0.06393798 | 39.32574 | 39.32574 |
| mapped_30-s1 | ref-06-minimal-smoke | 50 | False | 0.07258064 | 37.8546 | 37.8546 |
| mass_3-s0 | legacy-easy-seed-008 | 16 | False | 0.1773265 | 32.1216 | 18.81197 |
| mass_3-s0 | legacy-easy-seed-008 | 50 | True | 0.1910112 | 27.8652 | 7.442686 |
| mass_3-s0 | ref-01-ground-pair | 16 | False | 0.1429801 | 40.10003 | 37.96129 |
| mass_3-s0 | ref-01-ground-pair | 50 | False | 0.4561582 | 546.5374 | 88.87797 |
| mass_3-s0 | ref-06-minimal-smoke | 16 | False | 0.1126303 | 39.45462 | 39.45462 |
| mass_3-s0 | ref-06-minimal-smoke | 50 | False | 0.3339843 | 244.9896 | 59.54298 |
| mass_3-s1 | legacy-easy-seed-008 | 16 | False | 0.1755542 | 33.09811 | 20.59874 |
| mass_3-s1 | legacy-easy-seed-008 | 50 | True | 0.1891386 | 28.19873 | 8.839151 |
| mass_3-s1 | ref-01-ground-pair | 16 | False | 0.1207294 | 36.65306 | 36.6509 |
| mass_3-s1 | ref-01-ground-pair | 50 | False | 0.4222696 | 449.2254 | 79.18938 |
| mass_3-s1 | ref-06-minimal-smoke | 16 | False | 0.09319687 | 38.66436 | 38.66436 |
| mass_3-s1 | ref-06-minimal-smoke | 50 | False | 0.3032693 | 190.0701 | 54.04012 |
| original_checkpoint | legacy-easy-seed-008 | 16 | False | 0.1750936 | 35.46016 | 23.16719 |
| original_checkpoint | legacy-easy-seed-008 | 50 | False | 0.1872659 | 43.65197 | 25.32692 |
| original_checkpoint | ref-01-ground-pair | 16 | False | 0.1608797 | 46.75688 | 39.98872 |
| original_checkpoint | ref-01-ground-pair | 50 | False | 0.4902727 | 655.2009 | 99.93824 |
| original_checkpoint | ref-06-minimal-smoke | 16 | False | 0.1242102 | 41.04913 | 40.97734 |
| original_checkpoint | ref-06-minimal-smoke | 50 | False | 0.3716692 | 324.3116 | 67.79505 |
| W1_procedural | legacy-easy-seed-008 | static | True | 0.06367041 | 0.005291354 | 0.005291354 |
| W1_procedural | ref-01-ground-pair | static | True | 0.03904555 | 0.004327589 | 0.004327589 |
| W1_procedural | ref-06-minimal-smoke | static | True | 0.04301075 | 0.003213205 | 0.003213205 |

No control was rerun or selected by appearance. These are existing development scenes; direct per-scene fitting does not establish learned generalization, physical usability or safety.

## Verification and limits

{
  "saved_field_projections_verified": 48,
  "checkpoint_boundaries_verified": 48,
  "initial_final_scores_recomputed": 12,
  "gradient_norms_verified": 120,
  "controls_reused": 33,
  "scope": "Intermediate projections/checkpoints/norms verified; full objective recomputation at initial/final states."
}

Every update has a raw/projected field, pre-update gradient, objective trace and complete optimizer checkpoint. Field/checkpoint records are AFTER updates; traces describe BEFORE updates. All initial/final objectives and binary metrics were recomputed. Intermediate objectives were not all independently recomputed. No production checkpoint/default, paid compute or cloud operation.

## Cost admission for the frozen full comparison

{
  "pilot_run": "20260923T105120Z_4d1e4d2d20d6",
  "p90_update_seconds": 0.33364569999976085,
  "max_session_setup_seconds": 1.41796969997813,
  "startup_allowance_seconds": 5.0,
  "estimated_full_seconds": 799.5097823996098,
  "total_cap_seconds": 900,
  "admitted": true
}

Admission uses timing only, not quality-based tuning. Full comparison is not an outcome of this pilot.
