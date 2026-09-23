# F1 recovery: repeated single-scene fitting

Run `20260923T112316Z_f34a421f8302`.

Unchanged K2 update rule and original checkpoint; independent models repeat one development scene. One training seed, weak scaffold initialization, 16 growth steps per optimizer update. The only paired coefficient change is sparsity30 versus3. Evaluations use firing seed2, separate from training.

8 executed optimizer updates; 14 evaluations. All saved objectives and evaluation metrics rescored; complete checkpoint metadata/counters checked. Verification uses the shared formulas, not an independent implementation. Final study rollouts are replayed from all four final checkpoints at both horizons.

## Every evaluation boundary

| Member | Update | Growth steps | Connected | Mass/envelope | Coverage | Access | Sparsity | Support | Total weight30 | Total weight3 | Guide raw<0 / count |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| prefix | 0 | 16 | False | 0.1608797 | 0.7813446 | 1 | 0.2506725 | 0.311887 | 46.75688 | 39.98872 | 16 / 36 |
| prefix | 0 | 50 | False | 0.4902727 | 0.703013 | 1 | 20.56528 | 0.002008129 | 655.2009 | 99.93824 | 17 / 36 |
| prefix | 1 | 16 | False | 0.1513147 | 0.781225 | 1 | 0.1470916 | 0.2931893 | 43.12407 | 39.15259 | 16 / 36 |
| prefix | 1 | 50 | False | 0.4770313 | 0.7029975 | 1 | 19.1207 | 0.002585045 | 611.7895 | 95.53047 | 17 / 36 |
| repeat | 3 | 16 | False | 0.1331297 | 0.7808632 | 1 | 0.02585837 | 0.2395845 | 38.18335 | 37.48517 | 16 / 36 |
| repeat | 3 | 50 | False | 0.449705 | 0.7029602 | 1 | 16.30581 | 0.00164386 | 527.2561 | 86.99935 | 17 / 36 |
| resumed | 3 | 16 | False | 0.1331297 | 0.7808632 | 1 | 0.02585837 | 0.2395845 | 38.18335 | 37.48517 | 16 / 36 |
| resumed | 3 | 50 | False | 0.449705 | 0.7029602 | 1 | 16.30581 | 0.00164386 | 527.2561 | 86.99935 | 17 / 36 |
| whole | 0 | 16 | False | 0.1608797 | 0.7813446 | 1 | 0.2506725 | 0.311887 | 46.75688 | 39.98872 | 16 / 36 |
| whole | 0 | 50 | False | 0.4902727 | 0.703013 | 1 | 20.56528 | 0.002008129 | 655.2009 | 99.93824 | 17 / 36 |
| whole | 1 | 16 | False | 0.1513147 | 0.781225 | 1 | 0.1470916 | 0.2931893 | 43.12407 | 39.15259 | 16 / 36 |
| whole | 1 | 50 | False | 0.4770313 | 0.7029975 | 1 | 19.1207 | 0.002585045 | 611.7895 | 95.53047 | 17 / 36 |
| whole | 3 | 16 | False | 0.1331297 | 0.7808632 | 1 | 0.02585837 | 0.2395845 | 38.18335 | 37.48517 | 16 / 36 |
| whole | 3 | 50 | False | 0.449705 | 0.7029602 | 1 | 16.30581 | 0.00164386 | 527.2561 | 86.99935 | 17 / 36 |

r0 is ground-pair; r1 is minimal-smoke. Connected uses material>0.5. Mass is continuous; budget3%-12%, tolerance1e-6. Connectivity and budget do not certify all nine constraints, usable architecture or mechanical safety. Raw saturation counts do not prove a blocked parameter gradient.

## Complete objective values

| Member | Update | Growth | access | coverage | facade | ground | legality | sparsity | spill | support | thickness | cantilever_boundary | density_binary | tv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| whole | 0 | 16 | 1 | 0.7813446 | 0.2138448 | 0 | 0 | 0.2506725 | 0.002068451 | 0.311887 | 0 | 0.000342664 | 0.001688452 | 0.01105864 |
| whole | 0 | 50 | 1 | 0.703013 | 0.5313832 | 0 | 0 | 20.56528 | 0.01215305 | 0.002008129 | 0 | 0.0003662109 | 3.254303e-05 | 0.03141786 |
| whole | 1 | 16 | 1 | 0.781225 | 0.1775293 | 0 | 0 | 0.1470916 | 0.001735581 | 0.2931893 | 0 | 0.0003421802 | 0.001519508 | 0.01022751 |
| whole | 1 | 50 | 1 | 0.7029975 | 0.5247554 | 0 | 0 | 19.1207 | 0.01169948 | 0.002585045 | 0 | 0.0003662109 | 3.895021e-05 | 0.03076279 |
| whole | 3 | 16 | 1 | 0.7808632 | 0.09281118 | 0 | 0 | 0.02585837 | 0.001096839 | 0.2395845 | 0 | 0.0003413078 | 0.001159839 | 0.008622588 |
| whole | 3 | 50 | 1 | 0.7029602 | 0.5192027 | 0 | 0 | 16.30581 | 0.01087977 | 0.00164386 | 0 | 0.0003662109 | 2.485681e-05 | 0.02885923 |
| prefix | 0 | 16 | 1 | 0.7813446 | 0.2138448 | 0 | 0 | 0.2506725 | 0.002068451 | 0.311887 | 0 | 0.000342664 | 0.001688452 | 0.01105864 |
| prefix | 0 | 50 | 1 | 0.703013 | 0.5313832 | 0 | 0 | 20.56528 | 0.01215305 | 0.002008129 | 0 | 0.0003662109 | 3.254303e-05 | 0.03141786 |
| prefix | 1 | 16 | 1 | 0.781225 | 0.1775293 | 0 | 0 | 0.1470916 | 0.001735581 | 0.2931893 | 0 | 0.0003421802 | 0.001519508 | 0.01022751 |
| prefix | 1 | 50 | 1 | 0.7029975 | 0.5247554 | 0 | 0 | 19.1207 | 0.01169948 | 0.002585045 | 0 | 0.0003662109 | 3.895021e-05 | 0.03076279 |
| resumed | 3 | 16 | 1 | 0.7808632 | 0.09281118 | 0 | 0 | 0.02585837 | 0.001096839 | 0.2395845 | 0 | 0.0003413078 | 0.001159839 | 0.008622588 |
| resumed | 3 | 50 | 1 | 0.7029602 | 0.5192027 | 0 | 0 | 16.30581 | 0.01087977 | 0.00164386 | 0 | 0.0003662109 | 2.485681e-05 | 0.02885923 |
| repeat | 3 | 16 | 1 | 0.7808632 | 0.09281118 | 0 | 0 | 0.02585837 | 0.001096839 | 0.2395845 | 0 | 0.0003413078 | 0.001159839 | 0.008622588 |
| repeat | 3 | 50 | 1 | 0.7029602 | 0.5192027 | 0 | 0 | 16.30581 | 0.01087977 | 0.00164386 | 0 | 0.0003662109 | 2.485681e-05 | 0.02885923 |

## Saved controls on the same scenes

| Method | Scene | Growth steps | Connected | Mass/envelope | Total weight30 | Total weight3 |
|---|---|---:|---|---:|---:|---:|
| mapped_30-s0 | ref-01-ground-pair | 16 | False | 0.08697776 | 36.33255 | 36.33255 |
| mapped_30-s0 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59434 | 35.59434 |
| mapped_30-s1 | ref-01-ground-pair | 16 | False | 0.08691613 | 36.33854 | 36.33854 |
| mapped_30-s1 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59698 | 35.59698 |
| mass_3-s0 | ref-01-ground-pair | 16 | False | 0.1429801 | 40.10003 | 37.96129 |
| mass_3-s0 | ref-01-ground-pair | 50 | False | 0.4561582 | 546.5374 | 88.87797 |
| mass_3-s1 | ref-01-ground-pair | 16 | False | 0.1207294 | 36.65306 | 36.6509 |
| mass_3-s1 | ref-01-ground-pair | 50 | False | 0.4222696 | 449.2254 | 79.18938 |
| original_checkpoint | ref-01-ground-pair | 16 | False | 0.1608797 | 46.75688 | 39.98872 |
| original_checkpoint | ref-01-ground-pair | 50 | False | 0.4902727 | 655.2009 | 99.93824 |
| W1_procedural | ref-01-ground-pair | static | True | 0.03904555 | 0.004327589 | 0.004327589 |
| D1_mapped_30 | ref-01-ground-pair | static | True | 0.1072894 | 0.008973205 | 0.008973205 |
| D1_mass_3 | ref-01-ground-pair | static | True | 0.1045388 | 0.008848749 | 0.008848749 |

K2 controls used17 shared-weight updates across17 scenes. D1 used32 per-scene raw-voxel updates at learning rate0.05. F1 uses64 per-scene NCA updates at0.0001. W1 is a static procedural witness. Degrees of freedom and effort differ; these are diagnostic comparisons, not matched-cost model rankings.

## Every training update

| Member | Update | Objective before update | Gradient norm before clip | Seconds incl. checkpoint/fields |
|---|---:|---:|---:|---:|
| whole | 1 | 40.08418 | 402.0558 | 3.830 |
| whole | 2 | 39.45486 | 401.8119 | 3.351 |
| whole | 3 | 38.91521 | 400.1834 | 3.167 |
| prefix | 1 | 40.08418 | 402.0558 | 3.295 |
| resumed | 2 | 39.45486 | 401.8119 | 3.327 |
| resumed | 3 | 38.91521 | 400.1834 | 3.150 |
| repeat | 2 | 39.45486 | 401.8119 | 3.290 |
| repeat | 3 | 38.91521 | 400.1834 | 3.374 |

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
  "saved_fields_rescored": 22,
  "training_checkpoints_verified": 8,
  "evaluation_metrics_verified": 14,
  "final_checkpoint_rollouts_replayed": 0,
  "controls_reused": 13
}
```

All per-case metrics (including binary legality, blocked ground, unsupported material, thickness proxy and threshold counts) are retained in the machine-readable evaluation report and original immutable records. One seed/two scenes cannot establish generalization or architecture failure. Timing includes local overhead; it is not deployment performance.
