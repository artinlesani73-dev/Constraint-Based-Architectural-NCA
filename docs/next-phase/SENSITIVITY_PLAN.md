# K2 local sensitivity proposal

Prepared from K1; not executed. Two fixed recipes differ only in sparsity coefficient30 versus3. Both use the corrected objective and explicit boundary-aware cantilever. Their other numeric coefficients come from the historical checkpoint; that is a comparison control, not a claim of equivalent term scales.

| Recipe | Steps | Cases | Median combined gradient norm | Coverage improving raw direction | Sparsity improving raw direction |
|---|---:|---:|---:|---:|---:|
| mapped_30 | 4 | 34 | 35.64948 | 32 | 0 |
| mapped_30 | 16 | 34 | 347.2022 | 2 | 31 |
| mapped_30 | 50 | 3 | 6080.75 | 0 | 2 |
| mass_3 | 4 | 34 | 35.64948 | 32 | 0 |
| mass_3 | 16 | 34 | 51.49783 | 18 | 18 |
| mass_3 | 50 | 3 | 620.5267 | 0 | 2 |

Signs describe an infinitesimal step along the negative combined gradient. They do not model Adam, finite-step effects, future states or actual learning. Zero/inactive term gradients do not count as improving.

The proposed experiment uses two recipes x two training seeds,17 updates each, every calibration scene once in a recorded deterministic order, 16-step rollouts, Adam1e-4 and norm clipping1. Evaluate all17 development scenes at16/50 steps with firing seed2 and retain checkpoint/W1 baselines. This is68 logical training updates, not a paid pilot or geometry generalization test.

Before execution verify exact CPU restart with the full composed objective, coefficients and scene order in metadata. Checkpoint every update; cap each run at900 seconds and preserve an interrupted result if reached. No automatic Drive or Colab use. Do not select a final recipe solely from the directional table; retain per-family outcomes and failure cases.
