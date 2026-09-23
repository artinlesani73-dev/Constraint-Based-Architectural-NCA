# R2 composed-objective CPU recovery

Run `20260923T095240Z_652e01d22fee`; source `0fca1b83cf834e6a98d416507bdcb3afb716655a`.

Four logical updates,ten executed across four fresh processes. All seven recovery comparisons independently rechecked against registered checkpoint/field artifacts: prefix boundary, full model/optimizer/scheduler/RNG checkpoint trees, traces and fields for resume and repeated resume. All three scheduled scenes exercised.

| Update | Scene | Steps | Total objective | Gradient norm before clipping |
|---:|---|---:|---:|---:|
| 1 | legacy-easy-seed-000 | 2 | 40.31517 | 12.178408 |
| 2 | ref-01-ground-pair | 3 | 38.795002 | 51.357979 |
| 3 | ref-06-minimal-smoke | 2 | 38.371758 | 29.473536 |
| 4 | legacy-easy-seed-000 | 3 | 38.083889 | 24.388151 |

The mass_3 candidate recipe composes all nine corrected families and three retained regularizers. These updates validate recovery mechanics, not coefficient quality or a learning curve across differing scenes. Source/scene/proposal hashes and full coefficients are checkpoint metadata.

Boundary: completed CPU updates after orderly exit. This does not certify CUDA, mixed precision, sample pools, abrupt mid-write failure or the later K2 training loop. The original checkpoint remains unchanged. No paid compute or Drive operation.
