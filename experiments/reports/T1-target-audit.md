# T1 target compatibility audit

Run `20260923T082527Z_845d2aa6aec0`; source `f76797593f2e4595b60ac8cbe7e92c4598c78900`.

432 static target records, 72 necessary-bound records, 36 direct occupancy-gradient cases. All registered hashes verified; bound arithmetic, target mass/contact/physical volume, saved gradient norms and pairwise cosines independently rechecked. No optimizer updates.

## Joint compatibility

Passing bounds is necessary, not sufficient. The sealed scene is excluded from feasible counts and retained in raw records. A failure means these exact losses cannot all be zero, not that architecture in the scene is impossible.

| Envelope radius | Budget | Previous bounds + feasible route | Including facade bound + feasible route |
|---:|---|---:|---:|
| 3 | site | 3/17 | 3/17 |
| 3 | envelope | 17/17 | 11/17 |
| 6 | site | 12/17 | 12/17 |
| 6 | envelope | 17/17 | 15/17 |

Coverage fixes guide material to one. If C of its cells touch facade, the 15% contact cap requires total mass at least C/0.15. Coverage, zero spill and the upper material budget can contradict that requirement.

| Scene | Envelope | Mandatory facade cells | Required minimum mass | Allowed maximum mass |
|---|---:|---:|---:|---:|
| legacy-easy-seed-000 | 3 | 8 | 53.3333 | 33.0000 |
| legacy-easy-seed-001 | 3 | 8 | 53.3333 | 31.5600 |
| legacy-easy-seed-004 | 3 | 11 | 73.3333 | 46.5600 |
| legacy-easy-seed-005 | 3 | 8 | 53.3333 | 50.2800 |
| legacy-easy-seed-007 | 3 | 14 | 93.3333 | 31.8000 |
| legacy-easy-seed-007 | 6 | 14 | 93.3333 | 92.2800 |
| legacy-easy-seed-008 | 3 | 13 | 86.6667 | 22.5600 |
| legacy-easy-seed-008 | 6 | 13 | 86.6667 | 64.0800 |

## Candidate outcomes: radius-six envelope budget

Counts below use all 17 geometrically feasible scenes, including those with conflicting objective bounds. Empty is a negative control. Connected/supported are spatial proxies, not walkability or mechanical safety.

| Candidate | Binary connected | All nine near zero | Nonzero thickness | Nonzero sparsity | Nonzero facade |
|---|---:|---:|---:|---:|---:|
| empty | 0/17 | 0/17 | 0/17 | 17/17 | 0/17 |
| guide | 17/17 | 2/17 | 0/17 | 10/17 | 13/17 |
| scaffold | 17/17 | 0/17 | 0/17 | 17/17 | 14/17 |
| radius1 | 17/17 | 1/17 | 0/17 | 4/17 | 14/17 |
| radius3 | 17/17 | 0/17 | 0/17 | 17/17 | 14/17 |
| radius6 | 17/17 | 0/17 | 17/17 | 17/17 | 10/17 |

### Zero-loss witnesses across the full matrix

Near-zero means every term <=1e-7 with nonempty material and a valid context. Passing these proxies is not architectural quality. A one-voxel-wide guide can pass; thickness minimizes bulk, not minimum width, and access measures material rather than circulation void.

| Scene | Candidate | Envelope | Budget | Material voxels | Volume (m3) |
|---|---|---:|---|---:|---:|
| ref-01-ground-pair | guide | 3 | envelope | 36 | 18.432 |
| ref-01-ground-pair | guide | 6 | envelope | 36 | 18.432 |
| ref-02-facade-pair-and-ground | guide | 3 | envelope | 55 | 28.160 |
| ref-02-facade-pair-and-ground | radius1 | 6 | envelope | 215 | 110.080 |
| ref-02-facade-pair-and-ground | radius3 | 3 | site | 806 | 412.672 |
| ref-02-facade-pair-and-ground | radius3 | 6 | site | 806 | 412.672 |
| ref-04-asymmetric-heights | guide | 3 | envelope | 69 | 35.328 |
| ref-06-minimal-smoke | guide | 3 | envelope | 32 | 16.384 |
| ref-06-minimal-smoke | guide | 6 | envelope | 32 | 16.384 |

## Direct occupancy gradient scales

Only the 17 route-feasible scenes appear below; both budget choices remain separate. The synthetic state is high on guide and low elsewhere inside radius six. These are raw, unit-weight direct occupancy derivatives, not model parameter gradients or proposed weights. Legal-coordinate masking removes forbidden coordinates only: it is not the full feasible tangent cone at occupancy 0/1. Max/min ties have implementation-selected subgradients.

| Budget | Family | Median value | Median legal gradient norm | Range of legal norms |
|---|---|---:|---:|---|
| site | legality | 0 | 0 | 0 - 0 |
| site | coverage | 0.200591 | 0.171499 | 0.120386 - 0.242536 |
| site | spill | 0 | 0.00544223 | 0.00522873 - 0.00571412 |
| site | ground | 0 | 0 | 0 - 0 |
| site | thickness | 0.00564509 | 2.57741 | 1.44424 - 5.02507 |
| site | sparsity | 0.0276103 | 0.00566462 | 0.0053873 - 0.00586094 |
| site | facade | 0.0313366 | 0.556618 | 0 - 2.54055 |
| site | access | 0.293602 | 1 | 0.707107 - 1 |
| site | support | 0.612639 | 0.912512 | 0.442358 - 2.54129 |
| envelope | legality | 0 | 0 | 0 - 0 |
| envelope | coverage | 0.200591 | 0.171499 | 0.120386 - 0.242536 |
| envelope | spill | 0 | 0.00544223 | 0.00522873 - 0.00571412 |
| envelope | ground | 0 | 0 | 0 - 0 |
| envelope | thickness | 0.00564509 | 2.57741 | 1.44424 - 5.02507 |
| envelope | sparsity | 0 | 0 | 0 - 0 |
| envelope | facade | 0.0313366 | 0.556618 | 0 - 2.54055 |
| envelope | access | 0.293602 | 1 | 0.707107 - 1 |
| envelope | support | 0.612639 | 0.912512 | 0.442358 - 2.54129 |

### Pairwise opposition on legal coordinates

Negative cosine means locally opposed direct gradients in this probe. Counts use cosine <-0.01; null values mean a zero norm and are excluded. Alignment is state dependent and is not proof of global incompatibility.

| Budget | Pair | Opposed / defined | Median cosine |
|---|---|---:|---:|
| site | coverage / facade | 4/12 | -0.00799 |
| site | spill / sparsity | 17/17 | -0.97358 |
| site | spill / facade | 12/12 | -0.59567 |
| site | thickness / sparsity | 17/17 | -0.77327 |
| site | thickness / facade | 12/12 | -0.40786 |
| site | sparsity / support | 17/17 | -0.96198 |
| site | facade / access | 1/12 | 0.00475 |
| site | facade / support | 12/12 | -0.59109 |
| envelope | coverage / facade | 4/12 | -0.00799 |
| envelope | spill / facade | 12/12 | -0.59567 |
| envelope | thickness / facade | 12/12 | -0.40786 |
| envelope | facade / access | 1/12 | 0.00475 |
| envelope | facade / support | 12/12 | -0.59109 |

## All static cases

All nine raw terms retained below. Near-zero diagnostics do not override invalid scene/context labels. Full fields, binary metrics and all gradient pairs are in the immutable run archive.

| Scene | Candidate | Envelope | Budget | Valid context | Voxels | legality | coverage | spill | ground | thickness | sparsity | facade | access | support |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy-easy-seed-000 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-000 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-000 | empty | 6 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-000 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-000 | guide | 3 | site | False | 22 | 0 | 0 | 0 | 0 | 0 | 0.0292079 | 0.213636 | 0 | 0 |
| legacy-easy-seed-000 | guide | 3 | envelope | True | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0.213636 | 0 | 0 |
| legacy-easy-seed-000 | guide | 6 | site | False | 22 | 0 | 0 | 0 | 0 | 0 | 0.0292079 | 0.213636 | 0 | 0 |
| legacy-easy-seed-000 | guide | 6 | envelope | True | 22 | 0 | 0 | 0 | 0 | 0 | 0.000149254 | 0.213636 | 0 | 0 |
| legacy-easy-seed-000 | scaffold | 3 | site | False | 168 | 0 | 0 | 0.000180012 | 0 | 0 | 0.0239516 | 0.183333 | 0 | 0 |
| legacy-easy-seed-000 | scaffold | 3 | envelope | True | 168 | 0 | 0 | 0.000180012 | 0 | 0 | 36.1488 | 0.183333 | 0 | 0 |
| legacy-easy-seed-000 | scaffold | 6 | site | False | 168 | 0 | 0 | 0 | 0 | 0 | 0.0239516 | 0.183333 | 0 | 0 |
| legacy-easy-seed-000 | scaffold | 6 | envelope | True | 168 | 0 | 0 | 0 | 0 | 0 | 1.74802 | 0.183333 | 0 | 0 |
| legacy-easy-seed-000 | radius1 | 3 | site | False | 79 | 0 | 0 | 0 | 0 | 0 | 0.0271558 | 0.20443 | 0 | 0 |
| legacy-easy-seed-000 | radius1 | 3 | envelope | True | 79 | 0 | 0 | 0 | 0 | 0 | 4.19702 | 0.20443 | 0 | 0 |
| legacy-easy-seed-000 | radius1 | 6 | site | False | 79 | 0 | 0 | 0 | 0 | 0 | 0.0271558 | 0.20443 | 0 | 0 |
| legacy-easy-seed-000 | radius1 | 6 | envelope | True | 79 | 0 | 0 | 0 | 0 | 0 | 0 | 0.20443 | 0 | 0 |
| legacy-easy-seed-000 | radius3 | 3 | site | False | 275 | 0 | 0 | 0 | 0 | 0 | 0.0200994 | 0.137273 | 0 | 0 |
| legacy-easy-seed-000 | radius3 | 3 | envelope | True | 275 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.137273 | 0 | 0 |
| legacy-easy-seed-000 | radius3 | 6 | site | False | 275 | 0 | 0 | 0 | 0 | 0 | 0.0200994 | 0.137273 | 0 | 0 |
| legacy-easy-seed-000 | radius3 | 6 | envelope | True | 275 | 0 | 0 | 0 | 0 | 0 | 9.61155 | 0.137273 | 0 | 0 |
| legacy-easy-seed-000 | radius6 | 3 | site | False | 737 | 0 | 0 | 0.0166331 | 0 | 0.0230665 | 0.0034663 | 0.0481004 | 0 | 0 |
| legacy-easy-seed-000 | radius6 | 3 | envelope | True | 737 | 0 | 0 | 0.0166331 | 0 | 0.0230665 | 983.04 | 0.0481004 | 0 | 0 |
| legacy-easy-seed-000 | radius6 | 6 | site | False | 737 | 0 | 0 | 0 | 0 | 0.0230665 | 0.0034663 | 0.0481004 | 0 | 0 |
| legacy-easy-seed-000 | radius6 | 6 | envelope | True | 737 | 0 | 0 | 0 | 0 | 0.0230665 | 116.16 | 0.0481004 | 0 | 0 |
| legacy-easy-seed-001 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-001 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-001 | empty | 6 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-001 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-001 | guide | 3 | site | False | 23 | 0 | 0 | 0 | 0 | 0 | 0.0291473 | 0.197826 | 0 | 0 |
| legacy-easy-seed-001 | guide | 3 | envelope | True | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 0.197826 | 0 | 0 |
| legacy-easy-seed-001 | guide | 6 | site | False | 23 | 0 | 0 | 0 | 0 | 0 | 0.0291473 | 0.197826 | 0 | 0 |
| legacy-easy-seed-001 | guide | 6 | envelope | True | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 0.197826 | 0 | 0 |
| legacy-easy-seed-001 | scaffold | 3 | site | False | 170 | 0 | 0 | 0.000111226 | 0 | 0 | 0.0236972 | 0.214706 | 0 | 0 |
| legacy-easy-seed-001 | scaffold | 3 | envelope | True | 170 | 0 | 0 | 0.000111226 | 0 | 0 | 41.5626 | 0.214706 | 0 | 0 |
| legacy-easy-seed-001 | scaffold | 6 | site | False | 170 | 0 | 0 | 0 | 0 | 0 | 0.0236972 | 0.214706 | 0 | 0 |
| legacy-easy-seed-001 | scaffold | 6 | envelope | True | 170 | 0 | 0 | 0 | 0 | 0 | 2.43663 | 0.214706 | 0 | 0 |
| legacy-easy-seed-001 | radius1 | 3 | site | False | 82 | 0 | 0 | 0 | 0 | 0 | 0.0269598 | 0.203659 | 0 | 0 |
| legacy-easy-seed-001 | radius1 | 3 | envelope | True | 82 | 0 | 0 | 0 | 0 | 0 | 5.51734 | 0.203659 | 0 | 0 |
| legacy-easy-seed-001 | radius1 | 6 | site | False | 82 | 0 | 0 | 0 | 0 | 0 | 0.0269598 | 0.203659 | 0 | 0 |
| legacy-easy-seed-001 | radius1 | 6 | envelope | True | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0.203659 | 0 | 0 |
| legacy-easy-seed-001 | radius3 | 3 | site | False | 263 | 0 | 0 | 0 | 0 | 0 | 0.0202491 | 0.154182 | 0 | 0 |
| legacy-easy-seed-001 | radius3 | 3 | envelope | True | 263 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.154182 | 0 | 0 |
| legacy-easy-seed-001 | radius3 | 6 | site | False | 263 | 0 | 0 | 0 | 0 | 0 | 0.0202491 | 0.154182 | 0 | 0 |
| legacy-easy-seed-001 | radius3 | 6 | envelope | True | 263 | 0 | 0 | 0 | 0 | 0 | 10.3615 | 0.154182 | 0 | 0 |
| legacy-easy-seed-001 | radius6 | 3 | site | False | 687 | 0 | 0 | 0.01572 | 0 | 0.0160116 | 0.00452914 | 0.0683406 | 0 | 0 |
| legacy-easy-seed-001 | radius6 | 3 | envelope | True | 687 | 0 | 0 | 0.01572 | 0 | 0.0160116 | 931.635 | 0.0683406 | 0 | 0 |
| legacy-easy-seed-001 | radius6 | 6 | site | False | 687 | 0 | 0 | 0 | 0 | 0.0160116 | 0.00452914 | 0.0683406 | 0 | 0 |
| legacy-easy-seed-001 | radius6 | 6 | envelope | True | 687 | 0 | 0 | 0 | 0 | 0.0160116 | 116.16 | 0.0683406 | 0 | 0 |
| legacy-easy-seed-002 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-002 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-002 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-002 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-002 | guide | 3 | site | False | 38 | 0 | 0 | 0 | 0 | 0 | 0.028577 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-002 | guide | 3 | envelope | True | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-002 | guide | 6 | site | True | 38 | 0 | 0 | 0 | 0 | 0 | 0.028577 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-002 | guide | 6 | envelope | True | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-002 | scaffold | 3 | site | False | 390 | 0 | 0 | 0.000224685 | 0 | 0 | 0.0153954 | 0.0807692 | 0 | 0 |
| legacy-easy-seed-002 | scaffold | 3 | envelope | True | 390 | 0 | 0 | 0.000224685 | 0 | 0 | 56.3806 | 0.0807692 | 0 | 0 |
| legacy-easy-seed-002 | scaffold | 6 | site | True | 390 | 0 | 0 | 0 | 0 | 0 | 0.0153954 | 0.0807692 | 0 | 0 |
| legacy-easy-seed-002 | scaffold | 6 | envelope | True | 390 | 0 | 0 | 0 | 0 | 0 | 6.42157 | 0.0807692 | 0 | 0 |
| legacy-easy-seed-002 | radius1 | 3 | site | False | 154 | 0 | 0 | 0 | 0 | 0 | 0.0242331 | 0.0577922 | 0 | 0 |
| legacy-easy-seed-002 | radius1 | 3 | envelope | True | 154 | 0 | 0 | 0 | 0 | 0 | 4.3082 | 0.0577922 | 0 | 0 |
| legacy-easy-seed-002 | radius1 | 6 | site | True | 154 | 0 | 0 | 0 | 0 | 0 | 0.0242331 | 0.0577922 | 0 | 0 |
| legacy-easy-seed-002 | radius1 | 6 | envelope | True | 154 | 0 | 0 | 0 | 0 | 0 | 0.0123842 | 0.0577922 | 0 | 0 |
| legacy-easy-seed-002 | radius3 | 3 | site | False | 532 | 0 | 0 | 0 | 0 | 0 | 0.0100779 | 0.0473684 | 0 | 0 |
| legacy-easy-seed-002 | radius3 | 3 | envelope | True | 532 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.0473684 | 0 | 0 |
| legacy-easy-seed-002 | radius3 | 6 | site | True | 532 | 0 | 0 | 0 | 0 | 0 | 0.0100779 | 0.0473684 | 0 | 0 |
| legacy-easy-seed-002 | radius3 | 6 | envelope | True | 532 | 0 | 0 | 0 | 0 | 0 | 15.935 | 0.0473684 | 0 | 0 |
| legacy-easy-seed-002 | radius6 | 3 | site | False | 1193 | 0 | 0 | 0.0247528 | 0 | 0.0402347 | 0 | 0.0360855 | 0 | 0 |
| legacy-easy-seed-002 | radius6 | 3 | envelope | True | 1193 | 0 | 0 | 0.0247528 | 0 | 0.0402347 | 675.739 | 0.0360855 | 0 | 0 |
| legacy-easy-seed-002 | radius6 | 6 | site | True | 1193 | 0 | 0 | 0 | 0 | 0.0402347 | 0 | 0.0360855 | 0 | 0 |
| legacy-easy-seed-002 | radius6 | 6 | envelope | True | 1193 | 0 | 0 | 0 | 0 | 0.0402347 | 116.16 | 0.0360855 | 0 | 0 |
| legacy-easy-seed-003 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-003 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-003 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-003 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-003 | guide | 3 | site | False | 34 | 0 | 0 | 0 | 0 | 0 | 0.0286655 | 0.0852941 | 0 | 0 |
| legacy-easy-seed-003 | guide | 3 | envelope | True | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0852941 | 0 | 0 |
| legacy-easy-seed-003 | guide | 6 | site | True | 34 | 0 | 0 | 0 | 0 | 0 | 0.0286655 | 0.0852941 | 0 | 0 |
| legacy-easy-seed-003 | guide | 6 | envelope | True | 34 | 0 | 0 | 0 | 0 | 0 | 0.00716588 | 0.0852941 | 0 | 0 |
| legacy-easy-seed-003 | scaffold | 3 | site | False | 377 | 0 | 0 | 0.000353246 | 0 | 0 | 0.0152029 | 0.0754642 | 0 | 0 |
| legacy-easy-seed-003 | scaffold | 3 | envelope | True | 377 | 0 | 0 | 0.000353246 | 0 | 0 | 56.4505 | 0.0754642 | 0 | 0 |
| legacy-easy-seed-003 | scaffold | 6 | site | True | 377 | 0 | 0 | 0 | 0 | 0 | 0.0152029 | 0.0754642 | 0 | 0 |
| legacy-easy-seed-003 | scaffold | 6 | envelope | True | 377 | 0 | 0 | 0 | 0 | 0 | 2.66094 | 0.0754642 | 0 | 0 |
| legacy-easy-seed-003 | radius1 | 3 | site | False | 135 | 0 | 0 | 0 | 0 | 0 | 0.0247013 | 0.0648148 | 0 | 0 |
| legacy-easy-seed-003 | radius1 | 3 | envelope | True | 135 | 0 | 0 | 0 | 0 | 0 | 3.05218 | 0.0648148 | 0 | 0 |
| legacy-easy-seed-003 | radius1 | 6 | site | True | 135 | 0 | 0 | 0 | 0 | 0 | 0.0247013 | 0.0648148 | 0 | 0 |
| legacy-easy-seed-003 | radius1 | 6 | envelope | True | 135 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0648148 | 0 | 0 |
| legacy-easy-seed-003 | radius3 | 3 | site | False | 514 | 0 | 0 | 0 | 0 | 0 | 0.00982573 | 0.0445525 | 0 | 0 |
| legacy-easy-seed-003 | radius3 | 3 | envelope | True | 514 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.0445525 | 0 | 0 |
| legacy-easy-seed-003 | radius3 | 6 | site | True | 514 | 0 | 0 | 0 | 0 | 0 | 0.00982573 | 0.0445525 | 0 | 0 |
| legacy-easy-seed-003 | radius3 | 6 | envelope | True | 514 | 0 | 0 | 0 | 0 | 0 | 7.60713 | 0.0445525 | 0 | 0 |
| legacy-easy-seed-003 | radius6 | 3 | site | False | 1489 | 0 | 0 | 0.0382683 | 0 | 0.0429819 | 0 | 0.0098388 | 0 | 0 |
| legacy-easy-seed-003 | radius6 | 3 | envelope | True | 1489 | 0 | 0 | 0.0382683 | 0 | 0.0429819 | 1156.67 | 0.0098388 | 0 | 0 |
| legacy-easy-seed-003 | radius6 | 6 | site | True | 1489 | 0 | 0 | 0 | 0 | 0.0429819 | 0 | 0.0098388 | 0 | 0 |
| legacy-easy-seed-003 | radius6 | 6 | envelope | True | 1489 | 0 | 0 | 0 | 0 | 0.0429819 | 116.16 | 0.0098388 | 0 | 0 |
| legacy-easy-seed-004 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-004 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-004 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-004 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-004 | guide | 3 | site | False | 32 | 0 | 0 | 0 | 0 | 0 | 0.0288327 | 0.19375 | 0 | 0 |
| legacy-easy-seed-004 | guide | 3 | envelope | True | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0.19375 | 0 | 0 |
| legacy-easy-seed-004 | guide | 6 | site | True | 32 | 0 | 0 | 0 | 0 | 0 | 0.0288327 | 0.19375 | 0 | 0 |
| legacy-easy-seed-004 | guide | 6 | envelope | True | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0.19375 | 0 | 0 |
| legacy-easy-seed-004 | scaffold | 3 | site | False | 234 | 0 | 0 | 0.000109437 | 0 | 0 | 0.0214639 | 0.140598 | 0 | 0 |
| legacy-easy-seed-004 | scaffold | 3 | envelope | True | 234 | 0 | 0 | 0.000109437 | 0 | 0 | 35.0068 | 0.140598 | 0 | 0 |
| legacy-easy-seed-004 | scaffold | 6 | site | True | 234 | 0 | 0 | 0 | 0 | 0 | 0.0214639 | 0.140598 | 0 | 0 |
| legacy-easy-seed-004 | scaffold | 6 | envelope | True | 234 | 0 | 0 | 0 | 0 | 0 | 2.39335 | 0.140598 | 0 | 0 |
| legacy-easy-seed-004 | radius1 | 3 | site | False | 111 | 0 | 0 | 0 | 0 | 0 | 0.0259508 | 0.147297 | 0 | 0 |
| legacy-easy-seed-004 | radius1 | 3 | envelope | True | 111 | 0 | 0 | 0 | 0 | 0 | 4.13751 | 0.147297 | 0 | 0 |
| legacy-easy-seed-004 | radius1 | 6 | site | True | 111 | 0 | 0 | 0 | 0 | 0 | 0.0259508 | 0.147297 | 0 | 0 |
| legacy-easy-seed-004 | radius1 | 6 | envelope | True | 111 | 0 | 0 | 0 | 0 | 0 | 0 | 0.147297 | 0 | 0 |
| legacy-easy-seed-004 | radius3 | 3 | site | False | 388 | 0 | 0 | 0 | 0 | 0 | 0.0158461 | 0.0948454 | 0 | 0 |
| legacy-easy-seed-004 | radius3 | 3 | envelope | True | 388 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.0948454 | 0 | 0 |
| legacy-easy-seed-004 | radius3 | 6 | site | True | 388 | 0 | 0 | 0 | 0 | 0 | 0.0158461 | 0.0948454 | 0 | 0 |
| legacy-easy-seed-004 | radius3 | 6 | envelope | True | 388 | 0 | 0 | 0 | 0 | 0 | 12.478 | 0.0948454 | 0 | 0 |
| legacy-easy-seed-004 | radius6 | 3 | site | False | 950 | 0 | 0 | 0.0205012 | 0 | 0.0242105 | 0 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-004 | radius6 | 3 | envelope | True | 950 | 0 | 0 | 0.0205012 | 0 | 0.0242105 | 813.255 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-004 | radius6 | 6 | site | True | 950 | 0 | 0 | 0 | 0 | 0.0242105 | 0 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-004 | radius6 | 6 | envelope | True | 950 | 0 | 0 | 0 | 0 | 0.0242105 | 116.16 | 0.0605263 | 0 | 0 |
| legacy-easy-seed-005 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-005 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-005 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-005 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-005 | guide | 3 | site | False | 36 | 0 | 0 | 0 | 0 | 0 | 0.0287121 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-005 | guide | 3 | envelope | True | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-005 | guide | 6 | site | True | 36 | 0 | 0 | 0 | 0 | 0 | 0.0287121 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-005 | guide | 6 | envelope | True | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-005 | scaffold | 3 | site | False | 282 | 0 | 0 | 0.000178878 | 0 | 0 | 0.0199113 | 0.0982269 | 0 | 0 |
| legacy-easy-seed-005 | scaffold | 3 | envelope | True | 282 | 0 | 0 | 0.000178878 | 0 | 0 | 45.8765 | 0.0982269 | 0 | 0 |
| legacy-easy-seed-005 | scaffold | 6 | site | True | 282 | 0 | 0 | 0 | 0 | 0 | 0.0199113 | 0.0982269 | 0 | 0 |
| legacy-easy-seed-005 | scaffold | 6 | envelope | True | 282 | 0 | 0 | 0 | 0 | 0 | 3.40356 | 0.0982269 | 0 | 0 |
| legacy-easy-seed-005 | radius1 | 3 | site | False | 126 | 0 | 0 | 0 | 0 | 0 | 0.0254923 | 0.0801587 | 0 | 0 |
| legacy-easy-seed-005 | radius1 | 3 | envelope | True | 126 | 0 | 0 | 0 | 0 | 0 | 4.89874 | 0.0801587 | 0 | 0 |
| legacy-easy-seed-005 | radius1 | 6 | site | True | 126 | 0 | 0 | 0 | 0 | 0 | 0.0254923 | 0.0801587 | 0 | 0 |
| legacy-easy-seed-005 | radius1 | 6 | envelope | True | 126 | 0 | 0 | 0 | 0 | 0 | 0.000127322 | 0.0801587 | 0 | 0 |
| legacy-easy-seed-005 | radius3 | 3 | site | False | 419 | 0 | 0 | 0 | 0 | 0 | 0.01501 | 0.0480907 | 0 | 0 |
| legacy-easy-seed-005 | radius3 | 3 | envelope | True | 419 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.0480907 | 0 | 0 |
| legacy-easy-seed-005 | radius3 | 6 | site | True | 419 | 0 | 0 | 0 | 0 | 0 | 0.01501 | 0.0480907 | 0 | 0 |
| legacy-easy-seed-005 | radius3 | 6 | envelope | True | 419 | 0 | 0 | 0 | 0 | 0 | 11.938 | 0.0480907 | 0 | 0 |
| legacy-easy-seed-005 | radius6 | 3 | site | False | 1042 | 0 | 0 | 0.0222882 | 0 | 0.012476 | 0 | 0.0256238 | 0 | 0 |
| legacy-easy-seed-005 | radius6 | 3 | envelope | True | 1042 | 0 | 0 | 0.0222882 | 0 | 0.012476 | 840.314 | 0.0256238 | 0 | 0 |
| legacy-easy-seed-005 | radius6 | 6 | site | True | 1042 | 0 | 0 | 0 | 0 | 0.012476 | 0 | 0.0256238 | 0 | 0 |
| legacy-easy-seed-005 | radius6 | 6 | envelope | True | 1042 | 0 | 0 | 0 | 0 | 0.012476 | 116.16 | 0.0256238 | 0 | 0 |
| legacy-easy-seed-006 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-006 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-006 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-006 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-006 | guide | 3 | site | False | 33 | 0 | 0 | 0 | 0 | 0 | 0.028838 | 0.0924242 | 0 | 0 |
| legacy-easy-seed-006 | guide | 3 | envelope | True | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0924242 | 0 | 0 |
| legacy-easy-seed-006 | guide | 6 | site | True | 33 | 0 | 0 | 0 | 0 | 0 | 0.028838 | 0.0924242 | 0 | 0 |
| legacy-easy-seed-006 | guide | 6 | envelope | True | 33 | 0 | 0 | 0 | 0 | 0 | 0.0125489 | 0.0924242 | 0 | 0 |
| legacy-easy-seed-006 | scaffold | 3 | site | False | 381 | 0 | 0 | 0.000422535 | 0 | 0 | 0.0165845 | 0.025853 | 0 | 0 |
| legacy-easy-seed-006 | scaffold | 3 | envelope | True | 381 | 0 | 0 | 0.000422535 | 0 | 0 | 43.2387 | 0.025853 | 0 | 0 |
| legacy-easy-seed-006 | scaffold | 6 | site | True | 381 | 0 | 0 | 0 | 0 | 0 | 0.0165845 | 0.025853 | 0 | 0 |
| legacy-easy-seed-006 | scaffold | 6 | envelope | True | 381 | 0 | 0 | 0 | 0 | 0 | 0.995866 | 0.025853 | 0 | 0 |
| legacy-easy-seed-006 | radius1 | 3 | site | False | 134 | 0 | 0 | 0 | 0 | 0 | 0.0252817 | 0.0440298 | 0 | 0 |
| legacy-easy-seed-006 | radius1 | 3 | envelope | True | 134 | 0 | 0 | 0 | 0 | 0 | 1.8493 | 0.0440298 | 0 | 0 |
| legacy-easy-seed-006 | radius1 | 6 | site | True | 134 | 0 | 0 | 0 | 0 | 0 | 0.0252817 | 0.0440298 | 0 | 0 |
| legacy-easy-seed-006 | radius1 | 6 | envelope | True | 134 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0440298 | 0 | 0 |
| legacy-easy-seed-006 | radius3 | 3 | site | False | 580 | 0 | 0 | 0 | 0 | 0 | 0.00957746 | 0.00172414 | 0 | 0 |
| legacy-easy-seed-006 | radius3 | 3 | envelope | True | 580 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.00172414 | 0 | 0 |
| legacy-easy-seed-006 | radius3 | 6 | site | True | 580 | 0 | 0 | 0 | 0 | 0 | 0.00957746 | 0.00172414 | 0 | 0 |
| legacy-easy-seed-006 | radius3 | 6 | envelope | True | 580 | 0 | 0 | 0 | 0 | 0 | 5.22943 | 0.00172414 | 0 | 0 |
| legacy-easy-seed-006 | radius6 | 3 | site | False | 1891 | 0 | 0 | 0.046162 | 0 | 0.097303 | 0 | 0 | 0 | 0 |
| legacy-easy-seed-006 | radius6 | 3 | envelope | True | 1891 | 0 | 0 | 0.046162 | 0 | 0.097303 | 1479.26 | 0 | 0 | 0 |
| legacy-easy-seed-006 | radius6 | 6 | site | True | 1891 | 0 | 0 | 0 | 0 | 0.097303 | 0 | 0 | 0 | 0 |
| legacy-easy-seed-006 | radius6 | 6 | envelope | True | 1891 | 0 | 0 | 0 | 0 | 0.097303 | 116.16 | 0 | 0 | 0 |
| legacy-easy-seed-007 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-007 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-007 | empty | 6 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-007 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-007 | guide | 3 | site | False | 22 | 0 | 0 | 0 | 0 | 0 | 0.0291869 | 0.486364 | 0 | 0 |
| legacy-easy-seed-007 | guide | 3 | envelope | True | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0.486364 | 0 | 0 |
| legacy-easy-seed-007 | guide | 6 | site | False | 22 | 0 | 0 | 0 | 0 | 0 | 0.0291869 | 0.486364 | 0 | 0 |
| legacy-easy-seed-007 | guide | 6 | envelope | True | 22 | 0 | 0 | 0 | 0 | 0 | 0.00139142 | 0.486364 | 0 | 0 |
| legacy-easy-seed-007 | scaffold | 3 | site | False | 180 | 0 | 0 | 0.000369604 | 0 | 0 | 0.0233471 | 0.227778 | 0 | 0 |
| legacy-easy-seed-007 | scaffold | 3 | envelope | True | 180 | 0 | 0 | 0.000369604 | 0 | 0 | 46.9133 | 0.227778 | 0 | 0 |
| legacy-easy-seed-007 | scaffold | 6 | site | False | 180 | 0 | 0 | 0 | 0 | 0 | 0.0233471 | 0.227778 | 0 | 0 |
| legacy-easy-seed-007 | scaffold | 6 | envelope | True | 180 | 0 | 0 | 0 | 0 | 0 | 1.9518 | 0.227778 | 0 | 0 |
| legacy-easy-seed-007 | radius1 | 3 | site | False | 73 | 0 | 0 | 0 | 0 | 0 | 0.0273019 | 0.356849 | 0 | 0 |
| legacy-easy-seed-007 | radius1 | 3 | envelope | True | 73 | 0 | 0 | 0 | 0 | 0 | 3.62572 | 0.356849 | 0 | 0 |
| legacy-easy-seed-007 | radius1 | 6 | site | False | 73 | 0 | 0 | 0 | 0 | 0 | 0.0273019 | 0.356849 | 0 | 0 |
| legacy-easy-seed-007 | radius1 | 6 | envelope | True | 73 | 0 | 0 | 0 | 0 | 0 | 0 | 0.356849 | 0 | 0 |
| legacy-easy-seed-007 | radius3 | 3 | site | False | 265 | 0 | 0 | 0 | 0 | 0 | 0.0202055 | 0.189623 | 0 | 0 |
| legacy-easy-seed-007 | radius3 | 3 | envelope | True | 265 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.189623 | 0 | 0 |
| legacy-easy-seed-007 | radius3 | 6 | site | False | 265 | 0 | 0 | 0 | 0 | 0 | 0.0202055 | 0.189623 | 0 | 0 |
| legacy-easy-seed-007 | radius3 | 6 | envelope | True | 265 | 0 | 0 | 0 | 0 | 0 | 7.567 | 0.189623 | 0 | 0 |
| legacy-easy-seed-007 | radius6 | 3 | site | False | 769 | 0 | 0 | 0.018628 | 0 | 0.00780234 | 0.00157747 | 0.0580624 | 0 | 0 |
| legacy-easy-seed-007 | radius6 | 3 | envelope | True | 769 | 0 | 0 | 0.018628 | 0 | 0.00780234 | 1160.83 | 0.0580624 | 0 | 0 |
| legacy-easy-seed-007 | radius6 | 6 | site | False | 769 | 0 | 0 | 0 | 0 | 0.00780234 | 0.00157747 | 0.0580624 | 0 | 0 |
| legacy-easy-seed-007 | radius6 | 6 | envelope | True | 769 | 0 | 0 | 0 | 0 | 0.00780234 | 116.16 | 0.0580624 | 0 | 0 |
| legacy-easy-seed-008 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-008 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-008 | empty | 6 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-008 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-008 | guide | 3 | site | False | 17 | 0 | 0 | 0 | 0 | 0 | 0.02935 | 0.614706 | 0 | 0 |
| legacy-easy-seed-008 | guide | 3 | envelope | True | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0.614706 | 0 | 0 |
| legacy-easy-seed-008 | guide | 6 | site | False | 17 | 0 | 0 | 0 | 0 | 0 | 0.02935 | 0.614706 | 0 | 0 |
| legacy-easy-seed-008 | guide | 6 | envelope | True | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0.614706 | 0 | 0 |
| legacy-easy-seed-008 | scaffold | 3 | site | False | 106 | 0 | 0 | 0.00011471 | 0 | 0 | 0.0259469 | 0.293396 | 0 | 0 |
| legacy-easy-seed-008 | scaffold | 3 | envelope | True | 106 | 0 | 0 | 0.00011471 | 0 | 0 | 29.5477 | 0.293396 | 0 | 0 |
| legacy-easy-seed-008 | scaffold | 6 | site | False | 106 | 0 | 0 | 0 | 0 | 0 | 0.0259469 | 0.293396 | 0 | 0 |
| legacy-easy-seed-008 | scaffold | 6 | envelope | True | 106 | 0 | 0 | 0 | 0 | 0 | 0.924382 | 0.293396 | 0 | 0 |
| legacy-easy-seed-008 | radius1 | 3 | site | False | 53 | 0 | 0 | 0 | 0 | 0 | 0.0279735 | 0.472641 | 0 | 0 |
| legacy-easy-seed-008 | radius1 | 3 | envelope | True | 53 | 0 | 0 | 0 | 0 | 0 | 3.93246 | 0.472641 | 0 | 0 |
| legacy-easy-seed-008 | radius1 | 6 | site | False | 53 | 0 | 0 | 0 | 0 | 0 | 0.0279735 | 0.472641 | 0 | 0 |
| legacy-easy-seed-008 | radius1 | 6 | envelope | True | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0.472641 | 0 | 0 |
| legacy-easy-seed-008 | radius3 | 3 | site | False | 188 | 0 | 0 | 0 | 0 | 0 | 0.0228115 | 0.275532 | 0 | 0 |
| legacy-easy-seed-008 | radius3 | 3 | envelope | True | 188 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.275532 | 0 | 0 |
| legacy-easy-seed-008 | radius3 | 6 | site | False | 188 | 0 | 0 | 0 | 0 | 0 | 0.0228115 | 0.275532 | 0 | 0 |
| legacy-easy-seed-008 | radius3 | 6 | envelope | True | 188 | 0 | 0 | 0 | 0 | 0 | 8.07777 | 0.275532 | 0 | 0 |
| legacy-easy-seed-008 | radius6 | 3 | site | False | 534 | 0 | 0 | 0.0132298 | 0 | 0.00749064 | 0.00958169 | 0.102809 | 0 | 0 |
| legacy-easy-seed-008 | radius6 | 3 | envelope | True | 534 | 0 | 0 | 0.0132298 | 0 | 0.00749064 | 1110.11 | 0.102809 | 0 | 0 |
| legacy-easy-seed-008 | radius6 | 6 | site | False | 534 | 0 | 0 | 0 | 0 | 0.00749064 | 0.00958169 | 0.102809 | 0 | 0 |
| legacy-easy-seed-008 | radius6 | 6 | envelope | True | 534 | 0 | 0 | 0 | 0 | 0.00749064 | 116.16 | 0.102809 | 0 | 0 |
| legacy-easy-seed-009 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-009 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-009 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-009 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-009 | guide | 3 | site | False | 37 | 0 | 0 | 0 | 0 | 0 | 0.0286679 | 0.0662162 | 0 | 0 |
| legacy-easy-seed-009 | guide | 3 | envelope | True | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0662162 | 0 | 0 |
| legacy-easy-seed-009 | guide | 6 | site | True | 37 | 0 | 0 | 0 | 0 | 0 | 0.0286679 | 0.0662162 | 0 | 0 |
| legacy-easy-seed-009 | guide | 6 | envelope | True | 37 | 0 | 0 | 0 | 0 | 0 | 0.00375886 | 0.0662162 | 0 | 0 |
| legacy-easy-seed-009 | scaffold | 3 | site | False | 411 | 0 | 0 | 0.000324021 | 0 | 0 | 0.0152031 | 0.0616788 | 0 | 0 |
| legacy-easy-seed-009 | scaffold | 3 | envelope | True | 411 | 0 | 0 | 0.000324021 | 0 | 0 | 47.1179 | 0.0616788 | 0 | 0 |
| legacy-easy-seed-009 | scaffold | 6 | site | True | 411 | 0 | 0 | 0 | 0 | 0 | 0.0152031 | 0.0616788 | 0 | 0 |
| legacy-easy-seed-009 | scaffold | 6 | envelope | True | 411 | 0 | 0 | 0 | 0 | 0 | 4.41129 | 0.0616788 | 0 | 0 |
| legacy-easy-seed-009 | radius1 | 3 | site | False | 153 | 0 | 0 | 0 | 0 | 0 | 0.0244916 | 0.0526144 | 0 | 0 |
| legacy-easy-seed-009 | radius1 | 3 | envelope | True | 153 | 0 | 0 | 0 | 0 | 0 | 2.66578 | 0.0526144 | 0 | 0 |
| legacy-easy-seed-009 | radius1 | 6 | site | True | 153 | 0 | 0 | 0 | 0 | 0 | 0.0244916 | 0.0526144 | 0 | 0 |
| legacy-easy-seed-009 | radius1 | 6 | envelope | True | 153 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0526144 | 0 | 0 |
| legacy-easy-seed-009 | radius3 | 3 | site | False | 604 | 0 | 0 | 0 | 0 | 0 | 0.00825461 | 0.0205298 | 0 | 0 |
| legacy-easy-seed-009 | radius3 | 3 | envelope | True | 604 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.0205298 | 0 | 0 |
| legacy-easy-seed-009 | radius3 | 6 | site | True | 604 | 0 | 0 | 0 | 0 | 0 | 0.00825461 | 0.0205298 | 0 | 0 |
| legacy-easy-seed-009 | radius3 | 6 | envelope | True | 604 | 0 | 0 | 0 | 0 | 0 | 14.2637 | 0.0205298 | 0 | 0 |
| legacy-easy-seed-009 | radius6 | 3 | site | False | 1410 | 0 | 0 | 0.0290179 | 0 | 0.0787234 | 0 | 0.0102837 | 0 | 0 |
| legacy-easy-seed-009 | radius6 | 3 | envelope | True | 1410 | 0 | 0 | 0.0290179 | 0 | 0.0787234 | 735.56 | 0.0102837 | 0 | 0 |
| legacy-easy-seed-009 | radius6 | 6 | site | True | 1410 | 0 | 0 | 0 | 0 | 0.0787234 | 0 | 0.0102837 | 0 | 0 |
| legacy-easy-seed-009 | radius6 | 6 | envelope | True | 1410 | 0 | 0 | 0 | 0 | 0.0787234 | 116.16 | 0.0102837 | 0 | 0 |
| legacy-easy-seed-010 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-010 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-010 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-010 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-010 | guide | 3 | site | False | 36 | 0 | 0 | 0 | 0 | 0 | 0.0287255 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-010 | guide | 3 | envelope | True | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-010 | guide | 6 | site | True | 36 | 0 | 0 | 0 | 0 | 0 | 0.0287255 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-010 | guide | 6 | envelope | True | 36 | 0 | 0 | 0 | 0 | 0 | 0.00873597 | 0.0722222 | 0 | 0 |
| legacy-easy-seed-010 | scaffold | 3 | site | False | 402 | 0 | 0 | 0.000318629 | 0 | 0 | 0.0157679 | 0.0340796 | 0 | 0 |
| legacy-easy-seed-010 | scaffold | 3 | envelope | True | 402 | 0 | 0 | 0.000318629 | 0 | 0 | 42.0451 | 0.0340796 | 0 | 0 |
| legacy-easy-seed-010 | scaffold | 6 | site | True | 402 | 0 | 0 | 0 | 0 | 0 | 0.0157679 | 0.0340796 | 0 | 0 |
| legacy-easy-seed-010 | scaffold | 6 | envelope | True | 402 | 0 | 0 | 0 | 0 | 0 | 2.06912 | 0.0340796 | 0 | 0 |
| legacy-easy-seed-010 | radius1 | 3 | site | False | 148 | 0 | 0 | 0 | 0 | 0 | 0.0247603 | 0.0391892 | 0 | 0 |
| legacy-easy-seed-010 | radius1 | 3 | envelope | True | 148 | 0 | 0 | 0 | 0 | 0 | 2.12755 | 0.0391892 | 0 | 0 |
| legacy-easy-seed-010 | radius1 | 6 | site | True | 148 | 0 | 0 | 0 | 0 | 0 | 0.0247603 | 0.0391892 | 0 | 0 |
| legacy-easy-seed-010 | radius1 | 6 | envelope | True | 148 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0391892 | 0 | 0 |
| legacy-easy-seed-010 | radius3 | 3 | site | False | 619 | 0 | 0 | 0 | 0 | 0 | 0.00808539 | 0.00185783 | 0 | 0 |
| legacy-easy-seed-010 | radius3 | 3 | envelope | True | 619 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.00185783 | 0 | 0 |
| legacy-easy-seed-010 | radius3 | 6 | site | True | 619 | 0 | 0 | 0 | 0 | 0 | 0.00808539 | 0.00185783 | 0 | 0 |
| legacy-easy-seed-010 | radius3 | 6 | envelope | True | 619 | 0 | 0 | 0 | 0 | 0 | 9.04961 | 0.00185783 | 0 | 0 |
| legacy-easy-seed-010 | radius6 | 3 | site | False | 1693 | 0 | 0 | 0.0380231 | 0 | 0.101004 | 0 | 0 | 0 | 0 |
| legacy-easy-seed-010 | radius6 | 3 | envelope | True | 1693 | 0 | 0 | 0.0380231 | 0 | 0.101004 | 1025.78 | 0 | 0 | 0 |
| legacy-easy-seed-010 | radius6 | 6 | site | True | 1693 | 0 | 0 | 0 | 0 | 0.101004 | 0 | 0 | 0 | 0 |
| legacy-easy-seed-010 | radius6 | 6 | envelope | True | 1693 | 0 | 0 | 0 | 0 | 0.101004 | 116.16 | 0 | 0 | 0 |
| legacy-easy-seed-011 | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-011 | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-011 | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-011 | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| legacy-easy-seed-011 | guide | 3 | site | False | 31 | 0 | 0 | 0 | 0 | 0 | 0.0287872 | 0.0435484 | 0 | 0 |
| legacy-easy-seed-011 | guide | 3 | envelope | True | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0435484 | 0 | 0 |
| legacy-easy-seed-011 | guide | 6 | site | True | 31 | 0 | 0 | 0 | 0 | 0 | 0.0287872 | 0.0435484 | 0 | 0 |
| legacy-easy-seed-011 | guide | 6 | envelope | True | 31 | 0 | 0 | 0 | 0 | 0 | 0.00527911 | 0.0435484 | 0 | 0 |
| legacy-easy-seed-011 | scaffold | 3 | site | False | 243 | 0 | 0 | 0.000117371 | 0 | 0 | 0.020493 | 0.0598765 | 0 | 0 |
| legacy-easy-seed-011 | scaffold | 3 | envelope | True | 243 | 0 | 0 | 0.000117371 | 0 | 0 | 28.029 | 0.0598765 | 0 | 0 |
| legacy-easy-seed-011 | scaffold | 6 | site | True | 243 | 0 | 0 | 0 | 0 | 0 | 0.020493 | 0.0598765 | 0 | 0 |
| legacy-easy-seed-011 | scaffold | 6 | envelope | True | 243 | 0 | 0 | 0 | 0 | 0 | 0.816521 | 0.0598765 | 0 | 0 |
| legacy-easy-seed-011 | radius1 | 3 | site | False | 118 | 0 | 0 | 0 | 0 | 0 | 0.0253834 | 0.0533898 | 0 | 0 |
| legacy-easy-seed-011 | radius1 | 3 | envelope | True | 118 | 0 | 0 | 0 | 0 | 0 | 3.29368 | 0.0533898 | 0 | 0 |
| legacy-easy-seed-011 | radius1 | 6 | site | True | 118 | 0 | 0 | 0 | 0 | 0 | 0.0253834 | 0.0533898 | 0 | 0 |
| legacy-easy-seed-011 | radius1 | 6 | envelope | True | 118 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0533898 | 0 | 0 |
| legacy-easy-seed-011 | radius3 | 3 | site | False | 440 | 0 | 0 | 0 | 0 | 0 | 0.0127856 | 0.0409091 | 0 | 0 |
| legacy-easy-seed-011 | radius3 | 3 | envelope | True | 440 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.0409091 | 0 | 0 |
| legacy-easy-seed-011 | radius3 | 6 | site | True | 440 | 0 | 0 | 0 | 0 | 0 | 0.0127856 | 0.0409091 | 0 | 0 |
| legacy-easy-seed-011 | radius3 | 6 | envelope | True | 440 | 0 | 0 | 0 | 0 | 0 | 7.99564 | 0.0409091 | 0 | 0 |
| legacy-easy-seed-011 | radius6 | 3 | site | False | 1254 | 0 | 0 | 0.0318466 | 0 | 0.0255183 | 0 | 0.0134769 | 0 | 0 |
| legacy-easy-seed-011 | radius6 | 3 | envelope | True | 1254 | 0 | 0 | 0.0318466 | 0 | 0.0255183 | 1117.93 | 0.0134769 | 0 | 0 |
| legacy-easy-seed-011 | radius6 | 6 | site | True | 1254 | 0 | 0 | 0 | 0 | 0.0255183 | 0 | 0.0134769 | 0 | 0 |
| legacy-easy-seed-011 | radius6 | 6 | envelope | True | 1254 | 0 | 0 | 0 | 0 | 0.0255183 | 116.16 | 0.0134769 | 0 | 0 |
| ref-01-ground-pair | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-01-ground-pair | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-01-ground-pair | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-01-ground-pair | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-01-ground-pair | guide | 3 | site | False | 36 | 0 | 0 | 0 | 0 | 0 | 0.0285465 | 0 | 0 | 0 |
| ref-01-ground-pair | guide | 3 | envelope | True | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-01-ground-pair | guide | 6 | site | True | 36 | 0 | 0 | 0 | 0 | 0 | 0.0285465 | 0 | 0 | 0 |
| ref-01-ground-pair | guide | 6 | envelope | True | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-01-ground-pair | scaffold | 3 | site | False | 266 | 0 | 0 | 0.000322997 | 0 | 0 | 0.0192603 | 0 | 0 | 0 |
| ref-01-ground-pair | scaffold | 3 | envelope | True | 266 | 0 | 0 | 0.000322997 | 0 | 0 | 36.8638 | 0 | 0 | 0 |
| ref-01-ground-pair | scaffold | 6 | site | True | 266 | 0 | 0 | 0 | 0 | 0 | 0.0192603 | 0 | 0 | 0 |
| ref-01-ground-pair | scaffold | 6 | envelope | True | 266 | 0 | 0 | 0 | 0 | 0 | 4.259 | 0 | 0 | 0 |
| ref-01-ground-pair | radius1 | 3 | site | False | 132 | 0 | 0 | 0 | 0 | 0 | 0.0246705 | 0 | 0 | 0 |
| ref-01-ground-pair | radius1 | 3 | envelope | True | 132 | 0 | 0 | 0 | 0 | 0 | 5.16463 | 0 | 0 | 0 |
| ref-01-ground-pair | radius1 | 6 | site | True | 132 | 0 | 0 | 0 | 0 | 0 | 0.0246705 | 0 | 0 | 0 |
| ref-01-ground-pair | radius1 | 6 | envelope | True | 132 | 0 | 0 | 0 | 0 | 0 | 0.0805067 | 0 | 0 | 0 |
| ref-01-ground-pair | radius3 | 3 | site | False | 432 | 0 | 0 | 0 | 0 | 0 | 0.0125581 | 0 | 0 | 0 |
| ref-01-ground-pair | radius3 | 3 | envelope | True | 432 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0 | 0 | 0 |
| ref-01-ground-pair | radius3 | 6 | site | True | 432 | 0 | 0 | 0 | 0 | 0 | 0.0125581 | 0 | 0 | 0 |
| ref-01-ground-pair | radius3 | 6 | envelope | True | 432 | 0 | 0 | 0 | 0 | 0 | 18.2227 | 0 | 0 | 0 |
| ref-01-ground-pair | radius6 | 3 | site | False | 922 | 0 | 0 | 0.0197836 | 0 | 0.0130152 | 0 | 0 | 0 | 0 |
| ref-01-ground-pair | radius6 | 3 | envelope | True | 922 | 0 | 0 | 0.0197836 | 0 | 0.0130152 | 608.586 | 0 | 0 | 0 |
| ref-01-ground-pair | radius6 | 6 | site | True | 922 | 0 | 0 | 0 | 0 | 0.0130152 | 0 | 0 | 0 | 0 |
| ref-01-ground-pair | radius6 | 6 | envelope | True | 922 | 0 | 0 | 0 | 0 | 0.0130152 | 116.16 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | empty | 3 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-02-facade-pair-and-ground | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-02-facade-pair-and-ground | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-02-facade-pair-and-ground | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-02-facade-pair-and-ground | guide | 3 | site | True | 55 | 0 | 0 | 0 | 0 | 0 | 0.0277794 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | guide | 3 | envelope | True | 55 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | guide | 6 | site | True | 55 | 0 | 0 | 0 | 0 | 0 | 0.0277794 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | guide | 6 | envelope | True | 55 | 0 | 0 | 0 | 0 | 0 | 0.0029064 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 3 | site | True | 481 | 0 | 0 | 0.000363372 | 0 | 0 | 0.0105798 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 3 | envelope | True | 481 | 0 | 0 | 0.000363372 | 0 | 0 | 34.097 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 6 | site | True | 481 | 0 | 0 | 0 | 0 | 0 | 0.0105798 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 6 | envelope | True | 481 | 0 | 0 | 0 | 0 | 0 | 2.05145 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 3 | site | True | 215 | 0 | 0 | 0 | 0 | 0 | 0.0213194 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 3 | envelope | True | 215 | 0 | 0 | 0 | 0 | 0 | 3.23031 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 6 | site | True | 215 | 0 | 0 | 0 | 0 | 0 | 0.0213194 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 6 | envelope | True | 215 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 3 | site | True | 806 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 3 | envelope | True | 806 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 6 | site | True | 806 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 6 | envelope | True | 806 | 0 | 0 | 0 | 0 | 0 | 11.513 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 3 | site | True | 2030 | 0 | 0 | 0.0494186 | 0 | 0.100493 | 0 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 3 | envelope | True | 2030 | 0 | 0 | 0.0494186 | 0 | 0.100493 | 863 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 6 | site | True | 2030 | 0 | 0 | 0 | 0 | 0.100493 | 0 | 0 | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 6 | envelope | True | 2030 | 0 | 0 | 0 | 0 | 0.100493 | 116.16 | 0 | 0 | 0 |
| ref-03-wide-gap | empty | 3 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-03-wide-gap | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-03-wide-gap | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-03-wide-gap | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-03-wide-gap | guide | 3 | site | True | 51 | 0 | 0 | 0 | 0 | 0 | 0.0280053 | 0.00686274 | 0 | 0 |
| ref-03-wide-gap | guide | 3 | envelope | True | 51 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00686274 | 0 | 0 |
| ref-03-wide-gap | guide | 6 | site | True | 51 | 0 | 0 | 0 | 0 | 0 | 0.0280053 | 0.00686274 | 0 | 0 |
| ref-03-wide-gap | guide | 6 | envelope | True | 51 | 0 | 0 | 0 | 0 | 0 | 0.0051704 | 0.00686274 | 0 | 0 |
| ref-03-wide-gap | scaffold | 3 | site | True | 481 | 0 | 0 | 0.000234668 | 0 | 0 | 0.0111874 | 0.0267152 | 0 | 0 |
| ref-03-wide-gap | scaffold | 3 | envelope | True | 481 | 0 | 0 | 0.000234668 | 0 | 0 | 33.8861 | 0.0267152 | 0 | 0 |
| ref-03-wide-gap | scaffold | 6 | site | True | 481 | 0 | 0 | 0 | 0 | 0 | 0.0111874 | 0.0267152 | 0 | 0 |
| ref-03-wide-gap | scaffold | 6 | envelope | True | 481 | 0 | 0 | 0 | 0 | 0 | 1.95547 | 0.0267152 | 0 | 0 |
| ref-03-wide-gap | radius1 | 3 | site | True | 206 | 0 | 0 | 0 | 0 | 0 | 0.0219431 | 0.0296116 | 0 | 0 |
| ref-03-wide-gap | radius1 | 3 | envelope | True | 206 | 0 | 0 | 0 | 0 | 0 | 2.73175 | 0.0296116 | 0 | 0 |
| ref-03-wide-gap | radius1 | 6 | site | True | 206 | 0 | 0 | 0 | 0 | 0 | 0.0219431 | 0.0296116 | 0 | 0 |
| ref-03-wide-gap | radius1 | 6 | envelope | True | 206 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0296116 | 0 | 0 |
| ref-03-wide-gap | radius3 | 3 | site | True | 808 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00965346 | 0 | 0 |
| ref-03-wide-gap | radius3 | 3 | envelope | True | 808 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.00965346 | 0 | 0 |
| ref-03-wide-gap | radius3 | 6 | site | True | 808 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00965346 | 0 | 0 |
| ref-03-wide-gap | radius3 | 6 | envelope | True | 808 | 0 | 0 | 0 | 0 | 0 | 11.2104 | 0.00965346 | 0 | 0 |
| ref-03-wide-gap | radius6 | 3 | site | True | 2054 | 0 | 0 | 0.0487328 | 0 | 0.103213 | 0 | 0 | 0 | 0 |
| ref-03-wide-gap | radius6 | 3 | envelope | True | 2054 | 0 | 0 | 0.0487328 | 0 | 0.103213 | 879.97 | 0 | 0 | 0 |
| ref-03-wide-gap | radius6 | 6 | site | True | 2054 | 0 | 0 | 0 | 0 | 0.103213 | 0 | 0 | 0 | 0 |
| ref-03-wide-gap | radius6 | 6 | envelope | True | 2054 | 0 | 0 | 0 | 0 | 0.103213 | 116.16 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | empty | 3 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-04-asymmetric-heights | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-04-asymmetric-heights | empty | 6 | site | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-04-asymmetric-heights | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-04-asymmetric-heights | guide | 3 | site | True | 69 | 0 | 0 | 0 | 0 | 0 | 0.0273013 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | guide | 3 | envelope | True | 69 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | guide | 6 | site | True | 69 | 0 | 0 | 0 | 0 | 0 | 0.0273013 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | guide | 6 | envelope | True | 69 | 0 | 0 | 0 | 0 | 0 | 0.0046603 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | scaffold | 3 | site | True | 649 | 0 | 0 | 0.000469337 | 0 | 0 | 0.00461671 | 0.0626348 | 0 | 0 |
| ref-04-asymmetric-heights | scaffold | 3 | envelope | True | 649 | 0 | 0 | 0.000469337 | 0 | 0 | 37.2148 | 0.0626348 | 0 | 0 |
| ref-04-asymmetric-heights | scaffold | 6 | site | True | 649 | 0 | 0 | 0 | 0 | 0 | 0.00461671 | 0.0626348 | 0 | 0 |
| ref-04-asymmetric-heights | scaffold | 6 | envelope | True | 649 | 0 | 0 | 0 | 0 | 0 | 2.10066 | 0.0626348 | 0 | 0 |
| ref-04-asymmetric-heights | radius1 | 3 | site | True | 280 | 0 | 0 | 0 | 0 | 0 | 0.0190488 | 0.0178571 | 0 | 0 |
| ref-04-asymmetric-heights | radius1 | 3 | envelope | True | 280 | 0 | 0 | 0 | 0 | 0 | 3.22667 | 0.0178571 | 0 | 0 |
| ref-04-asymmetric-heights | radius1 | 6 | site | True | 280 | 0 | 0 | 0 | 0 | 0 | 0.0190488 | 0.0178571 | 0 | 0 |
| ref-04-asymmetric-heights | radius1 | 6 | envelope | True | 280 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0178571 | 0 | 0 |
| ref-04-asymmetric-heights | radius3 | 3 | site | True | 1050 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0157143 | 0 | 0 |
| ref-04-asymmetric-heights | radius3 | 3 | envelope | True | 1050 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.0157143 | 0 | 0 |
| ref-04-asymmetric-heights | radius3 | 6 | site | True | 1050 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0157143 | 0 | 0 |
| ref-04-asymmetric-heights | radius3 | 6 | envelope | True | 1050 | 0 | 0 | 0 | 0 | 0 | 10.5818 | 0.0157143 | 0 | 0 |
| ref-04-asymmetric-heights | radius6 | 3 | site | True | 2723 | 0 | 0 | 0.0654334 | 0 | 0.0918105 | 0 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | radius6 | 3 | envelope | True | 2723 | 0 | 0 | 0.0654334 | 0 | 0.0918105 | 917.607 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | radius6 | 6 | site | True | 2723 | 0 | 0 | 0 | 0 | 0.0918105 | 0 | 0 | 0 | 0 |
| ref-04-asymmetric-heights | radius6 | 6 | envelope | True | 2723 | 0 | 0 | 0 | 0 | 0.0918105 | 116.16 | 0 | 0 | 0 |
| ref-05-sealed-partition | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-05-sealed-partition | empty | 3 | envelope | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-05-sealed-partition | empty | 6 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-05-sealed-partition | empty | 6 | envelope | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-05-sealed-partition | guide | 3 | site | False | 16 | 0 | 0 | 0 | 0 | 0 | 0.0286979 | 0.35 | 1 | 0 |
| ref-05-sealed-partition | guide | 3 | envelope | False | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0.35 | 1 | 0 |
| ref-05-sealed-partition | guide | 6 | site | False | 16 | 0 | 0 | 0 | 0 | 0 | 0.0286979 | 0.35 | 1 | 0 |
| ref-05-sealed-partition | guide | 6 | envelope | False | 16 | 0 | 0 | 0 | 0 | 0 | 0.0108612 | 0.35 | 1 | 0 |
| ref-05-sealed-partition | scaffold | 3 | site | False | 144 | 0 | 0 | 0.000651042 | 0 | 0 | 0.0182812 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | scaffold | 3 | envelope | False | 144 | 0 | 0 | 0.000651042 | 0 | 0 | 34.56 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | scaffold | 6 | site | False | 144 | 0 | 0 | 0 | 0 | 0 | 0.0182812 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | scaffold | 6 | envelope | False | 144 | 0 | 0 | 0 | 0 | 0 | 0.409491 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | radius1 | 3 | site | False | 56 | 0 | 0 | 0 | 0 | 0 | 0.0254427 | 0.278571 | 1 | 0 |
| ref-05-sealed-partition | radius1 | 3 | envelope | False | 56 | 0 | 0 | 0 | 0 | 0 | 1.92667 | 0.278571 | 1 | 0 |
| ref-05-sealed-partition | radius1 | 6 | site | False | 56 | 0 | 0 | 0 | 0 | 0 | 0.0254427 | 0.278571 | 1 | 0 |
| ref-05-sealed-partition | radius1 | 6 | envelope | False | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0.278571 | 1 | 0 |
| ref-05-sealed-partition | radius3 | 3 | site | False | 240 | 0 | 0 | 0 | 0 | 0 | 0.0104687 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | radius3 | 3 | envelope | False | 240 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | radius3 | 6 | site | False | 240 | 0 | 0 | 0 | 0 | 0 | 0.0104687 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | radius3 | 6 | envelope | False | 240 | 0 | 0 | 0 | 0 | 0 | 4.18743 | 0.183333 | 1 | 0 |
| ref-05-sealed-partition | radius6 | 3 | site | False | 836 | 0 | 0 | 0.0485026 | 0 | 0 | 0 | 0.161005 | 1 | 0 |
| ref-05-sealed-partition | radius6 | 3 | envelope | False | 836 | 0 | 0 | 0.0485026 | 0 | 0 | 1696.8 | 0.161005 | 1 | 0 |
| ref-05-sealed-partition | radius6 | 6 | site | False | 836 | 0 | 0 | 0 | 0 | 0 | 0 | 0.161005 | 1 | 0 |
| ref-05-sealed-partition | radius6 | 6 | envelope | False | 836 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0.161005 | 1 | 0 |
| ref-06-minimal-smoke | empty | 3 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-06-minimal-smoke | empty | 3 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-06-minimal-smoke | empty | 6 | site | False | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-06-minimal-smoke | empty | 6 | envelope | True | 0 | 0 | 1 | 0 | 0 | 0 | 0.03 | 0 | 1 | 0 |
| ref-06-minimal-smoke | guide | 3 | site | False | 32 | 0 | 0 | 0 | 0 | 0 | 0.0289119 | 0 | 0 | 0 |
| ref-06-minimal-smoke | guide | 3 | envelope | True | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-06-minimal-smoke | guide | 6 | site | False | 32 | 0 | 0 | 0 | 0 | 0 | 0.0289119 | 0 | 0 | 0 |
| ref-06-minimal-smoke | guide | 6 | envelope | True | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 3 | site | False | 230 | 0 | 0 | 0.000272035 | 0 | 0 | 0.022179 | 0 | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 3 | envelope | True | 230 | 0 | 0 | 0.000272035 | 0 | 0 | 38.2537 | 0 | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 6 | site | False | 230 | 0 | 0 | 0 | 0 | 0 | 0.022179 | 0 | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 6 | envelope | True | 230 | 0 | 0 | 0 | 0 | 0 | 5.36608 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 3 | site | False | 116 | 0 | 0 | 0 | 0 | 0 | 0.0260555 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 3 | envelope | True | 116 | 0 | 0 | 0 | 0 | 0 | 5.71648 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 6 | site | False | 116 | 0 | 0 | 0 | 0 | 0 | 0.0260555 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 6 | envelope | True | 116 | 0 | 0 | 0 | 0 | 0 | 0.193472 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 3 | site | False | 368 | 0 | 0 | 0 | 0 | 0 | 0.0174864 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 3 | envelope | True | 368 | 0 | 0 | 0 | 0 | 0 | 116.16 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 6 | site | False | 368 | 0 | 0 | 0 | 0 | 0 | 0.0174864 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 6 | envelope | True | 368 | 0 | 0 | 0 | 0 | 0 | 21.0514 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 3 | site | False | 744 | 0 | 0 | 0.0127856 | 0 | 0.0107527 | 0.00470076 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 3 | envelope | True | 744 | 0 | 0 | 0.0127856 | 0 | 0.0107527 | 542.492 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 6 | site | False | 744 | 0 | 0 | 0 | 0 | 0.0107527 | 0.00470076 | 0 | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 6 | envelope | True | 744 | 0 | 0 | 0 | 0 | 0.0107527 | 116.16 | 0 | 0 | 0 |
