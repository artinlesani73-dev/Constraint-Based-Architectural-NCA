# K1 actual model-gradient calibration

Run `20260923T092355Z_f657f2f3bdb9`; source `f56932c32dff835fdf74b31d9acd83c13a8bce20`.

71 model cases and51 controlled material-budget probes. Zero optimizer updates. Registered hashes, all parameter-vector norms/cosines and pre-clamp hinge derivatives independently checked. Every one of17 feasible scenes has seeds0/1 at4/16 steps; three named scenes additionally have seed0 at50 steps. The sealed reference is explicitly excluded from this feasible-scene calibration, not reclassified.

## Gradient scales by horizon

| Steps | Term | Nonzero / cases | Median value | Median parameter norm | Maximum norm |
|---:|---|---:|---:|---:|---:|
| 4 | legality | 0/34 | 0 | 0 | 0 |
| 4 | coverage | 34/34 | 0.7310545 | 0.789122 | 1.083721 |
| 4 | spill | 18/34 | 2.486792e-08 | 2.149788e-05 | 0.01432929 |
| 4 | ground | 0/34 | 0 | 0 | 0 |
| 4 | thickness | 0/34 | 0 | 0 | 0 |
| 4 | sparsity | 0/34 | 0 | 0 | 0 |
| 4 | facade | 32/34 | 0.03708275 | 0.528606 | 5.328752 |
| 4 | access | 30/34 | 0.7950195 | 0.8637698 | 2.788246 |
| 4 | support | 34/34 | 0.7136053 | 0.8177902 | 2.803078 |
| 4 | density_binary | 34/34 | 0.001643621 | 0.00365251 | 0.01575919 |
| 4 | tv | 34/34 | 0.004540724 | 0.01924114 | 0.06251556 |
| 4 | cantilever_boundary | 32/34 | 0.001205868 | 0.001791435 | 0.002640752 |
| 4 | cantilever_historical | 34/34 | 0.001369534 | 0.003152624 | 0.01183906 |
| 16 | legality | 0/34 | 0 | 0 | 0 |
| 16 | coverage | 34/34 | 0.09405031 | 1.403227 | 2.869332 |
| 16 | spill | 10/34 | 0 | 0 | 0.1664262 |
| 16 | ground | 0/34 | 0 | 0 | 0 |
| 16 | thickness | 0/34 | 0 | 0 | 0 |
| 16 | sparsity | 33/34 | 1.266452 | 12.78243 | 70.70196 |
| 16 | facade | 33/34 | 0.07696888 | 0.4708271 | 15.90565 |
| 16 | access | 9/34 | 0.2003521 | 0 | 16.78185 |
| 16 | support | 34/34 | 0.0575391 | 1.194555 | 7.292281 |
| 16 | density_binary | 34/34 | 0.0005301676 | 0.009575433 | 0.0853526 |
| 16 | tv | 34/34 | 0.01324411 | 0.02696568 | 0.4369938 |
| 16 | cantilever_boundary | 30/34 | 0.001113577 | 0.001502107 | 0.002600047 |
| 16 | cantilever_historical | 34/34 | 0.00195927 | 0.00332883 | 0.01901118 |
| 50 | legality | 0/3 | 0 | 0 | 0 |
| 50 | coverage | 2/3 | 0.6910001 | 0.09808723 | 0.1149514 |
| 50 | spill | 1/3 | 0.005424147 | 0 | 0.01726367 |
| 50 | ground | 0/3 | 0 | 0 | 0 |
| 50 | thickness | 0/3 | 0 | 0 | 0 |
| 50 | sparsity | 2/3 | 9.337533 | 202.2336 | 581.007 |
| 50 | facade | 2/3 | 0.4562436 | 1.810999 | 6.425225 |
| 50 | access | 0/3 | 1 | 0 | 0 |
| 50 | support | 2/3 | 0.0005369348 | 0.5413666 | 1.412659 |
| 50 | density_binary | 2/3 | 1.361496e-05 | 0.01357395 | 0.03293098 |
| 50 | tv | 2/3 | 0.02054863 | 0.07725598 | 0.3016054 |
| 50 | cantilever_boundary | 0/3 | 0.0003662109 | 0 | 0 |
| 50 | cantilever_historical | 2/3 | 0.0009694355 | 3.647826e-05 | 0.1380438 |

Zero gradients may indicate correct inactivity/saturation or a blocked derivative. They must be interpreted alongside raw values and state. Hard legality/ground are enforced by projection. Density and TV retain notebook definitions; both cantilever variants are diagnostic, never implicitly added together.

## Budget probes

| Envelope occupancy | Expected mass penalty | Measured range | Derivative direction |
|---:|---:|---:|---|
| 0.015 | 0.015 | 0.015 - 0.015 | increase mass |
| 0.075 | 0 | 0 - 0 | inactive |
| 0.2 | 0.96 | 0.9599997 - 0.9600009 | decrease mass |

All17 scenes reproduce each branch. These synthetic fields are intentionally below/in/above budget, not NCA outputs or successful designs.

## Pre-clamp coverage saturation and binary outcomes

| Scene | Seed | Steps | Guide raw below0 | Guide raw >=1 | Maximum raw | Mass/envelope | Binary connected |
|---|---:|---:|---:|---:|---:|---:|---|
| legacy-easy-seed-000 | 0 | 4 | 0.000% | 0.000% | 0.42089 | 0.061606 | False |
| legacy-easy-seed-000 | 0 | 16 | 0.000% | 68.182% | 1.21415 | 0.207673 | True |
| legacy-easy-seed-000 | 1 | 4 | 0.000% | 0.000% | 0.37984 | 0.059726 | False |
| legacy-easy-seed-000 | 1 | 16 | 0.000% | 59.091% | 1.20818 | 0.211171 | True |
| legacy-easy-seed-000 | 0 | 50 | 0.000% | 100.000% | 1.23855 | 0.227951 | True |
| legacy-easy-seed-001 | 0 | 4 | 0.000% | 0.000% | 0.45607 | 0.065546 | False |
| legacy-easy-seed-001 | 0 | 16 | 0.000% | 56.522% | 1.21112 | 0.218773 | True |
| legacy-easy-seed-001 | 1 | 4 | 0.000% | 0.000% | 0.33945 | 0.064494 | False |
| legacy-easy-seed-001 | 1 | 16 | 0.000% | 47.826% | 1.19978 | 0.220476 | True |
| legacy-easy-seed-002 | 0 | 4 | 0.000% | 0.000% | 0.42215 | 0.091441 | False |
| legacy-easy-seed-002 | 0 | 16 | 0.000% | 65.789% | 1.22356 | 0.304640 | True |
| legacy-easy-seed-002 | 1 | 4 | 0.000% | 0.000% | 0.43394 | 0.089451 | False |
| legacy-easy-seed-002 | 1 | 16 | 0.000% | 68.421% | 1.20706 | 0.305660 | True |
| legacy-easy-seed-003 | 0 | 4 | 0.000% | 0.000% | 0.42211 | 0.070765 | False |
| legacy-easy-seed-003 | 0 | 16 | 0.000% | 76.471% | 1.22354 | 0.235536 | True |
| legacy-easy-seed-003 | 1 | 4 | 0.000% | 0.000% | 0.43386 | 0.069683 | False |
| legacy-easy-seed-003 | 1 | 16 | 0.000% | 67.647% | 1.20779 | 0.238049 | True |
| legacy-easy-seed-004 | 0 | 4 | 0.000% | 0.000% | 0.42738 | 0.064267 | False |
| legacy-easy-seed-004 | 0 | 16 | 15.625% | 37.500% | 1.19694 | 0.209550 | False |
| legacy-easy-seed-004 | 1 | 4 | 0.000% | 0.000% | 0.37852 | 0.065421 | False |
| legacy-easy-seed-004 | 1 | 16 | 9.375% | 71.875% | 1.20562 | 0.216061 | False |
| legacy-easy-seed-005 | 0 | 4 | 0.000% | 0.000% | 0.41966 | 0.071873 | False |
| legacy-easy-seed-005 | 0 | 16 | 0.000% | 55.556% | 1.19674 | 0.237420 | True |
| legacy-easy-seed-005 | 1 | 4 | 0.000% | 0.000% | 0.37902 | 0.071835 | False |
| legacy-easy-seed-005 | 1 | 16 | 0.000% | 69.444% | 1.20228 | 0.242683 | True |
| legacy-easy-seed-006 | 0 | 4 | 0.000% | 0.000% | 0.37964 | 0.056805 | False |
| legacy-easy-seed-006 | 0 | 16 | 0.000% | 69.697% | 1.20323 | 0.188817 | True |
| legacy-easy-seed-006 | 1 | 4 | 0.000% | 0.000% | 0.40058 | 0.055732 | False |
| legacy-easy-seed-006 | 1 | 16 | 0.000% | 75.758% | 1.20680 | 0.188708 | True |
| legacy-easy-seed-007 | 0 | 4 | 0.000% | 0.000% | 0.44396 | 0.066852 | False |
| legacy-easy-seed-007 | 0 | 16 | 0.000% | 81.818% | 1.22487 | 0.219556 | True |
| legacy-easy-seed-007 | 1 | 4 | 0.000% | 0.000% | 0.39341 | 0.064739 | False |
| legacy-easy-seed-007 | 1 | 16 | 0.000% | 90.909% | 1.23157 | 0.217466 | True |
| legacy-easy-seed-008 | 0 | 4 | 0.000% | 0.000% | 0.36282 | 0.052882 | False |
| legacy-easy-seed-008 | 0 | 16 | 11.765% | 47.059% | 1.21133 | 0.169646 | False |
| legacy-easy-seed-008 | 1 | 4 | 0.000% | 0.000% | 0.36432 | 0.052854 | False |
| legacy-easy-seed-008 | 1 | 16 | 11.765% | 58.824% | 1.23315 | 0.168816 | False |
| legacy-easy-seed-009 | 0 | 4 | 0.000% | 0.000% | 0.46037 | 0.081746 | False |
| legacy-easy-seed-009 | 0 | 16 | 0.000% | 78.378% | 1.22014 | 0.273005 | True |
| legacy-easy-seed-009 | 1 | 4 | 0.000% | 0.000% | 0.46115 | 0.081118 | False |
| legacy-easy-seed-009 | 1 | 16 | 0.000% | 67.568% | 1.22042 | 0.273976 | True |
| legacy-easy-seed-010 | 0 | 4 | 0.000% | 0.000% | 0.42862 | 0.067098 | False |
| legacy-easy-seed-010 | 0 | 16 | 0.000% | 66.667% | 1.20729 | 0.223006 | True |
| legacy-easy-seed-010 | 1 | 4 | 0.000% | 0.000% | 0.40070 | 0.065007 | False |
| legacy-easy-seed-010 | 1 | 16 | 0.000% | 63.889% | 1.21224 | 0.221529 | True |
| legacy-easy-seed-011 | 0 | 4 | 0.000% | 0.000% | 0.42557 | 0.054035 | False |
| legacy-easy-seed-011 | 0 | 16 | 0.000% | 67.742% | 1.19927 | 0.181995 | True |
| legacy-easy-seed-011 | 1 | 4 | 0.000% | 0.000% | 0.37824 | 0.052558 | False |
| legacy-easy-seed-011 | 1 | 16 | 0.000% | 54.839% | 1.21315 | 0.179398 | True |
| ref-01-ground-pair | 0 | 4 | 11.111% | 0.000% | 0.38514 | 0.055531 | False |
| ref-01-ground-pair | 0 | 16 | 36.111% | 13.889% | 1.19643 | 0.165637 | False |
| ref-01-ground-pair | 1 | 4 | 13.889% | 0.000% | 0.38035 | 0.053268 | False |
| ref-01-ground-pair | 1 | 16 | 44.444% | 16.667% | 1.19149 | 0.162472 | False |
| ref-01-ground-pair | 0 | 50 | 30.556% | 33.333% | 1.21907 | 0.493010 | False |
| ref-02-facade-pair-and-ground | 0 | 4 | 3.636% | 0.000% | 0.39079 | 0.061299 | False |
| ref-02-facade-pair-and-ground | 0 | 16 | 12.727% | 50.909% | 1.20939 | 0.209125 | False |
| ref-02-facade-pair-and-ground | 1 | 4 | 1.818% | 0.000% | 0.43368 | 0.060785 | False |
| ref-02-facade-pair-and-ground | 1 | 16 | 16.364% | 50.909% | 1.21615 | 0.210818 | False |
| ref-03-wide-gap | 0 | 4 | 3.922% | 0.000% | 0.38721 | 0.060691 | False |
| ref-03-wide-gap | 0 | 16 | 13.725% | 52.941% | 1.21667 | 0.202897 | False |
| ref-03-wide-gap | 1 | 4 | 7.843% | 0.000% | 0.46599 | 0.061675 | False |
| ref-03-wide-gap | 1 | 16 | 11.765% | 58.824% | 1.23274 | 0.205296 | False |
| ref-04-asymmetric-heights | 0 | 4 | 2.899% | 0.000% | 0.38693 | 0.062602 | False |
| ref-04-asymmetric-heights | 0 | 16 | 14.493% | 49.275% | 1.20994 | 0.212596 | False |
| ref-04-asymmetric-heights | 1 | 4 | 0.000% | 0.000% | 0.43348 | 0.061845 | False |
| ref-04-asymmetric-heights | 1 | 16 | 17.391% | 59.420% | 1.20299 | 0.213379 | False |
| ref-06-minimal-smoke | 0 | 4 | 15.625% | 0.000% | 0.38418 | 0.050710 | False |
| ref-06-minimal-smoke | 0 | 16 | 31.250% | 6.250% | 1.18643 | 0.121884 | False |
| ref-06-minimal-smoke | 1 | 4 | 12.500% | 0.000% | 0.31114 | 0.050195 | False |
| ref-06-minimal-smoke | 1 | 16 | 50.000% | 3.125% | 1.00000 | 0.119281 | False |
| ref-06-minimal-smoke | 0 | 50 | 37.500% | 25.000% | 1.21904 | 0.369500 | False |

The hinge has no gradient where raw material >=1; the saved raw derivatives match that definition. Continued coverage pressure elsewhere does not imply prevention of all overshoot or correction of projected-access gradient failure.

## Pairwise opposition, common4/16-step matrix

| Pair | Cosine <-0.1 / defined | Median cosine |
|---|---:|---:|
| coverage / spill | 28/28 | -0.57086 |
| coverage / sparsity | 33/33 | -0.86641 |
| coverage / facade | 64/65 | -0.49837 |
| coverage / support | 10/68 | 0.88574 |
| coverage / density_binary | 44/68 | -0.71624 |
| coverage / tv | 47/68 | -0.67420 |
| coverage / cantilever_boundary | 62/62 | -0.86885 |
| coverage / cantilever_historical | 31/68 | 0.21779 |
| spill / facade | 4/26 | 0.99277 |
| spill / access | 15/15 | -0.51118 |
| spill / support | 10/28 | 0.47418 |
| spill / cantilever_historical | 12/28 | 0.06588 |
| sparsity / access | 9/9 | -0.64778 |
| sparsity / support | 24/33 | -0.99613 |
| sparsity / density_binary | 24/33 | -0.99349 |
| sparsity / tv | 21/33 | -0.68611 |
| sparsity / cantilever_historical | 9/33 | 0.59605 |
| facade / access | 35/37 | -0.32876 |
| facade / support | 46/65 | -0.45394 |
| facade / density_binary | 22/65 | 0.47120 |
| facade / tv | 17/65 | 0.43144 |
| facade / cantilever_boundary | 5/59 | 0.49312 |
| facade / cantilever_historical | 25/65 | 0.08379 |
| access / support | 1/39 | 0.63716 |
| access / density_binary | 31/39 | -0.59339 |
| access / tv | 33/39 | -0.60020 |
| access / cantilever_boundary | 37/37 | -0.60426 |
| access / cantilever_historical | 11/39 | 0.54877 |
| support / density_binary | 30/68 | 0.75489 |
| support / tv | 32/68 | 0.53054 |
| support / cantilever_boundary | 52/62 | -0.93320 |
| support / cantilever_historical | 25/68 | 0.33908 |
| density_binary / cantilever_boundary | 22/62 | 0.72096 |
| density_binary / cantilever_historical | 47/68 | -0.84559 |
| tv / cantilever_boundary | 22/62 | 0.70948 |
| tv / cantilever_historical | 50/68 | -0.87742 |
| cantilever_boundary / cantilever_historical | 30/62 | 0.01818 |

Cosines are local parameter derivatives, not proof of global incompatibility. Inactive pairs are undefined and excluded. Longer50-step cases are not pooled into this paired17-scene comparison.

## Original checkpoint recipe provenance

The saved checkpoint weights equal the notebook trainer table exactly. K1 stores the exact cell19 regularizer class sources and notebook/checkpoint hashes. Historical coefficients are evidence, not calibration of corrected objective scales.

| Historical key | Coefficient |
|---|---:|
| access_conn | 15.0 |
| cantilever | 5.0 |
| coverage | 25.0 |
| density | 3.0 |
| facade | 10.0 |
| ground | 35.0 |
| legality | 30.0 |
| loadpath | 8.0 |
| sparsity | 30.0 |
| spill | 25.0 |
| thickness | 30.0 |
| tv | 1.0 |

## All individual values and parameter norms

| Scene | Seed | Steps | Term | Value | Parameter norm |
|---|---:|---:|---|---:|---:|
| legacy-easy-seed-000 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-000 | 0 | 4 | coverage | 0.70690775 | 1.0837207 |
| legacy-easy-seed-000 | 0 | 4 | spill | 6.2236246e-08 | 2.9993143e-05 |
| legacy-easy-seed-000 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-000 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-000 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-000 | 0 | 4 | facade | 0.11039644 | 0.68047262 |
| legacy-easy-seed-000 | 0 | 4 | access | 0.75363076 | 0.90040152 |
| legacy-easy-seed-000 | 0 | 4 | support | 0.73350769 | 0.90547197 |
| legacy-easy-seed-000 | 0 | 4 | density_binary | 0.00099404086 | 0.0018881359 |
| legacy-easy-seed-000 | 0 | 4 | tv | 0.0026720027 | 0.010890676 |
| legacy-easy-seed-000 | 0 | 4 | cantilever_boundary | 0.0005953284 | 0.00070084515 |
| legacy-easy-seed-000 | 0 | 4 | cantilever_historical | 0.00060015806 | 0.0018611069 |
| legacy-easy-seed-000 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-000 | 0 | 16 | coverage | 0.058520079 | 1.8622848 |
| legacy-easy-seed-000 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-000 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-000 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-000 | 0 | 16 | sparsity | 1.1529868 | 10.648534 |
| legacy-easy-seed-000 | 0 | 16 | facade | 0.12978688 | 0.50033235 |
| legacy-easy-seed-000 | 0 | 16 | access | 0 | 0 |
| legacy-easy-seed-000 | 0 | 16 | support | 0.066666998 | 1.3887751 |
| legacy-easy-seed-000 | 0 | 16 | density_binary | 0.00030774067 | 0.0057679529 |
| legacy-easy-seed-000 | 0 | 16 | tv | 0.0078522144 | 0.0094840558 |
| legacy-easy-seed-000 | 0 | 16 | cantilever_boundary | 0.00034763629 | 0.00045501494 |
| legacy-easy-seed-000 | 0 | 16 | cantilever_historical | 0.00062996667 | 0.0011199394 |
| legacy-easy-seed-000 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-000 | 1 | 4 | coverage | 0.72258252 | 0.9804541 |
| legacy-easy-seed-000 | 1 | 4 | spill | 2.0744443e-08 | 7.8254463e-05 |
| legacy-easy-seed-000 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-000 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-000 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-000 | 1 | 4 | facade | 0.10603052 | 0.60320231 |
| legacy-easy-seed-000 | 1 | 4 | access | 0.80714756 | 0.41530579 |
| legacy-easy-seed-000 | 1 | 4 | support | 0.73454905 | 0.85996733 |
| legacy-easy-seed-000 | 1 | 4 | density_binary | 0.00097539608 | 0.0018190106 |
| legacy-easy-seed-000 | 1 | 4 | tv | 0.0025169048 | 0.010168288 |
| legacy-easy-seed-000 | 1 | 4 | cantilever_boundary | 0.00058228982 | 0.00071136782 |
| legacy-easy-seed-000 | 1 | 4 | cantilever_historical | 0.00058001641 | 0.0015425712 |
| legacy-easy-seed-000 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-000 | 1 | 16 | coverage | 0.05335252 | 1.9384458 |
| legacy-easy-seed-000 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-000 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-000 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-000 | 1 | 16 | sparsity | 1.2468129 | 9.708513 |
| legacy-easy-seed-000 | 1 | 16 | facade | 0.12482584 | 0.42710785 |
| legacy-easy-seed-000 | 1 | 16 | access | 0.12601936 | 5.3215208 |
| legacy-easy-seed-000 | 1 | 16 | support | 0.056370176 | 1.1542103 |
| legacy-easy-seed-000 | 1 | 16 | density_binary | 0.00026773213 | 0.0050325219 |
| legacy-easy-seed-000 | 1 | 16 | tv | 0.0077424636 | 0.0096133745 |
| legacy-easy-seed-000 | 1 | 16 | cantilever_boundary | 0.00034099742 | 0.00040976792 |
| legacy-easy-seed-000 | 1 | 16 | cantilever_historical | 0.00063254265 | 0.00073274622 |
| legacy-easy-seed-000 | 0 | 50 | legality | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | coverage | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | spill | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | ground | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | thickness | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | sparsity | 1.7480178 | 0 |
| legacy-easy-seed-000 | 0 | 50 | facade | 0.13571429 | 0 |
| legacy-easy-seed-000 | 0 | 50 | access | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | support | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | density_binary | 0 | 0 |
| legacy-easy-seed-000 | 0 | 50 | tv | 0.0069304435 | 0 |
| legacy-easy-seed-000 | 0 | 50 | cantilever_boundary | 0.00036621094 | 0 |
| legacy-easy-seed-000 | 0 | 50 | cantilever_historical | 0.00067813764 | 0 |
| legacy-easy-seed-001 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-001 | 0 | 4 | coverage | 0.70636481 | 0.99882928 |
| legacy-easy-seed-001 | 0 | 4 | spill | 6.9163697e-08 | 3.1135258e-05 |
| legacy-easy-seed-001 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-001 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-001 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-001 | 0 | 4 | facade | 0.12417677 | 0.67603463 |
| legacy-easy-seed-001 | 0 | 4 | access | 0.7537508 | 0.90097253 |
| legacy-easy-seed-001 | 0 | 4 | support | 0.71450716 | 1.1193376 |
| legacy-easy-seed-001 | 0 | 4 | density_binary | 0.00099067192 | 0.0019230965 |
| legacy-easy-seed-001 | 0 | 4 | tv | 0.0025323965 | 0.010516117 |
| legacy-easy-seed-001 | 0 | 4 | cantilever_boundary | 0.00053581386 | 0.00074862044 |
| legacy-easy-seed-001 | 0 | 4 | cantilever_historical | 0.00061456487 | 0.0017345502 |
| legacy-easy-seed-001 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-001 | 0 | 16 | coverage | 0.085826099 | 2.2126013 |
| legacy-easy-seed-001 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-001 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-001 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-001 | 0 | 16 | sparsity | 1.463428 | 14.569185 |
| legacy-easy-seed-001 | 0 | 16 | facade | 0.14278641 | 0.73428614 |
| legacy-easy-seed-001 | 0 | 16 | access | 0 | 0 |
| legacy-easy-seed-001 | 0 | 16 | support | 0.065821752 | 1.5260602 |
| legacy-easy-seed-001 | 0 | 16 | density_binary | 0.00032463047 | 0.0061745801 |
| legacy-easy-seed-001 | 0 | 16 | tv | 0.0075941077 | 0.0071401054 |
| legacy-easy-seed-001 | 0 | 16 | cantilever_boundary | 0.00037437509 | 0.00031534865 |
| legacy-easy-seed-001 | 0 | 16 | cantilever_historical | 0.00062440452 | 0.0017782373 |
| legacy-easy-seed-001 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-001 | 1 | 4 | coverage | 0.74872333 | 0.77022984 |
| legacy-easy-seed-001 | 1 | 4 | spill | 5.6424479e-08 | 3.1053953e-05 |
| legacy-easy-seed-001 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-001 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-001 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-001 | 1 | 4 | facade | 0.12833056 | 0.68478391 |
| legacy-easy-seed-001 | 1 | 4 | access | 0.76189727 | 0.59843779 |
| legacy-easy-seed-001 | 1 | 4 | support | 0.71664357 | 0.85703386 |
| legacy-easy-seed-001 | 1 | 4 | density_binary | 0.00098295684 | 0.0019328104 |
| legacy-easy-seed-001 | 1 | 4 | tv | 0.0024788477 | 0.010103365 |
| legacy-easy-seed-001 | 1 | 4 | cantilever_boundary | 0.00054708176 | 0.00069980382 |
| legacy-easy-seed-001 | 1 | 4 | cantilever_historical | 0.00066701818 | 0.0014171009 |
| legacy-easy-seed-001 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-001 | 1 | 16 | coverage | 0.10205597 | 2.8693316 |
| legacy-easy-seed-001 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-001 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-001 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-001 | 1 | 16 | sparsity | 1.5143079 | 15.351396 |
| legacy-easy-seed-001 | 1 | 16 | facade | 0.13180396 | 0.84899332 |
| legacy-easy-seed-001 | 1 | 16 | access | 0.40414482 | 3.547231 |
| legacy-easy-seed-001 | 1 | 16 | support | 0.058516037 | 1.4942321 |
| legacy-easy-seed-001 | 1 | 16 | density_binary | 0.0003035405 | 0.0059558292 |
| legacy-easy-seed-001 | 1 | 16 | tv | 0.0074986056 | 0.0079158673 |
| legacy-easy-seed-001 | 1 | 16 | cantilever_boundary | 0.00038177834 | 0.00020858333 |
| legacy-easy-seed-001 | 1 | 16 | cantilever_historical | 0.00061155291 | 0.0043343003 |
| legacy-easy-seed-002 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-002 | 0 | 4 | coverage | 0.72104073 | 0.77411501 |
| legacy-easy-seed-002 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-002 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-002 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-002 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-002 | 0 | 4 | facade | 0.038176954 | 0.30155202 |
| legacy-easy-seed-002 | 0 | 4 | access | 0.80714756 | 0.41530579 |
| legacy-easy-seed-002 | 0 | 4 | support | 0.72269112 | 0.82000336 |
| legacy-easy-seed-002 | 0 | 4 | density_binary | 0.0023500989 | 0.0037972145 |
| legacy-easy-seed-002 | 0 | 4 | tv | 0.0051715113 | 0.02016666 |
| legacy-easy-seed-002 | 0 | 4 | cantilever_boundary | 0.0016502083 | 0.0020838257 |
| legacy-easy-seed-002 | 0 | 4 | cantilever_historical | 0.0015472347 | 0.0033785112 |
| legacy-easy-seed-002 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-002 | 0 | 16 | coverage | 0.091311768 | 1.2872853 |
| legacy-easy-seed-002 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-002 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-002 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-002 | 0 | 16 | sparsity | 5.1137924 | 25.130011 |
| legacy-easy-seed-002 | 0 | 16 | facade | 0.056272015 | 0.30810808 |
| legacy-easy-seed-002 | 0 | 16 | access | 0.51688731 | 2.750664 |
| legacy-easy-seed-002 | 0 | 16 | support | 0.051687792 | 1.0917768 |
| legacy-easy-seed-002 | 0 | 16 | density_binary | 0.00057186437 | 0.011134102 |
| legacy-easy-seed-002 | 0 | 16 | tv | 0.014291792 | 0.027591806 |
| legacy-easy-seed-002 | 0 | 16 | cantilever_boundary | 0.0013419038 | 0.0015912842 |
| legacy-easy-seed-002 | 0 | 16 | cantilever_historical | 0.0021116894 | 0.0031092034 |
| legacy-easy-seed-002 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-002 | 1 | 4 | coverage | 0.73403507 | 0.71643604 |
| legacy-easy-seed-002 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-002 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-002 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-002 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-002 | 1 | 4 | facade | 0.038012743 | 0.28012461 |
| legacy-easy-seed-002 | 1 | 4 | access | 0.69618803 | 0.90838863 |
| legacy-easy-seed-002 | 1 | 4 | support | 0.72692961 | 0.78624728 |
| legacy-easy-seed-002 | 1 | 4 | density_binary | 0.0023237392 | 0.003776114 |
| legacy-easy-seed-002 | 1 | 4 | tv | 0.004982301 | 0.018928335 |
| legacy-easy-seed-002 | 1 | 4 | cantilever_boundary | 0.0016445529 | 0.0020894025 |
| legacy-easy-seed-002 | 1 | 4 | cantilever_historical | 0.0015745854 | 0.0031574354 |
| legacy-easy-seed-002 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-002 | 1 | 16 | coverage | 0.058753178 | 1.2527818 |
| legacy-easy-seed-002 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-002 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-002 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-002 | 1 | 16 | sparsity | 5.1704307 | 26.685505 |
| legacy-easy-seed-002 | 1 | 16 | facade | 0.050445944 | 0.36989735 |
| legacy-easy-seed-002 | 1 | 16 | access | 0.27468485 | 4.330435 |
| legacy-easy-seed-002 | 1 | 16 | support | 0.050481062 | 1.1591331 |
| legacy-easy-seed-002 | 1 | 16 | density_binary | 0.00056031358 | 0.011895677 |
| legacy-easy-seed-002 | 1 | 16 | tv | 0.013584108 | 0.029368081 |
| legacy-easy-seed-002 | 1 | 16 | cantilever_boundary | 0.0013086955 | 0.0016845809 |
| legacy-easy-seed-002 | 1 | 16 | cantilever_historical | 0.0020559561 | 0.0029027225 |
| legacy-easy-seed-003 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-003 | 0 | 4 | coverage | 0.69034296 | 0.96470255 |
| legacy-easy-seed-003 | 0 | 4 | spill | 7.3219375e-08 | 3.2960993e-05 |
| legacy-easy-seed-003 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-003 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-003 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-003 | 0 | 4 | facade | 0.036152765 | 0.27524786 |
| legacy-easy-seed-003 | 0 | 4 | access | 0.69569051 | 0.87447962 |
| legacy-easy-seed-003 | 0 | 4 | support | 0.72026956 | 0.80846978 |
| legacy-easy-seed-003 | 0 | 4 | density_binary | 0.0022744481 | 0.0036607621 |
| legacy-easy-seed-003 | 0 | 4 | tv | 0.0050362889 | 0.018990678 |
| legacy-easy-seed-003 | 0 | 4 | cantilever_boundary | 0.0017031659 | 0.0022700159 |
| legacy-easy-seed-003 | 0 | 4 | cantilever_historical | 0.0015009482 | 0.0031478124 |
| legacy-easy-seed-003 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-003 | 0 | 16 | coverage | 0.032880474 | 1.1334852 |
| legacy-easy-seed-003 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-003 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-003 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-003 | 0 | 16 | sparsity | 2.002269 | 13.49926 |
| legacy-easy-seed-003 | 0 | 16 | facade | 0.049576744 | 0.27990214 |
| legacy-easy-seed-003 | 0 | 16 | access | 0 | 0 |
| legacy-easy-seed-003 | 0 | 16 | support | 0.055290174 | 1.2299777 |
| legacy-easy-seed-003 | 0 | 16 | density_binary | 0.00059176481 | 0.012185998 |
| legacy-easy-seed-003 | 0 | 16 | tv | 0.013748914 | 0.02844531 |
| legacy-easy-seed-003 | 0 | 16 | cantilever_boundary | 0.0014577796 | 0.0018928708 |
| legacy-easy-seed-003 | 0 | 16 | cantilever_historical | 0.0019737845 | 0.0038797824 |
| legacy-easy-seed-003 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-003 | 1 | 4 | coverage | 0.72271276 | 0.76659274 |
| legacy-easy-seed-003 | 1 | 4 | spill | 5.9733146e-08 | 3.2874921e-05 |
| legacy-easy-seed-003 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-003 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-003 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-003 | 1 | 4 | facade | 0.040971011 | 0.31685961 |
| legacy-easy-seed-003 | 1 | 4 | access | 0.80703765 | 0.41470061 |
| legacy-easy-seed-003 | 1 | 4 | support | 0.72789973 | 0.81557699 |
| legacy-easy-seed-003 | 1 | 4 | density_binary | 0.0022537196 | 0.0036442576 |
| legacy-easy-seed-003 | 1 | 4 | tv | 0.0049633663 | 0.018835066 |
| legacy-easy-seed-003 | 1 | 4 | cantilever_boundary | 0.0016883067 | 0.0022185548 |
| legacy-easy-seed-003 | 1 | 4 | cantilever_historical | 0.0015209141 | 0.0028240579 |
| legacy-easy-seed-003 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-003 | 1 | 16 | coverage | 0.041956153 | 1.4309714 |
| legacy-easy-seed-003 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-003 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-003 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-003 | 1 | 16 | sparsity | 2.0903509 | 12.065606 |
| legacy-easy-seed-003 | 1 | 16 | facade | 0.047355339 | 0.20024213 |
| legacy-easy-seed-003 | 1 | 16 | access | 0 | 0 |
| legacy-easy-seed-003 | 1 | 16 | support | 0.046732828 | 1.0571782 |
| legacy-easy-seed-003 | 1 | 16 | density_binary | 0.00050551497 | 0.010712389 |
| legacy-easy-seed-003 | 1 | 16 | tv | 0.013389269 | 0.027274314 |
| legacy-easy-seed-003 | 1 | 16 | cantilever_boundary | 0.0014950577 | 0.0014916843 |
| legacy-easy-seed-003 | 1 | 16 | cantilever_historical | 0.002025241 | 0.0027642862 |
| legacy-easy-seed-004 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-004 | 0 | 4 | coverage | 0.7637468 | 0.7502523 |
| legacy-easy-seed-004 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-004 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-004 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-004 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-004 | 0 | 4 | facade | 0.076838136 | 0.3323859 |
| legacy-easy-seed-004 | 0 | 4 | access | 0.86006969 | 1.8626132 |
| legacy-easy-seed-004 | 0 | 4 | support | 0.69491923 | 0.8346062 |
| legacy-easy-seed-004 | 0 | 4 | density_binary | 0.0013462305 | 0.0023389901 |
| legacy-easy-seed-004 | 0 | 4 | tv | 0.003804445 | 0.013759338 |
| legacy-easy-seed-004 | 0 | 4 | cantilever_boundary | 0.00085870246 | 0.0011582568 |
| legacy-easy-seed-004 | 0 | 4 | cantilever_historical | 0.00094230345 | 0.0016633132 |
| legacy-easy-seed-004 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-004 | 0 | 16 | coverage | 0.27786249 | 1.713391 |
| legacy-easy-seed-004 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-004 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-004 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-004 | 0 | 16 | sparsity | 1.2028694 | 11.676171 |
| legacy-easy-seed-004 | 0 | 16 | facade | 0.08057566 | 0.41902378 |
| legacy-easy-seed-004 | 0 | 16 | access | 1 | 0 |
| legacy-easy-seed-004 | 0 | 16 | support | 0.074694261 | 1.2919624 |
| legacy-easy-seed-004 | 0 | 16 | density_binary | 0.00049690623 | 0.0065728818 |
| legacy-easy-seed-004 | 0 | 16 | tv | 0.01147751 | 0.0096650877 |
| legacy-easy-seed-004 | 0 | 16 | cantilever_boundary | 0.00090958609 | 0.0013132396 |
| legacy-easy-seed-004 | 0 | 16 | cantilever_historical | 0.0011542211 | 0.0036382483 |
| legacy-easy-seed-004 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-004 | 1 | 4 | coverage | 0.74450976 | 0.81768731 |
| legacy-easy-seed-004 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-004 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-004 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-004 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-004 | 1 | 4 | facade | 0.07320492 | 0.31193938 |
| legacy-easy-seed-004 | 1 | 4 | access | 0.86295497 | 2.7882463 |
| legacy-easy-seed-004 | 1 | 4 | support | 0.69059372 | 0.87208409 |
| legacy-easy-seed-004 | 1 | 4 | density_binary | 0.0013619402 | 0.0023578295 |
| legacy-easy-seed-004 | 1 | 4 | tv | 0.0037374375 | 0.013089765 |
| legacy-easy-seed-004 | 1 | 4 | cantilever_boundary | 0.00089839683 | 0.0013055364 |
| legacy-easy-seed-004 | 1 | 4 | cantilever_historical | 0.0010130161 | 0.0013790477 |
| legacy-easy-seed-004 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-004 | 1 | 16 | coverage | 0.18290488 | 0.97766305 |
| legacy-easy-seed-004 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-004 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-004 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-004 | 1 | 16 | sparsity | 1.3841513 | 11.578458 |
| legacy-easy-seed-004 | 1 | 16 | facade | 0.068216622 | 0.66592655 |
| legacy-easy-seed-004 | 1 | 16 | access | 1 | 0 |
| legacy-easy-seed-004 | 1 | 16 | support | 0.051271599 | 1.0639814 |
| legacy-easy-seed-004 | 1 | 16 | density_binary | 0.00038560794 | 0.0056647986 |
| legacy-easy-seed-004 | 1 | 16 | tv | 0.010828983 | 0.0088573715 |
| legacy-easy-seed-004 | 1 | 16 | cantilever_boundary | 0.00091861602 | 0.00067971605 |
| legacy-easy-seed-004 | 1 | 16 | cantilever_historical | 0.0011332531 | 0.0046686281 |
| legacy-easy-seed-005 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-005 | 0 | 4 | coverage | 0.72807384 | 0.78124309 |
| legacy-easy-seed-005 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-005 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-005 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-005 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-005 | 0 | 4 | facade | 0.032021627 | 0.29457839 |
| legacy-easy-seed-005 | 0 | 4 | access | 0.75201601 | 0.88844817 |
| legacy-easy-seed-005 | 0 | 4 | support | 0.70133412 | 0.76557196 |
| legacy-easy-seed-005 | 0 | 4 | density_binary | 0.0016417052 | 0.0027500519 |
| legacy-easy-seed-005 | 0 | 4 | tv | 0.0045884135 | 0.016408978 |
| legacy-easy-seed-005 | 0 | 4 | cantilever_boundary | 0.0012050912 | 0.0019392879 |
| legacy-easy-seed-005 | 0 | 4 | cantilever_historical | 0.0013577652 | 0.0009962194 |
| legacy-easy-seed-005 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-005 | 0 | 16 | coverage | 0.096788846 | 1.9684777 |
| legacy-easy-seed-005 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-005 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-005 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-005 | 0 | 16 | sparsity | 2.0681224 | 17.914798 |
| legacy-easy-seed-005 | 0 | 16 | facade | 0.028704569 | 0.59653487 |
| legacy-easy-seed-005 | 0 | 16 | access | 0.11947656 | 5.4282895 |
| legacy-easy-seed-005 | 0 | 16 | support | 0.066291325 | 1.3956896 |
| legacy-easy-seed-005 | 0 | 16 | density_binary | 0.00055952283 | 0.0089775067 |
| legacy-easy-seed-005 | 0 | 16 | tv | 0.013384335 | 0.010264195 |
| legacy-easy-seed-005 | 0 | 16 | cantilever_boundary | 0.0017003376 | 0.0024892557 |
| legacy-easy-seed-005 | 0 | 16 | cantilever_historical | 0.0022488644 | 0.0035209276 |
| legacy-easy-seed-005 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-005 | 1 | 4 | coverage | 0.73792201 | 0.76666314 |
| legacy-easy-seed-005 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-005 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-005 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-005 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-005 | 1 | 4 | facade | 0.035084814 | 0.31523742 |
| legacy-easy-seed-005 | 1 | 4 | access | 0.75593209 | 0.57644291 |
| legacy-easy-seed-005 | 1 | 4 | support | 0.7045626 | 0.84420681 |
| legacy-easy-seed-005 | 1 | 4 | density_binary | 0.0016455378 | 0.0028001034 |
| legacy-easy-seed-005 | 1 | 4 | tv | 0.0044930354 | 0.015812955 |
| legacy-easy-seed-005 | 1 | 4 | cantilever_boundary | 0.0012066448 | 0.0019487745 |
| legacy-easy-seed-005 | 1 | 4 | cantilever_historical | 0.0013813038 | 0.00090999303 |
| legacy-easy-seed-005 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-005 | 1 | 16 | coverage | 0.12538627 | 1.0294435 |
| legacy-easy-seed-005 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-005 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-005 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-005 | 1 | 16 | sparsity | 2.2576818 | 14.879372 |
| legacy-easy-seed-005 | 1 | 16 | facade | 0.038459256 | 0.46090075 |
| legacy-easy-seed-005 | 1 | 16 | access | 0 | 0 |
| legacy-easy-seed-005 | 1 | 16 | support | 0.049875926 | 1.0079656 |
| legacy-easy-seed-005 | 1 | 16 | density_binary | 0.00044304578 | 0.006460229 |
| legacy-easy-seed-005 | 1 | 16 | tv | 0.013103876 | 0.011097123 |
| legacy-easy-seed-005 | 1 | 16 | cantilever_boundary | 0.001677351 | 0.0021474441 |
| legacy-easy-seed-005 | 1 | 16 | cantilever_historical | 0.0022194334 | 0.004262336 |
| legacy-easy-seed-006 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-006 | 0 | 4 | coverage | 0.72789097 | 0.7594358 |
| legacy-easy-seed-006 | 0 | 4 | spill | 5.5451984e-08 | 4.9951828e-05 |
| legacy-easy-seed-006 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-006 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-006 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-006 | 0 | 4 | facade | 0 | 0 |
| legacy-easy-seed-006 | 0 | 4 | access | 0.72184741 | 0.9512256 |
| legacy-easy-seed-006 | 0 | 4 | support | 0.72360849 | 0.86696362 |
| legacy-easy-seed-006 | 0 | 4 | density_binary | 0.0023101973 | 0.0035040036 |
| legacy-easy-seed-006 | 0 | 4 | tv | 0.0059574721 | 0.021248353 |
| legacy-easy-seed-006 | 0 | 4 | cantilever_boundary | 0.0018416937 | 0.0024185044 |
| legacy-easy-seed-006 | 0 | 4 | cantilever_historical | 0.0015855158 | 0.0031054311 |
| legacy-easy-seed-006 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-006 | 0 | 16 | coverage | 0.052894648 | 1.2577604 |
| legacy-easy-seed-006 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-006 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-006 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-006 | 0 | 16 | sparsity | 0.71037424 | 5.0376625 |
| legacy-easy-seed-006 | 0 | 16 | facade | 0.0047225654 | 0.08541246 |
| legacy-easy-seed-006 | 0 | 16 | access | 0 | 0 |
| legacy-easy-seed-006 | 0 | 16 | support | 0.04750305 | 0.94066633 |
| legacy-easy-seed-006 | 0 | 16 | density_binary | 0.00051761279 | 0.0095812072 |
| legacy-easy-seed-006 | 0 | 16 | tv | 0.016182978 | 0.024710462 |
| legacy-easy-seed-006 | 0 | 16 | cantilever_boundary | 0.001576985 | 0.0021122109 |
| legacy-easy-seed-006 | 0 | 16 | cantilever_historical | 0.0019594745 | 0.003273501 |
| legacy-easy-seed-006 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-006 | 1 | 4 | coverage | 0.71893346 | 0.78819904 |
| legacy-easy-seed-006 | 1 | 4 | spill | 2.8991403e-08 | 1.3002613e-05 |
| legacy-easy-seed-006 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-006 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-006 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-006 | 1 | 4 | facade | 0 | 0 |
| legacy-easy-seed-006 | 1 | 4 | access | 0.79273582 | 0.4734988 |
| legacy-easy-seed-006 | 1 | 4 | support | 0.72736454 | 0.73848209 |
| legacy-easy-seed-006 | 1 | 4 | density_binary | 0.0022877788 | 0.0034952156 |
| legacy-easy-seed-006 | 1 | 4 | tv | 0.005634719 | 0.019491601 |
| legacy-easy-seed-006 | 1 | 4 | cantilever_boundary | 0.0018089269 | 0.0022660583 |
| legacy-easy-seed-006 | 1 | 4 | cantilever_historical | 0.001500489 | 0.0034328717 |
| legacy-easy-seed-006 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-006 | 1 | 16 | coverage | 0.042599864 | 1.0269618 |
| legacy-easy-seed-006 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-006 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-006 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-006 | 1 | 16 | sparsity | 0.70812809 | 4.6715149 |
| legacy-easy-seed-006 | 1 | 16 | facade | 0 | 0 |
| legacy-easy-seed-006 | 1 | 16 | access | 0 | 0 |
| legacy-easy-seed-006 | 1 | 16 | support | 0.048281729 | 0.84574022 |
| legacy-easy-seed-006 | 1 | 16 | density_binary | 0.00052357517 | 0.0084450654 |
| legacy-easy-seed-006 | 1 | 16 | tv | 0.016040798 | 0.025684993 |
| legacy-easy-seed-006 | 1 | 16 | cantilever_boundary | 0.0015872363 | 0.0021771696 |
| legacy-easy-seed-006 | 1 | 16 | cantilever_historical | 0.0019590645 | 0.0030703299 |
| legacy-easy-seed-007 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-007 | 0 | 4 | coverage | 0.69985437 | 0.9805957 |
| legacy-easy-seed-007 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-007 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-007 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-007 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-007 | 0 | 4 | facade | 0.18282378 | 0.54684465 |
| legacy-easy-seed-007 | 0 | 4 | access | 0.72141433 | 0.961224 |
| legacy-easy-seed-007 | 0 | 4 | support | 0.71270347 | 0.85273182 |
| legacy-easy-seed-007 | 0 | 4 | density_binary | 0.0011002783 | 0.0017921004 |
| legacy-easy-seed-007 | 0 | 4 | tv | 0.0031066509 | 0.010818598 |
| legacy-easy-seed-007 | 0 | 4 | cantilever_boundary | 0.0005985403 | 0.00066366603 |
| legacy-easy-seed-007 | 0 | 4 | cantilever_historical | 0.00062769529 | 0.0019221854 |
| legacy-easy-seed-007 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-007 | 0 | 16 | coverage | 0.046540324 | 0.7286105 |
| legacy-easy-seed-007 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-007 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-007 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-007 | 0 | 16 | sparsity | 1.4867125 | 7.8819477 |
| legacy-easy-seed-007 | 0 | 16 | facade | 0.18343121 | 0.19216291 |
| legacy-easy-seed-007 | 0 | 16 | access | 0 | 0 |
| legacy-easy-seed-007 | 0 | 16 | support | 0.046351291 | 0.8272376 |
| legacy-easy-seed-007 | 0 | 16 | density_binary | 0.00023882718 | 0.0039755693 |
| legacy-easy-seed-007 | 0 | 16 | tv | 0.0087347543 | 0.011251087 |
| legacy-easy-seed-007 | 0 | 16 | cantilever_boundary | 0.00029122102 | 0.0004908319 |
| legacy-easy-seed-007 | 0 | 16 | cantilever_historical | 0.00059676718 | 0.00096918423 |
| legacy-easy-seed-007 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-007 | 1 | 4 | coverage | 0.71558625 | 0.88909063 |
| legacy-easy-seed-007 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-007 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-007 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-007 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-007 | 1 | 4 | facade | 0.17352322 | 0.52272331 |
| legacy-easy-seed-007 | 1 | 4 | access | 0.69665861 | 0.87549908 |
| legacy-easy-seed-007 | 1 | 4 | support | 0.71853822 | 0.79696608 |
| legacy-easy-seed-007 | 1 | 4 | density_binary | 0.0010809202 | 0.0017837051 |
| legacy-easy-seed-007 | 1 | 4 | tv | 0.0029567925 | 0.010225386 |
| legacy-easy-seed-007 | 1 | 4 | cantilever_boundary | 0.00060103217 | 0.00071826808 |
| legacy-easy-seed-007 | 1 | 4 | cantilever_historical | 0.00064922782 | 0.0018209768 |
| legacy-easy-seed-007 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-007 | 1 | 16 | coverage | 0.0099628679 | 0.42826692 |
| legacy-easy-seed-007 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-007 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-007 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-007 | 1 | 16 | sparsity | 1.4249519 | 8.355311 |
| legacy-easy-seed-007 | 1 | 16 | facade | 0.18740103 | 0.2078848 |
| legacy-easy-seed-007 | 1 | 16 | access | 0 | 0 |
| legacy-easy-seed-007 | 1 | 16 | support | 0.056286845 | 0.95438584 |
| legacy-easy-seed-007 | 1 | 16 | density_binary | 0.00027267757 | 0.0042830909 |
| legacy-easy-seed-007 | 1 | 16 | tv | 0.0086477399 | 0.0049183842 |
| legacy-easy-seed-007 | 1 | 16 | cantilever_boundary | 0.00028537621 | 0.0002216982 |
| legacy-easy-seed-007 | 1 | 16 | cantilever_historical | 0.00060686615 | 0.00072676661 |
| legacy-easy-seed-008 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-008 | 0 | 4 | coverage | 0.75358188 | 1.0518036 |
| legacy-easy-seed-008 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-008 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-008 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-008 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-008 | 0 | 4 | facade | 0.21403885 | 0.56234249 |
| legacy-easy-seed-008 | 0 | 4 | access | 0.84380031 | 1.8000611 |
| legacy-easy-seed-008 | 0 | 4 | support | 0.66127139 | 0.9595061 |
| legacy-easy-seed-008 | 0 | 4 | density_binary | 0.00061686774 | 0.0011157055 |
| legacy-easy-seed-008 | 0 | 4 | tv | 0.0019346582 | 0.0073081133 |
| legacy-easy-seed-008 | 0 | 4 | cantilever_boundary | 0.00026586454 | 0.00028708712 |
| legacy-easy-seed-008 | 0 | 4 | cantilever_historical | 0.00034855038 | 0.0014871298 |
| legacy-easy-seed-008 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-008 | 0 | 16 | coverage | 0.30323842 | 2.4130026 |
| legacy-easy-seed-008 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-008 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-008 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-008 | 0 | 16 | sparsity | 0.3697072 | 5.7105542 |
| legacy-easy-seed-008 | 0 | 16 | facade | 0.21041197 | 0.48075341 |
| legacy-easy-seed-008 | 0 | 16 | access | 0.88618565 | 13.185886 |
| legacy-easy-seed-008 | 0 | 16 | support | 0.068698809 | 1.5180137 |
| legacy-easy-seed-008 | 0 | 16 | density_binary | 0.00021257882 | 0.0025813946 |
| legacy-easy-seed-008 | 0 | 16 | tv | 0.0056008976 | 0.0055161702 |
| legacy-easy-seed-008 | 0 | 16 | cantilever_boundary | 0 | 0 |
| legacy-easy-seed-008 | 0 | 16 | cantilever_historical | 2.5160331e-05 | 9.6302716e-05 |
| legacy-easy-seed-008 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-008 | 1 | 4 | coverage | 0.74751323 | 0.92650264 |
| legacy-easy-seed-008 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-008 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-008 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-008 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-008 | 1 | 4 | facade | 0.19967705 | 0.53448866 |
| legacy-easy-seed-008 | 1 | 4 | access | 0.84541982 | 1.0716446 |
| legacy-easy-seed-008 | 1 | 4 | support | 0.66385621 | 0.86627705 |
| legacy-easy-seed-008 | 1 | 4 | density_binary | 0.00061907107 | 0.0010756085 |
| legacy-easy-seed-008 | 1 | 4 | tv | 0.0018465043 | 0.0064468437 |
| legacy-easy-seed-008 | 1 | 4 | cantilever_boundary | 0.00026173127 | 0.00027756773 |
| legacy-easy-seed-008 | 1 | 4 | cantilever_historical | 0.00035540678 | 0.0013781803 |
| legacy-easy-seed-008 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-008 | 1 | 16 | coverage | 0.26545465 | 2.0659218 |
| legacy-easy-seed-008 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-008 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-008 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-008 | 1 | 16 | sparsity | 0.3574442 | 5.2934891 |
| legacy-easy-seed-008 | 1 | 16 | facade | 0.20428681 | 0.5907179 |
| legacy-easy-seed-008 | 1 | 16 | access | 0.88028044 | 16.781845 |
| legacy-easy-seed-008 | 1 | 16 | support | 0.06724108 | 0.94622788 |
| legacy-easy-seed-008 | 1 | 16 | density_binary | 0.00021946161 | 0.00189579 |
| legacy-easy-seed-008 | 1 | 16 | tv | 0.0056953104 | 0.0052511189 |
| legacy-easy-seed-008 | 1 | 16 | cantilever_boundary | 0 | 0 |
| legacy-easy-seed-008 | 1 | 16 | cantilever_historical | 1.5437394e-05 | 0.00013193195 |
| legacy-easy-seed-009 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-009 | 0 | 4 | coverage | 0.70033991 | 0.87959148 |
| legacy-easy-seed-009 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-009 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-009 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-009 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-009 | 0 | 4 | facade | 0.03051585 | 0.22672863 |
| legacy-easy-seed-009 | 0 | 4 | access | 0.69684756 | 0.90615037 |
| legacy-easy-seed-009 | 0 | 4 | support | 0.72090882 | 0.80134547 |
| legacy-easy-seed-009 | 0 | 4 | density_binary | 0.0024832613 | 0.0038503463 |
| legacy-easy-seed-009 | 0 | 4 | tv | 0.0061627342 | 0.022707259 |
| legacy-easy-seed-009 | 0 | 4 | cantilever_boundary | 0.0018145444 | 0.0021625734 |
| legacy-easy-seed-009 | 0 | 4 | cantilever_historical | 0.0016148281 | 0.0036801682 |
| legacy-easy-seed-009 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-009 | 0 | 16 | coverage | 0.032105975 | 0.96832096 |
| legacy-easy-seed-009 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-009 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-009 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-009 | 0 | 16 | sparsity | 3.5115933 | 15.39328 |
| legacy-easy-seed-009 | 0 | 16 | facade | 0.038358957 | 0.092032263 |
| legacy-easy-seed-009 | 0 | 16 | access | 0 | 0 |
| legacy-easy-seed-009 | 0 | 16 | support | 0.047448013 | 0.85654526 |
| legacy-easy-seed-009 | 0 | 16 | density_binary | 0.00055738888 | 0.0093779806 |
| legacy-easy-seed-009 | 0 | 16 | tv | 0.016610431 | 0.027499816 |
| legacy-easy-seed-009 | 0 | 16 | cantilever_boundary | 0.0013209524 | 0.0023068373 |
| legacy-easy-seed-009 | 0 | 16 | cantilever_historical | 0.001971982 | 0.0033841586 |
| legacy-easy-seed-009 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-009 | 1 | 4 | coverage | 0.70659143 | 0.85009362 |
| legacy-easy-seed-009 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-009 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-009 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-009 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-009 | 1 | 4 | facade | 0.029275909 | 0.22438323 |
| legacy-easy-seed-009 | 1 | 4 | access | 0.70173335 | 0.85306005 |
| legacy-easy-seed-009 | 1 | 4 | support | 0.7220974 | 0.78898553 |
| legacy-easy-seed-009 | 1 | 4 | density_binary | 0.0024765367 | 0.0039455637 |
| legacy-easy-seed-009 | 1 | 4 | tv | 0.0058719544 | 0.021342552 |
| legacy-easy-seed-009 | 1 | 4 | cantilever_boundary | 0.0018107845 | 0.00218022 |
| legacy-easy-seed-009 | 1 | 4 | cantilever_historical | 0.0016279044 | 0.0035326544 |
| legacy-easy-seed-009 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-009 | 1 | 16 | coverage | 0.047950953 | 1.3381297 |
| legacy-easy-seed-009 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-009 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-009 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-009 | 1 | 16 | sparsity | 3.5563102 | 15.246364 |
| legacy-easy-seed-009 | 1 | 16 | facade | 0.039945513 | 0.068055409 |
| legacy-easy-seed-009 | 1 | 16 | access | 0 | 0 |
| legacy-easy-seed-009 | 1 | 16 | support | 0.044364367 | 0.87643112 |
| legacy-easy-seed-009 | 1 | 16 | density_binary | 0.0005209287 | 0.0095696595 |
| legacy-easy-seed-009 | 1 | 16 | tv | 0.016607059 | 0.029077739 |
| legacy-easy-seed-009 | 1 | 16 | cantilever_boundary | 0.0013749638 | 0.0012229938 |
| legacy-easy-seed-009 | 1 | 16 | cantilever_historical | 0.0020377161 | 0.0019186291 |
| legacy-easy-seed-010 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-010 | 0 | 4 | coverage | 0.725389 | 0.74271701 |
| legacy-easy-seed-010 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-010 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-010 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-010 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-010 | 0 | 4 | facade | 0.004338637 | 0.18739681 |
| legacy-easy-seed-010 | 0 | 4 | access | 0.72141844 | 0.96335416 |
| legacy-easy-seed-010 | 0 | 4 | support | 0.71756285 | 0.81119077 |
| legacy-easy-seed-010 | 0 | 4 | density_binary | 0.0024397806 | 0.0037394848 |
| legacy-easy-seed-010 | 0 | 4 | tv | 0.0061721108 | 0.022138543 |
| legacy-easy-seed-010 | 0 | 4 | cantilever_boundary | 0.0019557998 | 0.0024907989 |
| legacy-easy-seed-010 | 0 | 4 | cantilever_historical | 0.0015644514 | 0.0035389358 |
| legacy-easy-seed-010 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-010 | 0 | 16 | coverage | 0.078265585 | 1.2202289 |
| legacy-easy-seed-010 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-010 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-010 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-010 | 0 | 16 | sparsity | 1.5915376 | 8.8364604 |
| legacy-easy-seed-010 | 0 | 16 | facade | 0.012364283 | 0.086263315 |
| legacy-easy-seed-010 | 0 | 16 | access | 0 | 0 |
| legacy-easy-seed-010 | 0 | 16 | support | 0.046586104 | 0.9191872 |
| legacy-easy-seed-010 | 0 | 16 | density_binary | 0.00053675997 | 0.0099028173 |
| legacy-easy-seed-010 | 0 | 16 | tv | 0.016595952 | 0.02895673 |
| legacy-easy-seed-010 | 0 | 16 | cantilever_boundary | 0.0016160825 | 0.0019023033 |
| legacy-easy-seed-010 | 0 | 16 | cantilever_historical | 0.0019848265 | 0.0026590695 |
| legacy-easy-seed-010 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-010 | 1 | 4 | coverage | 0.72115654 | 0.77624912 |
| legacy-easy-seed-010 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-010 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-010 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-010 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-010 | 1 | 4 | facade | 0.0012910813 | 0.16670276 |
| legacy-easy-seed-010 | 1 | 4 | access | 0.72174561 | 0.9510374 |
| legacy-easy-seed-010 | 1 | 4 | support | 0.72747487 | 0.78030192 |
| legacy-easy-seed-010 | 1 | 4 | density_binary | 0.0023982618 | 0.0036860079 |
| legacy-easy-seed-010 | 1 | 4 | tv | 0.0058799861 | 0.020960029 |
| legacy-easy-seed-010 | 1 | 4 | cantilever_boundary | 0.0019279479 | 0.0024215122 |
| legacy-easy-seed-010 | 1 | 4 | cantilever_historical | 0.0016145735 | 0.0034512954 |
| legacy-easy-seed-010 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-010 | 1 | 16 | coverage | 0.073308669 | 1.3754826 |
| legacy-easy-seed-010 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-010 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-010 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-010 | 1 | 16 | sparsity | 1.546212 | 9.0775957 |
| legacy-easy-seed-010 | 1 | 16 | facade | 0.011168092 | 0.10582522 |
| legacy-easy-seed-010 | 1 | 16 | access | 0 | 0 |
| legacy-easy-seed-010 | 1 | 16 | support | 0.050809 | 0.98072796 |
| legacy-easy-seed-010 | 1 | 16 | density_binary | 0.00057717896 | 0.010192635 |
| legacy-easy-seed-010 | 1 | 16 | tv | 0.016669005 | 0.026657045 |
| legacy-easy-seed-010 | 1 | 16 | cantilever_boundary | 0.00160698 | 0.0022460628 |
| legacy-easy-seed-010 | 1 | 16 | cantilever_historical | 0.0020030495 | 0.002935623 |
| legacy-easy-seed-011 | 0 | 4 | legality | 0 | 0 |
| legacy-easy-seed-011 | 0 | 4 | coverage | 0.708758 | 0.84639734 |
| legacy-easy-seed-011 | 0 | 4 | spill | 0 | 0 |
| legacy-easy-seed-011 | 0 | 4 | ground | 0 | 0 |
| legacy-easy-seed-011 | 0 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-011 | 0 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-011 | 0 | 4 | facade | 0.010955542 | 0.27934381 |
| legacy-easy-seed-011 | 0 | 4 | access | 0.75620234 | 0.57318272 |
| legacy-easy-seed-011 | 0 | 4 | support | 0.72671086 | 0.83443906 |
| legacy-easy-seed-011 | 0 | 4 | density_binary | 0.0014675006 | 0.0024278839 |
| legacy-easy-seed-011 | 0 | 4 | tv | 0.0040405476 | 0.014751722 |
| legacy-easy-seed-011 | 0 | 4 | cantilever_boundary | 0.0011998309 | 0.0018283732 |
| legacy-easy-seed-011 | 0 | 4 | cantilever_historical | 0.0010536794 | 0.0015377758 |
| legacy-easy-seed-011 | 0 | 16 | legality | 0 | 0 |
| legacy-easy-seed-011 | 0 | 16 | coverage | 0.045896653 | 1.5125211 |
| legacy-easy-seed-011 | 0 | 16 | spill | 0 | 0 |
| legacy-easy-seed-011 | 0 | 16 | ground | 0 | 0 |
| legacy-easy-seed-011 | 0 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-011 | 0 | 16 | sparsity | 0.57649827 | 4.590848 |
| legacy-easy-seed-011 | 0 | 16 | facade | 0.029131994 | 0.29763102 |
| legacy-easy-seed-011 | 0 | 16 | access | 0.059821725 | 4.4511709 |
| legacy-easy-seed-011 | 0 | 16 | support | 0.045262739 | 0.97180058 |
| legacy-easy-seed-011 | 0 | 16 | density_binary | 0.00031524393 | 0.0063409981 |
| legacy-easy-seed-011 | 0 | 16 | tv | 0.011614423 | 0.012764758 |
| legacy-easy-seed-011 | 0 | 16 | cantilever_boundary | 0.0012643198 | 0.0017169835 |
| legacy-easy-seed-011 | 0 | 16 | cantilever_historical | 0.0015219626 | 0.0023672324 |
| legacy-easy-seed-011 | 1 | 4 | legality | 0 | 0 |
| legacy-easy-seed-011 | 1 | 4 | coverage | 0.7217648 | 0.79004501 |
| legacy-easy-seed-011 | 1 | 4 | spill | 0 | 0 |
| legacy-easy-seed-011 | 1 | 4 | ground | 0 | 0 |
| legacy-easy-seed-011 | 1 | 4 | thickness | 0 | 0 |
| legacy-easy-seed-011 | 1 | 4 | sparsity | 0 | 0 |
| legacy-easy-seed-011 | 1 | 4 | facade | 0.016108677 | 0.30816669 |
| legacy-easy-seed-011 | 1 | 4 | access | 0.7467199 | 0.91059687 |
| legacy-easy-seed-011 | 1 | 4 | support | 0.72884721 | 0.80208527 |
| legacy-easy-seed-011 | 1 | 4 | density_binary | 0.0014433966 | 0.0024096048 |
| legacy-easy-seed-011 | 1 | 4 | tv | 0.0038556382 | 0.013809899 |
| legacy-easy-seed-011 | 1 | 4 | cantilever_boundary | 0.0011785186 | 0.0017341099 |
| legacy-easy-seed-011 | 1 | 4 | cantilever_historical | 0.001095045 | 0.0015972753 |
| legacy-easy-seed-011 | 1 | 16 | legality | 0 | 0 |
| legacy-easy-seed-011 | 1 | 16 | coverage | 0.087127879 | 1.7808135 |
| legacy-easy-seed-011 | 1 | 16 | spill | 0 | 0 |
| legacy-easy-seed-011 | 1 | 16 | ground | 0 | 0 |
| legacy-easy-seed-011 | 1 | 16 | thickness | 0 | 0 |
| legacy-easy-seed-011 | 1 | 16 | sparsity | 0.5292179 | 4.9997537 |
| legacy-easy-seed-011 | 1 | 16 | facade | 0.029442534 | 0.36769036 |
| legacy-easy-seed-011 | 1 | 16 | access | 0 | 0 |
| legacy-easy-seed-011 | 1 | 16 | support | 0.05656217 | 1.1444784 |
| legacy-easy-seed-011 | 1 | 16 | density_binary | 0.0003867026 | 0.0070488849 |
| legacy-easy-seed-011 | 1 | 16 | tv | 0.0116941 | 0.0095313264 |
| legacy-easy-seed-011 | 1 | 16 | cantilever_boundary | 0.0012435719 | 0.0018545974 |
| legacy-easy-seed-011 | 1 | 16 | cantilever_historical | 0.0015337521 | 0.0023452719 |
| ref-01-ground-pair | 0 | 4 | legality | 0 | 0 |
| ref-01-ground-pair | 0 | 4 | coverage | 0.87074733 | 0.8154003 |
| ref-01-ground-pair | 0 | 4 | spill | 0.00026857562 | 0.014329291 |
| ref-01-ground-pair | 0 | 4 | ground | 0 | 0 |
| ref-01-ground-pair | 0 | 4 | thickness | 0 | 0 |
| ref-01-ground-pair | 0 | 4 | sparsity | 0 | 0 |
| ref-01-ground-pair | 0 | 4 | facade | 0.041256741 | 5.3287521 |
| ref-01-ground-pair | 0 | 4 | access | 1 | 0 |
| ref-01-ground-pair | 0 | 4 | support | 0.48888403 | 2.8030783 |
| ref-01-ground-pair | 0 | 4 | density_binary | 0.0012516414 | 0.013893696 |
| ref-01-ground-pair | 0 | 4 | tv | 0.0034607381 | 0.043834909 |
| ref-01-ground-pair | 0 | 4 | cantilever_boundary | 0.00024589963 | 0.00041947234 |
| ref-01-ground-pair | 0 | 4 | cantilever_historical | 0.00089745416 | 0.011839063 |
| ref-01-ground-pair | 0 | 16 | legality | 0 | 0 |
| ref-01-ground-pair | 0 | 16 | coverage | 0.77596784 | 2.0984913 |
| ref-01-ground-pair | 0 | 16 | spill | 0.00216659 | 0.16610163 |
| ref-01-ground-pair | 0 | 16 | ground | 0 | 0 |
| ref-01-ground-pair | 0 | 16 | thickness | 0 | 0 |
| ref-01-ground-pair | 0 | 16 | sparsity | 0.31241369 | 70.701957 |
| ref-01-ground-pair | 0 | 16 | facade | 0.21952274 | 15.905654 |
| ref-01-ground-pair | 0 | 16 | access | 1 | 0 |
| ref-01-ground-pair | 0 | 16 | support | 0.3100177 | 7.1252101 |
| ref-01-ground-pair | 0 | 16 | density_binary | 0.0017430602 | 0.085352596 |
| ref-01-ground-pair | 0 | 16 | tv | 0.011225676 | 0.43699379 |
| ref-01-ground-pair | 0 | 16 | cantilever_boundary | 0.00034991285 | 0.00046956552 |
| ref-01-ground-pair | 0 | 16 | cantilever_historical | 0.0018010513 | 0.0072305679 |
| ref-01-ground-pair | 1 | 4 | legality | 0 | 0 |
| ref-01-ground-pair | 1 | 4 | coverage | 0.8775636 | 0.75809648 |
| ref-01-ground-pair | 1 | 4 | spill | 0.00025160686 | 0.013246187 |
| ref-01-ground-pair | 1 | 4 | ground | 0 | 0 |
| ref-01-ground-pair | 1 | 4 | thickness | 0 | 0 |
| ref-01-ground-pair | 1 | 4 | sparsity | 0 | 0 |
| ref-01-ground-pair | 1 | 4 | facade | 0.042920843 | 5.1657117 |
| ref-01-ground-pair | 1 | 4 | access | 1 | 0 |
| ref-01-ground-pair | 1 | 4 | support | 0.48443452 | 2.7613462 |
| ref-01-ground-pair | 1 | 4 | density_binary | 0.0012192369 | 0.013178745 |
| ref-01-ground-pair | 1 | 4 | tv | 0.0033367984 | 0.041201635 |
| ref-01-ground-pair | 1 | 4 | cantilever_boundary | 0.0002165106 | 0.00031249114 |
| ref-01-ground-pair | 1 | 4 | cantilever_historical | 0.00087984686 | 0.011101892 |
| ref-01-ground-pair | 1 | 16 | legality | 0 | 0 |
| ref-01-ground-pair | 1 | 16 | coverage | 0.76890171 | 2.5585922 |
| ref-01-ground-pair | 1 | 16 | spill | 0.0021346142 | 0.15947175 |
| ref-01-ground-pair | 1 | 16 | ground | 0 | 0 |
| ref-01-ground-pair | 1 | 16 | thickness | 0 | 0 |
| ref-01-ground-pair | 1 | 16 | sparsity | 0.27058646 | 63.65091 |
| ref-01-ground-pair | 1 | 16 | facade | 0.22123376 | 15.465505 |
| ref-01-ground-pair | 1 | 16 | access | 1 | 0 |
| ref-01-ground-pair | 1 | 16 | support | 0.31717274 | 6.4131157 |
| ref-01-ground-pair | 1 | 16 | density_binary | 0.0017381511 | 0.081846084 |
| ref-01-ground-pair | 1 | 16 | tv | 0.011265836 | 0.42775077 |
| ref-01-ground-pair | 1 | 16 | cantilever_boundary | 0.00032153499 | 0.00060428178 |
| ref-01-ground-pair | 1 | 16 | cantilever_historical | 0.0017386883 | 0.018460952 |
| ref-01-ground-pair | 0 | 50 | legality | 0 | 0 |
| ref-01-ground-pair | 0 | 50 | coverage | 0.6910001 | 0.098087229 |
| ref-01-ground-pair | 0 | 50 | spill | 0.012273902 | 0 |
| ref-01-ground-pair | 0 | 50 | ground | 0 | 0 |
| ref-01-ground-pair | 0 | 50 | thickness | 0 | 0 |
| ref-01-ground-pair | 0 | 50 | sparsity | 20.870443 | 581.00697 |
| ref-01-ground-pair | 0 | 50 | facade | 0.53260398 | 6.4252252 |
| ref-01-ground-pair | 0 | 50 | access | 1 | 0 |
| ref-01-ground-pair | 0 | 50 | support | 0.00053693476 | 0.54136664 |
| ref-01-ground-pair | 0 | 50 | density_binary | 1.3614957e-05 | 0.013573955 |
| ref-01-ground-pair | 0 | 50 | tv | 0.031473979 | 0.30160541 |
| ref-01-ground-pair | 0 | 50 | cantilever_boundary | 0.00036621094 | 0 |
| ref-01-ground-pair | 0 | 50 | cantilever_historical | 0.0012264759 | 0.1380438 |
| ref-02-facade-pair-and-ground | 0 | 4 | legality | 0 | 0 |
| ref-02-facade-pair-and-ground | 0 | 4 | coverage | 0.76829028 | 0.71840411 |
| ref-02-facade-pair-and-ground | 0 | 4 | spill | 0.00026925484 | 0.014308877 |
| ref-02-facade-pair-and-ground | 0 | 4 | ground | 0 | 0 |
| ref-02-facade-pair-and-ground | 0 | 4 | thickness | 0 | 0 |
| ref-02-facade-pair-and-ground | 0 | 4 | sparsity | 0 | 0 |
| ref-02-facade-pair-and-ground | 0 | 4 | facade | 0.025438234 | 2.233427 |
| ref-02-facade-pair-and-ground | 0 | 4 | access | 0.80521661 | 0.72755505 |
| ref-02-facade-pair-and-ground | 0 | 4 | support | 0.68211716 | 0.78586802 |
| ref-02-facade-pair-and-ground | 0 | 4 | density_binary | 0.0028114936 | 0.014729674 |
| ref-02-facade-pair-and-ground | 0 | 4 | tv | 0.0069776289 | 0.053664334 |
| ref-02-facade-pair-and-ground | 0 | 4 | cantilever_boundary | 0.0016637623 | 0.0017377531 |
| ref-02-facade-pair-and-ground | 0 | 4 | cantilever_historical | 0.0017397481 | 0.0092737556 |
| ref-02-facade-pair-and-ground | 0 | 16 | legality | 0 | 0 |
| ref-02-facade-pair-and-ground | 0 | 16 | coverage | 0.29849055 | 1.5304016 |
| ref-02-facade-pair-and-ground | 0 | 16 | spill | 0.002178242 | 0.16642623 |
| ref-02-facade-pair-and-ground | 0 | 16 | ground | 0 | 0 |
| ref-02-facade-pair-and-ground | 0 | 16 | thickness | 0 | 0 |
| ref-02-facade-pair-and-ground | 0 | 16 | sparsity | 1.1914771 | 63.151387 |
| ref-02-facade-pair-and-ground | 0 | 16 | facade | 0.099451348 | 7.0406235 |
| ref-02-facade-pair-and-ground | 0 | 16 | access | 0.5 | 0 |
| ref-02-facade-pair-and-ground | 0 | 16 | support | 0.15161668 | 3.8091145 |
| ref-02-facade-pair-and-ground | 0 | 16 | density_binary | 0.0021395748 | 0.075344361 |
| ref-02-facade-pair-and-ground | 0 | 16 | tv | 0.022410532 | 0.40235104 |
| ref-02-facade-pair-and-ground | 0 | 16 | cantilever_boundary | 0.00084961532 | 0.0017318447 |
| ref-02-facade-pair-and-ground | 0 | 16 | cantilever_historical | 0.0022902184 | 0.0087946664 |
| ref-02-facade-pair-and-ground | 1 | 4 | legality | 0 | 0 |
| ref-02-facade-pair-and-ground | 1 | 4 | coverage | 0.76333988 | 0.79432774 |
| ref-02-facade-pair-and-ground | 1 | 4 | spill | 0.0002521344 | 0.013239065 |
| ref-02-facade-pair-and-ground | 1 | 4 | ground | 0 | 0 |
| ref-02-facade-pair-and-ground | 1 | 4 | thickness | 0 | 0 |
| ref-02-facade-pair-and-ground | 1 | 4 | sparsity | 0 | 0 |
| ref-02-facade-pair-and-ground | 1 | 4 | facade | 0.021941796 | 2.1025176 |
| ref-02-facade-pair-and-ground | 1 | 4 | access | 0.89743614 | 0.40225715 |
| ref-02-facade-pair-and-ground | 1 | 4 | support | 0.68486029 | 0.78076931 |
| ref-02-facade-pair-and-ground | 1 | 4 | density_binary | 0.0027859665 | 0.013952599 |
| ref-02-facade-pair-and-ground | 1 | 4 | tv | 0.0069824704 | 0.051164065 |
| ref-02-facade-pair-and-ground | 1 | 4 | cantilever_boundary | 0.0016659693 | 0.001754497 |
| ref-02-facade-pair-and-ground | 1 | 4 | cantilever_historical | 0.001723098 | 0.0085429463 |
| ref-02-facade-pair-and-ground | 1 | 16 | legality | 0 | 0 |
| ref-02-facade-pair-and-ground | 1 | 16 | coverage | 0.29536778 | 1.5398963 |
| ref-02-facade-pair-and-ground | 1 | 16 | spill | 0.0021481961 | 0.15975844 |
| ref-02-facade-pair-and-ground | 1 | 16 | ground | 0 | 0 |
| ref-02-facade-pair-and-ground | 1 | 16 | thickness | 0 | 0 |
| ref-02-facade-pair-and-ground | 1 | 16 | sparsity | 1.2371899 | 61.850457 |
| ref-02-facade-pair-and-ground | 1 | 16 | facade | 0.10185391 | 6.6262711 |
| ref-02-facade-pair-and-ground | 1 | 16 | access | 0.80358976 | 1.0637519 |
| ref-02-facade-pair-and-ground | 1 | 16 | support | 0.1435506 | 3.5980316 |
| ref-02-facade-pair-and-ground | 1 | 16 | density_binary | 0.0020546366 | 0.070810444 |
| ref-02-facade-pair-and-ground | 1 | 16 | tv | 0.022431107 | 0.39246122 |
| ref-02-facade-pair-and-ground | 1 | 16 | cantilever_boundary | 0.00085959135 | 0.0010994147 |
| ref-02-facade-pair-and-ground | 1 | 16 | cantilever_historical | 0.0021359168 | 0.019011179 |
| ref-03-wide-gap | 0 | 4 | legality | 0 | 0 |
| ref-03-wide-gap | 0 | 4 | coverage | 0.76779741 | 0.75342964 |
| ref-03-wide-gap | 0 | 4 | spill | 0.00017185882 | 0.0092153676 |
| ref-03-wide-gap | 0 | 4 | ground | 0 | 0 |
| ref-03-wide-gap | 0 | 4 | thickness | 0 | 0 |
| ref-03-wide-gap | 0 | 4 | sparsity | 0 | 0 |
| ref-03-wide-gap | 0 | 4 | facade | 0.0229287 | 1.5854824 |
| ref-03-wide-gap | 0 | 4 | access | 0.7973032 | 0.54729848 |
| ref-03-wide-gap | 0 | 4 | support | 0.67718732 | 0.69477674 |
| ref-03-wide-gap | 0 | 4 | density_binary | 0.0027890727 | 0.011109285 |
| ref-03-wide-gap | 0 | 4 | tv | 0.007093661 | 0.043325683 |
| ref-03-wide-gap | 0 | 4 | cantilever_boundary | 0.0016954982 | 0.0019333662 |
| ref-03-wide-gap | 0 | 4 | cantilever_historical | 0.001794828 | 0.006057387 |
| ref-03-wide-gap | 0 | 16 | legality | 0 | 0 |
| ref-03-wide-gap | 0 | 16 | coverage | 0.27882487 | 1.199731 |
| ref-03-wide-gap | 0 | 16 | spill | 0.0014103237 | 0.1100533 |
| ref-03-wide-gap | 0 | 16 | ground | 0 | 0 |
| ref-03-wide-gap | 0 | 16 | thickness | 0 | 0 |
| ref-03-wide-gap | 0 | 16 | sparsity | 1.0307965 | 42.369817 |
| ref-03-wide-gap | 0 | 16 | facade | 0.066747814 | 5.3877501 |
| ref-03-wide-gap | 0 | 16 | access | 0.5 | 0 |
| ref-03-wide-gap | 0 | 16 | support | 0.11738985 | 3.0137315 |
| ref-03-wide-gap | 0 | 16 | density_binary | 0.0016726997 | 0.05458864 |
| ref-03-wide-gap | 0 | 16 | tv | 0.021042114 | 0.27389668 |
| ref-03-wide-gap | 0 | 16 | cantilever_boundary | 0.0011319834 | 0.0015125294 |
| ref-03-wide-gap | 0 | 16 | cantilever_historical | 0.0024438004 | 0.012683504 |
| ref-03-wide-gap | 1 | 4 | legality | 0 | 0 |
| ref-03-wide-gap | 1 | 4 | coverage | 0.7600252 | 0.8017003 |
| ref-03-wide-gap | 1 | 4 | spill | 0.00018191226 | 0.0098449146 |
| ref-03-wide-gap | 1 | 4 | ground | 0 | 0 |
| ref-03-wide-gap | 1 | 4 | thickness | 0 | 0 |
| ref-03-wide-gap | 1 | 4 | sparsity | 0 | 0 |
| ref-03-wide-gap | 1 | 4 | facade | 0.0267023 | 1.6633958 |
| ref-03-wide-gap | 1 | 4 | access | 0.81659353 | 1.135279 |
| ref-03-wide-gap | 1 | 4 | support | 0.67389095 | 0.73169003 |
| ref-03-wide-gap | 1 | 4 | density_binary | 0.0028158734 | 0.011672097 |
| ref-03-wide-gap | 1 | 4 | tv | 0.0070387125 | 0.044502671 |
| ref-03-wide-gap | 1 | 4 | cantilever_boundary | 0.0017185174 | 0.0020400681 |
| ref-03-wide-gap | 1 | 4 | cantilever_historical | 0.0018306953 | 0.0065694047 |
| ref-03-wide-gap | 1 | 16 | legality | 0 | 0 |
| ref-03-wide-gap | 1 | 16 | coverage | 0.26858306 | 0.89137924 |
| ref-03-wide-gap | 1 | 16 | spill | 0.0015373628 | 0.11852917 |
| ref-03-wide-gap | 1 | 16 | ground | 0 | 0 |
| ref-03-wide-gap | 1 | 16 | thickness | 0 | 0 |
| ref-03-wide-gap | 1 | 16 | sparsity | 1.0912993 | 45.987213 |
| ref-03-wide-gap | 1 | 16 | facade | 0.073362097 | 5.6567282 |
| ref-03-wide-gap | 1 | 16 | access | 0.5 | 0 |
| ref-03-wide-gap | 1 | 16 | support | 0.11848299 | 2.9476777 |
| ref-03-wide-gap | 1 | 16 | density_binary | 0.0017122249 | 0.054202729 |
| ref-03-wide-gap | 1 | 16 | tv | 0.021267053 | 0.29927336 |
| ref-03-wide-gap | 1 | 16 | cantilever_boundary | 0.00109517 | 0.0018531098 |
| ref-03-wide-gap | 1 | 16 | cantilever_historical | 0.0023742893 | 0.015971737 |
| ref-04-asymmetric-heights | 0 | 4 | legality | 0 | 0 |
| ref-04-asymmetric-heights | 0 | 4 | coverage | 0.76110685 | 0.73280513 |
| ref-04-asymmetric-heights | 0 | 4 | spill | 0.00024975737 | 0.012927086 |
| ref-04-asymmetric-heights | 0 | 4 | ground | 0 | 0 |
| ref-04-asymmetric-heights | 0 | 4 | thickness | 0 | 0 |
| ref-04-asymmetric-heights | 0 | 4 | sparsity | 0 | 0 |
| ref-04-asymmetric-heights | 0 | 4 | facade | 0.083401337 | 1.5106012 |
| ref-04-asymmetric-heights | 0 | 4 | access | 0.80846477 | 0.71783139 |
| ref-04-asymmetric-heights | 0 | 4 | support | 0.69845545 | 0.73085229 |
| ref-04-asymmetric-heights | 0 | 4 | density_binary | 0.0038013812 | 0.015759191 |
| ref-04-asymmetric-heights | 0 | 4 | tv | 0.0097383475 | 0.062515563 |
| ref-04-asymmetric-heights | 0 | 4 | cantilever_boundary | 0.0022963758 | 0.0026407518 |
| ref-04-asymmetric-heights | 0 | 4 | cantilever_historical | 0.0024507882 | 0.0086238391 |
| ref-04-asymmetric-heights | 0 | 16 | legality | 0 | 0 |
| ref-04-asymmetric-heights | 0 | 16 | coverage | 0.26033452 | 1.20174 |
| ref-04-asymmetric-heights | 0 | 16 | spill | 0.0019945456 | 0.14828328 |
| ref-04-asymmetric-heights | 0 | 16 | ground | 0 | 0 |
| ref-04-asymmetric-heights | 0 | 16 | thickness | 0 | 0 |
| ref-04-asymmetric-heights | 0 | 16 | sparsity | 1.2860913 | 47.129198 |
| ref-04-asymmetric-heights | 0 | 16 | facade | 0.14435944 | 4.5958917 |
| ref-04-asymmetric-heights | 0 | 16 | access | 0.5 | 0 |
| ref-04-asymmetric-heights | 0 | 16 | support | 0.12223621 | 2.7530556 |
| ref-04-asymmetric-heights | 0 | 16 | density_binary | 0.0022277378 | 0.06803583 |
| ref-04-asymmetric-heights | 0 | 16 | tv | 0.029464839 | 0.36613755 |
| ref-04-asymmetric-heights | 0 | 16 | cantilever_boundary | 0.0014449776 | 0.0026000473 |
| ref-04-asymmetric-heights | 0 | 16 | cantilever_historical | 0.0032266087 | 0.0064883565 |
| ref-04-asymmetric-heights | 1 | 4 | legality | 0 | 0 |
| ref-04-asymmetric-heights | 1 | 4 | coverage | 0.74987221 | 0.78662814 |
| ref-04-asymmetric-heights | 1 | 4 | spill | 0.00024068089 | 0.01256493 |
| ref-04-asymmetric-heights | 1 | 4 | ground | 0 | 0 |
| ref-04-asymmetric-heights | 1 | 4 | thickness | 0 | 0 |
| ref-04-asymmetric-heights | 1 | 4 | sparsity | 0 | 0 |
| ref-04-asymmetric-heights | 1 | 4 | facade | 0.082968891 | 1.4943599 |
| ref-04-asymmetric-heights | 1 | 4 | access | 0.83787823 | 0.68866185 |
| ref-04-asymmetric-heights | 1 | 4 | support | 0.69745678 | 0.74513671 |
| ref-04-asymmetric-heights | 1 | 4 | density_binary | 0.003768364 | 0.015520991 |
| ref-04-asymmetric-heights | 1 | 4 | tv | 0.0095865801 | 0.061363573 |
| ref-04-asymmetric-heights | 1 | 4 | cantilever_boundary | 0.0022657055 | 0.0025424369 |
| ref-04-asymmetric-heights | 1 | 4 | cantilever_historical | 0.0024898848 | 0.0084400141 |
| ref-04-asymmetric-heights | 1 | 16 | legality | 0 | 0 |
| ref-04-asymmetric-heights | 1 | 16 | coverage | 0.25451025 | 0.80798094 |
| ref-04-asymmetric-heights | 1 | 16 | spill | 0.0020710684 | 0.15258333 |
| ref-04-asymmetric-heights | 1 | 16 | ground | 0 | 0 |
| ref-04-asymmetric-heights | 1 | 16 | thickness | 0 | 0 |
| ref-04-asymmetric-heights | 1 | 16 | sparsity | 1.3079456 | 48.073703 |
| ref-04-asymmetric-heights | 1 | 16 | facade | 0.1420908 | 4.775039 |
| ref-04-asymmetric-heights | 1 | 16 | access | 0.5 | 0 |
| ref-04-asymmetric-heights | 1 | 16 | support | 0.12008445 | 2.7853301 |
| ref-04-asymmetric-heights | 1 | 16 | density_binary | 0.0022186935 | 0.068665587 |
| ref-04-asymmetric-heights | 1 | 16 | tv | 0.029693589 | 0.38484425 |
| ref-04-asymmetric-heights | 1 | 16 | cantilever_boundary | 0.0015030226 | 0.0016345042 |
| ref-04-asymmetric-heights | 1 | 16 | cantilever_historical | 0.0032893543 | 0.007927862 |
| ref-06-minimal-smoke | 0 | 4 | legality | 0 | 0 |
| ref-06-minimal-smoke | 0 | 4 | coverage | 0.90270823 | 0.79429934 |
| ref-06-minimal-smoke | 0 | 4 | spill | 0.0001063701 | 0.0057033388 |
| ref-06-minimal-smoke | 0 | 4 | ground | 0 | 0 |
| ref-06-minimal-smoke | 0 | 4 | thickness | 0 | 0 |
| ref-06-minimal-smoke | 0 | 4 | sparsity | 0 | 0 |
| ref-06-minimal-smoke | 0 | 4 | facade | 0.019553274 | 3.4831743 |
| ref-06-minimal-smoke | 0 | 4 | access | 1 | 0 |
| ref-06-minimal-smoke | 0 | 4 | support | 0.38268143 | 2.3649081 |
| ref-06-minimal-smoke | 0 | 4 | density_binary | 0.00093595538 | 0.0088451437 |
| ref-06-minimal-smoke | 0 | 4 | tv | 0.0025586034 | 0.025690252 |
| ref-06-minimal-smoke | 0 | 4 | cantilever_boundary | 0 | 0 |
| ref-06-minimal-smoke | 0 | 4 | cantilever_historical | 0.00060280622 | 0.0062680693 |
| ref-06-minimal-smoke | 0 | 16 | legality | 0 | 0 |
| ref-06-minimal-smoke | 0 | 16 | coverage | 0.86788607 | 2.3548111 |
| ref-06-minimal-smoke | 0 | 16 | spill | 0.00088229519 | 0.068070482 |
| ref-06-minimal-smoke | 0 | 16 | ground | 0 | 0 |
| ref-06-minimal-smoke | 0 | 16 | thickness | 0 | 0 |
| ref-06-minimal-smoke | 0 | 16 | sparsity | 0.00053251861 | 2.0364923 |
| ref-06-minimal-smoke | 0 | 16 | facade | 0.1654968 | 13.765037 |
| ref-06-minimal-smoke | 0 | 16 | access | 1 | 0 |
| ref-06-minimal-smoke | 0 | 16 | support | 0.27210003 | 7.292281 |
| ref-06-minimal-smoke | 0 | 16 | density_binary | 0.0010488019 | 0.050692144 |
| ref-06-minimal-smoke | 0 | 16 | tv | 0.007057779 | 0.25102911 |
| ref-06-minimal-smoke | 0 | 16 | cantilever_boundary | 0 | 0 |
| ref-06-minimal-smoke | 0 | 16 | cantilever_historical | 0.001046709 | 0.010513118 |
| ref-06-minimal-smoke | 1 | 4 | legality | 0 | 0 |
| ref-06-minimal-smoke | 1 | 4 | coverage | 0.9041757 | 0.78724517 |
| ref-06-minimal-smoke | 1 | 4 | spill | 0.00010646732 | 0.005720694 |
| ref-06-minimal-smoke | 1 | 4 | ground | 0 | 0 |
| ref-06-minimal-smoke | 1 | 4 | thickness | 0 | 0 |
| ref-06-minimal-smoke | 1 | 4 | sparsity | 0 | 0 |
| ref-06-minimal-smoke | 1 | 4 | facade | 0.028142631 | 3.4763552 |
| ref-06-minimal-smoke | 1 | 4 | access | 1 | 0 |
| ref-06-minimal-smoke | 1 | 4 | support | 0.37428451 | 2.4062534 |
| ref-06-minimal-smoke | 1 | 4 | density_binary | 0.0009337698 | 0.0086888088 |
| ref-06-minimal-smoke | 1 | 4 | tv | 0.0025229163 | 0.024501319 |
| ref-06-minimal-smoke | 1 | 4 | cantilever_boundary | 0 | 0 |
| ref-06-minimal-smoke | 1 | 4 | cantilever_historical | 0.00060258509 | 0.0063381111 |
| ref-06-minimal-smoke | 1 | 16 | legality | 0 | 0 |
| ref-06-minimal-smoke | 1 | 16 | coverage | 0.89779502 | 2.4959603 |
| ref-06-minimal-smoke | 1 | 16 | spill | 0.00091932312 | 0.073061944 |
| ref-06-minimal-smoke | 1 | 16 | ground | 0 | 0 |
| ref-06-minimal-smoke | 1 | 16 | thickness | 0 | 0 |
| ref-06-minimal-smoke | 1 | 16 | sparsity | 0 | 0 |
| ref-06-minimal-smoke | 1 | 16 | facade | 0.18746439 | 14.532856 |
| ref-06-minimal-smoke | 1 | 16 | access | 1 | 0 |
| ref-06-minimal-smoke | 1 | 16 | support | 0.29771742 | 7.1450399 |
| ref-06-minimal-smoke | 1 | 16 | density_binary | 0.0010788075 | 0.053080087 |
| ref-06-minimal-smoke | 1 | 16 | tv | 0.0072797453 | 0.26740543 |
| ref-06-minimal-smoke | 1 | 16 | cantilever_boundary | 0 | 0 |
| ref-06-minimal-smoke | 1 | 16 | cantilever_historical | 0.00099880365 | 0.014667592 |
| ref-06-minimal-smoke | 0 | 50 | legality | 0 | 0 |
| ref-06-minimal-smoke | 0 | 50 | coverage | 0.78057104 | 0.11495136 |
| ref-06-minimal-smoke | 0 | 50 | spill | 0.0054241465 | 0.01726367 |
| ref-06-minimal-smoke | 0 | 50 | ground | 0 | 0 |
| ref-06-minimal-smoke | 0 | 50 | thickness | 0 | 0 |
| ref-06-minimal-smoke | 0 | 50 | sparsity | 9.337533 | 202.23356 |
| ref-06-minimal-smoke | 0 | 50 | facade | 0.4562436 | 1.8109985 |
| ref-06-minimal-smoke | 0 | 50 | access | 1 | 0 |
| ref-06-minimal-smoke | 0 | 50 | support | 0.0015848726 | 1.4126587 |
| ref-06-minimal-smoke | 0 | 50 | density_binary | 1.7142096e-05 | 0.032930981 |
| ref-06-minimal-smoke | 0 | 50 | tv | 0.020548632 | 0.077255977 |
| ref-06-minimal-smoke | 0 | 50 | cantilever_boundary | 0 | 0 |
| ref-06-minimal-smoke | 0 | 50 | cantilever_historical | 0.0009694355 | 3.6478257e-05 |
