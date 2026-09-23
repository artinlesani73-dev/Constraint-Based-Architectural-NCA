# K2 coefficient sensitivity: complete local comparison

Run `20260923T102524Z_f5e1cc169dea`; recovery gate `20260923T102414Z_eaec7bd1510e`.

68 logical optimizer updates: two fixed recipes x two training seeds x17 scenes once each. 187 evaluation records:136 trained-model,34 original-checkpoint and17 static W1 controls. All development geometry; no unseen-scene claim. Material-budget coefficient30 versus3 is the only difference between paired training arms.

Registered hashes and source snapshot checked. All68 completed checkpoints have matching metadata/update counters and constant learning rate. All255 saved training/evaluation fields have recomputed objective values; every evaluation binary metric and both weighted totals were checked. Actual-loop recovery comparisons pass. This checks saved results, not an independent reimplementation of each formula.

## Geometry and budget outcomes

| Model | Steps | Cases | Connected | Scorable | Empty | Over budget | Under budget | Mean mass/envelope | Illegal / blocked voxels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1_procedural | static | 17 | 17 | 17 | 0 | 0 | 0 | 0.0355383 | 0 / 0 |
| mapped_30-s0 | 16 | 17 | 10 | 17 | 0 | 15 | 0 | 0.185286 | 0 / 0 |
| mapped_30-s0 | 50 | 17 | 10 | 17 | 0 | 15 | 0 | 0.214401 | 0 / 0 |
| mapped_30-s1 | 16 | 17 | 10 | 17 | 0 | 15 | 0 | 0.186258 | 0 / 0 |
| mapped_30-s1 | 50 | 17 | 10 | 17 | 0 | 15 | 0 | 0.214394 | 0 / 0 |
| mass_3-s0 | 16 | 17 | 10 | 17 | 0 | 16 | 0 | 0.209934 | 0 / 0 |
| mass_3-s0 | 50 | 17 | 12 | 17 | 0 | 17 | 0 | 0.273985 | 0 / 0 |
| mass_3-s1 | 16 | 17 | 10 | 17 | 0 | 16 | 0 | 0.20651 | 0 / 0 |
| mass_3-s1 | 50 | 17 | 11 | 17 | 0 | 17 | 0 | 0.268045 | 0 / 0 |
| original_checkpoint | 16 | 17 | 10 | 17 | 0 | 17 | 0 | 0.212616 | 0 / 0 |
| original_checkpoint | 50 | 17 | 10 | 17 | 0 | 17 | 0 | 0.280069 | 0 / 0 |

Budget tolerance is1e-6 around the3%-12% limits. Connected uses binary_v1 at material>0.5. Unscorable cases remain explicit; neither empty nor unscorable is counted connected. W1 is static, not a recurrent trajectory.

## Per-family means and common scoring

| Model | Steps | access | coverage | facade | ground | legality | sparsity | spill | support | thickness | Total mapped_30 | Total mass_3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1_procedural | static | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00692245 | 0.00692245 |
| mapped_30-s0 | 16 | 0.36551 | 0.262209 | 0.0462689 | 0 | 0 | 1.00063 | 0 | 0.108979 | 0 | 43.4108 | 16.3937 |
| mapped_30-s0 | 50 | 0.323529 | 0.172156 | 0.0642207 | 0 | 0 | 1.84366 | 0 | 0 | 0 | 65.1246 | 15.3458 |
| mapped_30-s1 | 16 | 0.362449 | 0.261166 | 0.0478384 | 0 | 0 | 1.02635 | 0 | 0.105033 | 0 | 44.0943 | 16.3829 |
| mapped_30-s1 | 50 | 0.323529 | 0.172188 | 0.0642207 | 0 | 0 | 1.84349 | 0 | 0 | 0 | 65.1205 | 15.3462 |
| mass_3-s0 | 16 | 0.292083 | 0.193573 | 0.0813653 | 0 | 0 | 1.50524 | 0.000324319 | 0.0797041 | 0 | 55.8574 | 15.2159 |
| mass_3-s0 | 50 | 0.205882 | 0.141567 | 0.170665 | 0 | 0 | 4.21151 | 0.00264097 | 0.000365103 | 0 | 134.77 | 21.0594 |
| mass_3-s1 | 16 | 0.301711 | 0.196378 | 0.0680351 | 0 | 0 | 1.48292 | 0.000180961 | 0.0679526 | 0 | 55.1704 | 15.1316 |
| mass_3-s1 | 50 | 0.264706 | 0.147308 | 0.166899 | 0 | 0 | 3.79423 | 0.00242967 | 0.00284905 | 0 | 123.255 | 20.8103 |
| original_checkpoint | 16 | 0.313425 | 0.205677 | 0.0976475 | 0 | 0 | 1.54028 | 0.000505982 | 0.0907591 | 0 | 57.7878 | 16.2002 |
| original_checkpoint | 50 | 0.323529 | 0.150911 | 0.17667 | 0 | 0 | 4.71042 | 0.0029493 | 0.000328426 | 0 | 151.804 | 24.6225 |

Totals use the SAME coefficient set down each column. Lowering a coefficient alone must not be called an improvement. Some zero family residuals arise from hard projection or inactivity.

## Retained regularizers

| Model | Steps | Density/binarization | TV | Boundary cantilever |
|---|---:|---:|---:|---:|
| W1_procedural | static | 0 | 0.00460671 | 0.000463149 |
| mapped_30-s0 | 16 | 0.000763784 | 0.0125382 | 0.000916784 |
| mapped_30-s0 | 50 | 3.44421e-07 | 0.0108118 | 0.00100349 |
| mapped_30-s1 | 16 | 0.000742897 | 0.0125435 | 0.000916574 |
| mapped_30-s1 | 50 | 0 | 0.0108089 | 0.00100349 |
| mass_3-s0 | 16 | 0.000692803 | 0.0133109 | 0.000958405 |
| mass_3-s0 | 50 | 5.26629e-06 | 0.0168911 | 0.00100349 |
| mass_3-s1 | 16 | 0.000557899 | 0.0127732 | 0.000958182 |
| mass_3-s1 | 50 | 4.51533e-05 | 0.0166707 | 0.00100349 |
| original_checkpoint | 16 | 0.000783587 | 0.0137462 | 0.000957511 |
| original_checkpoint | 50 | 7.06099e-06 | 0.0174503 | 0.00100349 |

## Binary support and threshold sensitivity

| Model | Steps | Unsupported voxels total | Radius1 eroded core voxels total | Mean material count >.3 | >.5 | >.7 |
|---|---:|---:|---:|---:|---:|---:|
| W1_procedural | static | 0 | 0 | 43.1176 | 43.1176 | 43.1176 |
| mapped_30-s0 | 16 | 0 | 271 | 282.235 | 269.529 | 227.882 |
| mapped_30-s0 | 50 | 0 | 737 | 282.471 | 282.471 | 282.471 |
| mapped_30-s1 | 16 | 0 | 284 | 282.235 | 271.294 | 231.706 |
| mapped_30-s1 | 50 | 0 | 737 | 282.471 | 282.471 | 282.471 |
| mass_3-s0 | 16 | 0 | 562 | 285.882 | 280.235 | 261.529 |
| mass_3-s0 | 50 | 0 | 737 | 363.882 | 363.706 | 363.353 |
| mass_3-s1 | 16 | 0 | 579 | 283.471 | 280.294 | 262 |
| mass_3-s1 | 50 | 0 | 737 | 358.118 | 356.647 | 354.706 |
| original_checkpoint | 16 | 0 | 543 | 290.765 | 279.941 | 262 |
| original_checkpoint | 50 | 0 | 737 | 371.882 | 371.765 | 371.412 |

Binary radius1 erosion is an independent bulk proxy; the training thickness term uses radius2. Neither is a minimum-thickness or mechanical certification. Unsupported means disconnected from the declared geometric support boundary.

## Paired per-scene tradeoffs: mass_3 versus mapped_30

| Training seed | Steps | Coverage lower / higher / equal | Sparsity lower / higher / equal | Connectivity gained / lost |
|---:|---:|---|---|---|
| 0 | 16 | 17 / 0 / 0 | 0 / 16 / 1 | 0 / 0 |
| 0 | 50 | 8 / 0 / 9 | 0 / 9 / 8 | 2 / 0 |
| 1 | 16 | 17 / 0 / 0 | 0 / 16 / 1 | 0 / 0 |
| 1 | 50 | 8 / 0 / 9 | 0 / 9 / 8 | 1 / 0 |

## Every evaluation case

| Model | Scene | Steps | Connected | Mass/envelope | Coverage | Access | Sparsity | Support | Total mapped_30 | Total mass_3 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| W1_procedural | legacy-easy-seed-000 | static | True | 0.0312076 | 0 | 0 | 0 | 0 | 0.00263632 | 0.00263632 |
| W1_procedural | legacy-easy-seed-001 | static | True | 0.0334789 | 0 | 0 | 0 | 0 | 0.00278891 | 0.00278891 |
| W1_procedural | legacy-easy-seed-002 | static | True | 0.0318525 | 0 | 0 | 0 | 0 | 0.00638408 | 0.00638408 |
| W1_procedural | legacy-easy-seed-003 | static | True | 0.0302216 | 0 | 0 | 0 | 0 | 0.00768255 | 0.00768255 |
| W1_procedural | legacy-easy-seed-004 | static | True | 0.0336842 | 0 | 0 | 0 | 0 | 0.00518011 | 0.00518011 |
| W1_procedural | legacy-easy-seed-005 | static | True | 0.0345489 | 0 | 0 | 0 | 0 | 0.00736261 | 0.00736261 |
| W1_procedural | legacy-easy-seed-006 | static | True | 0.0301428 | 0 | 0 | 0 | 0 | 0.0121086 | 0.0121086 |
| W1_procedural | legacy-easy-seed-007 | static | True | 0.0520156 | 0 | 0 | 0 | 0 | 0.00616849 | 0.00616849 |
| W1_procedural | legacy-easy-seed-008 | static | True | 0.0636704 | 0 | 0 | 0 | 0 | 0.00529135 | 0.00529135 |
| W1_procedural | legacy-easy-seed-009 | static | True | 0.0304965 | 0 | 0 | 0 | 0 | 0.00759789 | 0.00759789 |
| W1_procedural | legacy-easy-seed-010 | static | True | 0.030124 | 0 | 0 | 0 | 0 | 0.00997925 | 0.00997925 |
| W1_procedural | legacy-easy-seed-011 | static | True | 0.030303 | 0 | 0 | 0 | 0 | 0.00737246 | 0.00737246 |
| W1_procedural | ref-01-ground-pair | static | True | 0.0390456 | 0 | 0 | 0 | 0 | 0.00432759 | 0.00432759 |
| W1_procedural | ref-02-facade-pair-and-ground | static | True | 0.0300493 | 0 | 0 | 0 | 0 | 0.00877823 | 0.00877823 |
| W1_procedural | ref-03-wide-gap | static | True | 0.030185 | 0 | 0 | 0 | 0 | 0.00845829 | 0.00845829 |
| W1_procedural | ref-04-asymmetric-heights | static | True | 0.0301138 | 0 | 0 | 0 | 0 | 0.0123517 | 0.0123517 |
| W1_procedural | ref-06-minimal-smoke | static | True | 0.0430108 | 0 | 0 | 0 | 0 | 0.00321321 | 0.00321321 |
| mapped_30-s0 | legacy-easy-seed-000 | 16 | True | 0.189146 | 0.123987 | 0.0363867 | 0.71717 | 0.13093 | 27.0966 | 7.73301 |
| mapped_30-s0 | legacy-easy-seed-000 | 50 | True | 0.227951 | 0 | 0 | 1.74802 | 0 | 53.8064 | 6.60996 |
| mapped_30-s0 | legacy-easy-seed-001 | 16 | True | 0.196403 | 0.139946 | 0.051455 | 0.875606 | 0.136825 | 32.5143 | 8.87294 |
| mapped_30-s0 | legacy-easy-seed-001 | 50 | True | 0.237263 | 0 | 0 | 2.06261 | 0 | 63.2702 | 7.5798 |
| mapped_30-s0 | legacy-easy-seed-002 | 16 | True | 0.285015 | 0.092874 | 0.109535 | 4.08449 | 0.0987248 | 127.663 | 17.3823 |
| mapped_30-s0 | legacy-easy-seed-002 | 50 | True | 0.326907 | 0 | 0 | 6.42157 | 0 | 193.269 | 19.8861 |
| mapped_30-s0 | legacy-easy-seed-003 | 16 | True | 0.220633 | 0.140322 | 0.0394101 | 1.51905 | 0.107456 | 50.9192 | 9.90494 |
| mapped_30-s0 | legacy-easy-seed-003 | 50 | True | 0.25319 | 0 | 0 | 2.66094 | 0 | 80.3901 | 8.5447 |
| mapped_30-s0 | legacy-easy-seed-004 | 16 | False | 0.195071 | 0.244132 | 1 | 0.845355 | 0.127955 | 47.9331 | 25.1085 |
| mapped_30-s0 | legacy-easy-seed-004 | 50 | False | 0.231579 | 0.158524 | 1 | 1.86748 | 0 | 75.7752 | 25.3533 |
| mapped_30-s0 | legacy-easy-seed-005 | 16 | True | 0.222176 | 0.159331 | 0.121268 | 1.56598 | 0.11853 | 53.8633 | 11.5817 |
| mapped_30-s0 | legacy-easy-seed-005 | 50 | True | 0.260077 | 0.057587 | 0 | 2.94322 | 0 | 90.2133 | 10.7463 |
| mapped_30-s0 | legacy-easy-seed-006 | 16 | True | 0.173708 | 0.105043 | 0.0241818 | 0.432682 | 0.113465 | 16.9047 | 5.22227 |
| mapped_30-s0 | legacy-easy-seed-006 | 50 | True | 0.201481 | 0 | 0 | 0.995866 | 0 | 29.947 | 3.05866 |
| mapped_30-s0 | legacy-easy-seed-007 | 16 | True | 0.208147 | 0.107467 | 0 | 1.16549 | 0.0937932 | 40.2686 | 8.8005 |
| mapped_30-s0 | legacy-easy-seed-007 | 50 | True | 0.23407 | 0 | 0 | 1.9518 | 0 | 60.3969 | 7.69827 |
| mapped_30-s0 | legacy-easy-seed-008 | 16 | False | 0.16234 | 0.297289 | 1 | 0.268899 | 0.0906278 | 33.2804 | 26.0201 |
| mapped_30-s0 | legacy-easy-seed-008 | 50 | False | 0.185393 | 0.244389 | 1 | 0.641442 | 0 | 42.4946 | 25.1757 |
| mapped_30-s0 | legacy-easy-seed-009 | 16 | True | 0.257644 | 0.0881939 | 0.0438605 | 2.84187 | 0.0914595 | 89.1939 | 12.4634 |
| mapped_30-s0 | legacy-easy-seed-009 | 50 | True | 0.291489 | 0 | 0 | 4.41129 | 0 | 132.783 | 13.6777 |
| mapped_30-s0 | legacy-easy-seed-010 | 16 | True | 0.208519 | 0.122785 | 0 | 1.17534 | 0.0933615 | 39.1107 | 7.37655 |
| mapped_30-s0 | legacy-easy-seed-010 | 50 | True | 0.237448 | 0 | 0 | 2.06912 | 0 | 62.2385 | 6.37236 |
| mapped_30-s0 | legacy-easy-seed-011 | 16 | True | 0.166533 | 0.111016 | 0.171069 | 0.324805 | 0.112851 | 16.0849 | 7.3152 |
| mapped_30-s0 | legacy-easy-seed-011 | 50 | True | 0.19378 | 0 | 0 | 0.816521 | 0 | 24.8648 | 2.81871 |
| mapped_30-s0 | ref-01-ground-pair | 16 | False | 0.0869778 | 0.822275 | 1 | 0 | 0.0960477 | 36.3326 | 36.3326 |
| mapped_30-s0 | ref-01-ground-pair | 50 | False | 0.0976139 | 0.823526 | 1 | 0 | 0 | 35.5943 | 35.5943 |
| mapped_30-s0 | ref-02-facade-pair-and-ground | 16 | False | 0.165724 | 0.344107 | 0.589378 | 0.313609 | 0.112935 | 27.7797 | 19.3122 |
| mapped_30-s0 | ref-02-facade-pair-and-ground | 50 | False | 0.193596 | 0.266394 | 0.5 | 0.812457 | 0 | 38.6046 | 16.6683 |
| mapped_30-s0 | ref-03-wide-gap | 16 | False | 0.168414 | 0.332604 | 0.512386 | 0.35159 | 0.107811 | 27.437 | 17.9441 |
| mapped_30-s0 | ref-03-wide-gap | 50 | False | 0.194381 | 0.245127 | 0.5 | 0.829883 | 0 | 38.5455 | 16.1387 |
| mapped_30-s0 | ref-04-asymmetric-heights | 16 | False | 0.179377 | 0.291997 | 0.514732 | 0.528845 | 0.100723 | 32.2901 | 18.0113 |
| mapped_30-s0 | ref-04-asymmetric-heights | 50 | False | 0.206023 | 0.217169 | 0.5 | 1.10999 | 0 | 47.0753 | 17.1056 |
| mapped_30-s0 | ref-06-minimal-smoke | 16 | False | 0.0640272 | 0.934184 | 1 | 0 | 0.11915 | 39.3115 | 39.3115 |
| mapped_30-s0 | ref-06-minimal-smoke | 50 | False | 0.0725806 | 0.91393 | 1 | 0 | 0 | 37.8511 | 37.8511 |
| mapped_30-s1 | legacy-easy-seed-000 | 16 | True | 0.191296 | 0.119203 | 0.0304048 | 0.762469 | 0.12271 | 28.2443 | 7.6576 |
| mapped_30-s1 | legacy-easy-seed-000 | 50 | True | 0.227951 | 0 | 0 | 1.74802 | 0 | 53.8064 | 6.60996 |
| mapped_30-s1 | legacy-easy-seed-001 | 16 | True | 0.198442 | 0.134737 | 0.0450922 | 0.922967 | 0.130246 | 33.7124 | 8.79229 |
| mapped_30-s1 | legacy-easy-seed-001 | 50 | True | 0.237263 | 0 | 0 | 2.06261 | 0 | 63.2702 | 7.5798 |
| mapped_30-s1 | legacy-easy-seed-002 | 16 | True | 0.286927 | 0.0923438 | 0.090548 | 4.17967 | 0.0943879 | 130.218 | 17.3666 |
| mapped_30-s1 | legacy-easy-seed-002 | 50 | True | 0.326907 | 0 | 0 | 6.42157 | 0 | 193.269 | 19.8861 |
| mapped_30-s1 | legacy-easy-seed-003 | 16 | True | 0.221858 | 0.137844 | 0.0365028 | 1.55625 | 0.103115 | 51.917 | 9.89823 |
| mapped_30-s1 | legacy-easy-seed-003 | 50 | True | 0.25319 | 0 | 0 | 2.66094 | 0 | 80.3901 | 8.5447 |
| mapped_30-s1 | legacy-easy-seed-004 | 16 | False | 0.196267 | 0.243872 | 1 | 0.872487 | 0.123266 | 48.7279 | 25.1707 |
| mapped_30-s1 | legacy-easy-seed-004 | 50 | False | 0.231579 | 0.158569 | 1 | 1.86748 | 0 | 75.7763 | 25.3544 |
| mapped_30-s1 | legacy-easy-seed-005 | 16 | True | 0.22346 | 0.160594 | 0.115436 | 1.60561 | 0.113244 | 54.978 | 11.6266 |
| mapped_30-s1 | legacy-easy-seed-005 | 50 | True | 0.260077 | 0.057624 | 0 | 2.94322 | 0 | 90.2143 | 10.7472 |
| mapped_30-s1 | legacy-easy-seed-006 | 16 | True | 0.174307 | 0.103103 | 0.0197155 | 0.442384 | 0.110249 | 17.0544 | 5.11006 |
| mapped_30-s1 | legacy-easy-seed-006 | 50 | True | 0.201481 | 0 | 0 | 0.995866 | 0 | 29.947 | 3.05866 |
| mapped_30-s1 | legacy-easy-seed-007 | 16 | True | 0.208591 | 0.10894 | 0 | 1.17727 | 0.0918958 | 40.6339 | 8.84769 |
| mapped_30-s1 | legacy-easy-seed-007 | 50 | True | 0.23407 | 0 | 0 | 1.9518 | 0 | 60.3969 | 7.69827 |
| mapped_30-s1 | legacy-easy-seed-008 | 16 | False | 0.162482 | 0.297284 | 1 | 0.270703 | 0.0878512 | 33.2829 | 25.9739 |
| mapped_30-s1 | legacy-easy-seed-008 | 50 | False | 0.185393 | 0.244518 | 1 | 0.641442 | 0 | 42.4979 | 25.1789 |
| mapped_30-s1 | legacy-easy-seed-009 | 16 | True | 0.25862 | 0.0858315 | 0.0441778 | 2.88232 | 0.0890118 | 90.3418 | 12.5192 |
| mapped_30-s1 | legacy-easy-seed-009 | 50 | True | 0.291489 | 0 | 0 | 4.41129 | 0 | 132.783 | 13.6777 |
| mapped_30-s1 | legacy-easy-seed-010 | 16 | True | 0.209369 | 0.12184 | 0 | 1.19802 | 0.0904895 | 39.7575 | 7.411 |
| mapped_30-s1 | legacy-easy-seed-010 | 50 | True | 0.237448 | 0 | 0 | 2.06912 | 0 | 62.2385 | 6.37236 |
| mapped_30-s1 | legacy-easy-seed-011 | 16 | True | 0.167671 | 0.109644 | 0.166536 | 0.340882 | 0.107591 | 16.4563 | 7.25248 |
| mapped_30-s1 | legacy-easy-seed-011 | 50 | True | 0.19378 | 0 | 0 | 0.816521 | 0 | 24.8648 | 2.81871 |
| mapped_30-s1 | ref-01-ground-pair | 16 | False | 0.0869161 | 0.823099 | 1 | 0 | 0.0942293 | 36.3385 | 36.3385 |
| mapped_30-s1 | ref-01-ground-pair | 50 | False | 0.0976139 | 0.823632 | 1 | 0 | 0 | 35.597 | 35.597 |
| mapped_30-s1 | ref-02-facade-pair-and-ground | 16 | False | 0.166592 | 0.342844 | 0.587506 | 0.325615 | 0.109441 | 28.0522 | 19.2606 |
| mapped_30-s1 | ref-02-facade-pair-and-ground | 50 | False | 0.193596 | 0.266382 | 0.5 | 0.812457 | 0 | 38.6043 | 16.668 |
| mapped_30-s1 | ref-03-wide-gap | 16 | False | 0.169186 | 0.331987 | 0.510245 | 0.362896 | 0.104334 | 27.7008 | 17.9026 |
| mapped_30-s1 | ref-03-wide-gap | 50 | False | 0.194255 | 0.245175 | 0.5 | 0.827073 | 0 | 38.4623 | 16.1313 |
| mapped_30-s1 | ref-04-asymmetric-heights | 16 | False | 0.180464 | 0.291019 | 0.51547 | 0.548393 | 0.0970761 | 32.8625 | 18.0559 |
| mapped_30-s1 | ref-04-asymmetric-heights | 50 | False | 0.206023 | 0.21723 | 0.5 | 1.10999 | 0 | 47.0768 | 17.1072 |
| mapped_30-s1 | ref-06-minimal-smoke | 16 | False | 0.063938 | 0.935628 | 1 | 0 | 0.116423 | 39.3257 | 39.3257 |
| mapped_30-s1 | ref-06-minimal-smoke | 50 | False | 0.0725806 | 0.91407 | 1 | 0 | 0 | 37.8546 | 37.8546 |
| mass_3-s0 | legacy-easy-seed-000 | 16 | True | 0.210757 | 0.0416968 | 0 | 1.23554 | 0.056094 | 39.7591 | 6.39964 |
| mass_3-s0 | legacy-easy-seed-000 | 50 | True | 0.227951 | 0 | 0 | 1.74802 | 0 | 53.8064 | 6.60996 |
| mass_3-s0 | legacy-easy-seed-001 | 16 | True | 0.222371 | 0.0235924 | 0 | 1.57197 | 0.0502132 | 49.3843 | 6.94103 |
| mass_3-s0 | legacy-easy-seed-001 | 50 | True | 0.243086 | 0 | 0 | 2.27252 | 0 | 69.7383 | 8.38027 |
| mass_3-s0 | legacy-easy-seed-002 | 16 | True | 0.307792 | 0.0386415 | 0 | 5.28989 | 0.0439986 | 160.567 | 17.7403 |
| mass_3-s0 | legacy-easy-seed-002 | 50 | True | 0.326907 | 0 | 0 | 6.42157 | 0 | 193.269 | 19.8861 |
| mass_3-s0 | legacy-easy-seed-003 | 16 | True | 0.237808 | 0.0672842 | 0 | 2.08182 | 0.0477362 | 65.0431 | 8.83399 |
| mass_3-s0 | legacy-easy-seed-003 | 50 | True | 0.25319 | 0 | 0 | 2.66094 | 0 | 80.3901 | 8.5447 |
| mass_3-s0 | legacy-easy-seed-004 | 16 | False | 0.216883 | 0.18336 | 0.785959 | 1.40796 | 0.0509395 | 59.7506 | 21.7358 |
| mass_3-s0 | legacy-easy-seed-004 | 50 | True | 0.236842 | 0.125 | 0 | 2.04781 | 0 | 65.4749 | 10.184 |
| mass_3-s0 | legacy-easy-seed-005 | 16 | True | 0.246357 | 0.0748901 | 0 | 2.39492 | 0.0449028 | 74.504 | 9.84104 |
| mass_3-s0 | legacy-easy-seed-005 | 50 | True | 0.262956 | 0.0295351 | 0 | 3.06546 | 0 | 93.2308 | 10.4635 |
| mass_3-s0 | legacy-easy-seed-006 | 16 | True | 0.186816 | 0.0469046 | 0 | 0.669664 | 0.0553544 | 21.7374 | 3.65647 |
| mass_3-s0 | legacy-easy-seed-006 | 50 | True | 0.201481 | 0 | 0 | 0.995866 | 0 | 29.947 | 3.05866 |
| mass_3-s0 | legacy-easy-seed-007 | 16 | True | 0.221924 | 0.0580241 | 0 | 1.55827 | 0.0400694 | 50.3976 | 8.32435 |
| mass_3-s0 | legacy-easy-seed-007 | 50 | True | 0.23407 | 0 | 0 | 1.9518 | 0 | 60.3969 | 7.69827 |
| mass_3-s0 | legacy-easy-seed-008 | 16 | False | 0.177327 | 0.202115 | 0.658632 | 0.492949 | 0.0354783 | 32.1216 | 18.812 |
| mass_3-s0 | legacy-easy-seed-008 | 50 | True | 0.191011 | 0.121623 | 0 | 0.756389 | 0 | 27.8652 | 7.44269 |
| mass_3-s0 | legacy-easy-seed-009 | 16 | True | 0.275606 | 0.0269668 | 0 | 3.63199 | 0.0417186 | 110.394 | 12.3302 |
| mass_3-s0 | legacy-easy-seed-009 | 50 | True | 0.291489 | 0 | 0 | 4.41129 | 0 | 132.783 | 13.6777 |
| mass_3-s0 | legacy-easy-seed-010 | 16 | True | 0.223 | 0.05761 | 0 | 1.59134 | 0.0453154 | 49.6846 | 6.71846 |
| mass_3-s0 | legacy-easy-seed-010 | 50 | True | 0.237448 | 0 | 0 | 2.06912 | 0 | 62.2385 | 6.37236 |
| mass_3-s0 | legacy-easy-seed-011 | 16 | True | 0.181091 | 0.0322826 | 0 | 0.559825 | 0.0494932 | 18.2818 | 3.16649 |
| mass_3-s0 | legacy-easy-seed-011 | 50 | True | 0.19378 | 0 | 0 | 0.816521 | 0 | 24.8648 | 2.81871 |
| mass_3-s0 | ref-01-ground-pair | 16 | False | 0.14298 | 0.775734 | 1 | 0.0792126 | 0.254537 | 40.1 | 37.9613 |
| mass_3-s0 | ref-01-ground-pair | 50 | False | 0.456158 | 0.701426 | 1 | 16.9503 | 0.00178928 | 546.537 | 88.878 |
| mass_3-s0 | ref-02-facade-pair-and-ground | 16 | False | 0.199213 | 0.270068 | 0.520828 | 0.94121 | 0.12293 | 44.4972 | 19.0845 |
| mass_3-s0 | ref-02-facade-pair-and-ground | 50 | False | 0.343642 | 0.189122 | 0.5 | 7.50234 | 0.000353937 | 240.84 | 38.277 |
| mass_3-s0 | ref-03-wide-gap | 16 | False | 0.197996 | 0.2734 | 0.5 | 0.912513 | 0.100929 | 43.0256 | 18.3878 |
| mass_3-s0 | ref-03-wide-gap | 50 | False | 0.305322 | 0.244019 | 0.5 | 5.15166 | 0.000591268 | 171.126 | 32.0315 |
| mass_3-s0 | ref-04-asymmetric-heights | 16 | False | 0.20832 | 0.25001 | 0.5 | 1.17006 | 0.0955942 | 50.8737 | 19.2821 |
| mass_3-s0 | ref-04-asymmetric-heights | 50 | False | 0.31842 | 0.215311 | 0.5 | 5.90559 | 0.000583231 | 193.595 | 34.1443 |
| mass_3-s0 | ref-06-minimal-smoke | 16 | False | 0.11263 | 0.868163 | 1 | 0 | 0.219666 | 39.4546 | 39.4546 |
| mass_3-s0 | ref-06-minimal-smoke | 50 | False | 0.333984 | 0.780596 | 1 | 6.86839 | 0.00288904 | 244.99 | 59.543 |
| mass_3-s1 | legacy-easy-seed-000 | 16 | True | 0.212048 | 0.0353678 | 0 | 1.27092 | 0.0517698 | 40.6635 | 6.34869 |
| mass_3-s1 | legacy-easy-seed-000 | 50 | True | 0.227951 | 0 | 0 | 1.74802 | 0 | 53.8064 | 6.60996 |
| mass_3-s1 | legacy-easy-seed-001 | 16 | True | 0.223411 | 0.0165546 | 0 | 1.60408 | 0.0462492 | 50.1635 | 6.85326 |
| mass_3-s1 | legacy-easy-seed-001 | 50 | True | 0.24163 | 0 | 0 | 2.21909 | 0 | 68.0935 | 8.1781 |
| mass_3-s1 | legacy-easy-seed-002 | 16 | True | 0.308887 | 0.0386208 | 0 | 5.35174 | 0.0414757 | 162.417 | 17.9203 |
| mass_3-s1 | legacy-easy-seed-002 | 50 | True | 0.326907 | 0 | 0 | 6.42157 | 0 | 193.269 | 19.8861 |
| mass_3-s1 | legacy-easy-seed-003 | 16 | True | 0.238555 | 0.0636113 | 0 | 2.10828 | 0.0452379 | 65.7322 | 8.80852 |
| mass_3-s1 | legacy-easy-seed-003 | 50 | True | 0.25319 | 0 | 0 | 2.66094 | 0 | 80.3901 | 8.5447 |
| mass_3-s1 | legacy-easy-seed-004 | 16 | False | 0.216666 | 0.186021 | 0.880263 | 1.40165 | 0.0482119 | 60.999 | 23.1545 |
| mass_3-s1 | legacy-easy-seed-004 | 50 | False | 0.234737 | 0.158125 | 1 | 1.97468 | 0 | 79.0856 | 25.7692 |
| mass_3-s1 | legacy-easy-seed-005 | 16 | True | 0.246352 | 0.0803319 | 0 | 2.39472 | 0.042491 | 74.6047 | 9.94737 |
| mass_3-s1 | legacy-easy-seed-005 | 50 | True | 0.262956 | 0.0297058 | 0 | 3.06546 | 0 | 93.2351 | 10.4678 |
| mass_3-s1 | legacy-easy-seed-006 | 16 | True | 0.187141 | 0.0457219 | 0 | 0.676178 | 0.0539271 | 21.8907 | 3.63386 |
| mass_3-s1 | legacy-easy-seed-006 | 50 | True | 0.201481 | 0 | 0 | 0.995866 | 0 | 29.947 | 3.05866 |
| mass_3-s1 | legacy-easy-seed-007 | 16 | True | 0.222233 | 0.0598806 | 0 | 1.56773 | 0.0387909 | 50.7033 | 8.37462 |
| mass_3-s1 | legacy-easy-seed-007 | 50 | True | 0.23407 | 0 | 0 | 1.9518 | 0 | 60.3969 | 7.69827 |
| mass_3-s1 | legacy-easy-seed-008 | 16 | False | 0.175554 | 0.234639 | 0.731779 | 0.46294 | 0.0348925 | 33.0981 | 20.5987 |
| mass_3-s1 | legacy-easy-seed-008 | 50 | True | 0.189139 | 0.180772 | 0 | 0.717021 | 0 | 28.1987 | 8.83915 |
| mass_3-s1 | legacy-easy-seed-009 | 16 | True | 0.276216 | 0.0251344 | 0 | 3.66052 | 0.0400957 | 111.195 | 12.3608 |
| mass_3-s1 | legacy-easy-seed-009 | 50 | True | 0.291489 | 0 | 0 | 4.41129 | 0 | 132.783 | 13.6777 |
| mass_3-s1 | legacy-easy-seed-010 | 16 | True | 0.223571 | 0.056133 | 0 | 1.60905 | 0.0435023 | 50.1679 | 6.72341 |
| mass_3-s1 | legacy-easy-seed-010 | 50 | True | 0.237448 | 0 | 0 | 2.06912 | 0 | 62.2385 | 6.37236 |
| mass_3-s1 | legacy-easy-seed-011 | 16 | True | 0.181713 | 0.0297919 | 0 | 0.571273 | 0.0471382 | 18.5554 | 3.13101 |
| mass_3-s1 | legacy-easy-seed-011 | 50 | True | 0.19378 | 0 | 0 | 0.816521 | 0 | 24.8648 | 2.81871 |
| mass_3-s1 | ref-01-ground-pair | 16 | False | 0.120729 | 0.787054 | 1 | 7.97997e-05 | 0.197071 | 36.6531 | 36.6509 |
| mass_3-s1 | ref-01-ground-pair | 50 | False | 0.42227 | 0.705086 | 1 | 13.705 | 0.0161978 | 449.225 | 79.1894 |
| mass_3-s1 | ref-02-facade-pair-and-ground | 16 | False | 0.190946 | 0.27598 | 0.517048 | 0.75501 | 0.0970285 | 38.5149 | 18.1296 |
| mass_3-s1 | ref-02-facade-pair-and-ground | 50 | False | 0.331162 | 0.189266 | 0.5 | 6.68841 | 0.00938078 | 216.331 | 35.7439 |
| mass_3-s1 | ref-03-wide-gap | 16 | False | 0.191187 | 0.274122 | 0.5 | 0.760141 | 0.080987 | 38.0661 | 17.5423 |
| mass_3-s1 | ref-03-wide-gap | 50 | False | 0.29652 | 0.244159 | 0.5 | 4.6739 | 0.00331884 | 156.689 | 30.4939 |
| mass_3-s1 | ref-04-asymmetric-heights | 16 | False | 0.202271 | 0.247749 | 0.5 | 1.01527 | 0.0738605 | 45.8074 | 18.3951 |
| mass_3-s1 | ref-04-asymmetric-heights | 50 | False | 0.308768 | 0.215692 | 0.5 | 5.34503 | 0.0057121 | 176.704 | 32.3878 |
| mass_3-s1 | ref-06-minimal-smoke | 16 | False | 0.0931969 | 0.881718 | 1 | 0 | 0.172465 | 38.6644 | 38.6644 |
| mass_3-s1 | ref-06-minimal-smoke | 50 | False | 0.303269 | 0.781434 | 1 | 5.03815 | 0.0138243 | 190.07 | 54.0401 |
| original_checkpoint | legacy-easy-seed-000 | 16 | True | 0.210868 | 0.0555873 | 0 | 1.23856 | 0.0561157 | 40.2317 | 6.79063 |
| original_checkpoint | legacy-easy-seed-000 | 50 | True | 0.227951 | 0 | 0 | 1.74802 | 0 | 53.8064 | 6.60996 |
| original_checkpoint | legacy-easy-seed-001 | 16 | True | 0.222215 | 0.0415462 | 0 | 1.56718 | 0.0513312 | 49.7299 | 7.41614 |
| original_checkpoint | legacy-easy-seed-001 | 50 | True | 0.24237 | 0 | 0 | 2.24618 | 0 | 68.9276 | 8.28075 |
| original_checkpoint | legacy-easy-seed-002 | 16 | True | 0.307943 | 0.0451521 | 0 | 5.29836 | 0.0439264 | 160.999 | 17.9435 |
| original_checkpoint | legacy-easy-seed-002 | 50 | True | 0.326907 | 0 | 0 | 6.42157 | 0 | 193.269 | 19.8861 |
| original_checkpoint | legacy-easy-seed-003 | 16 | True | 0.237728 | 0.0756512 | 0 | 2.07896 | 0.0478732 | 65.1806 | 9.04859 |
| original_checkpoint | legacy-easy-seed-003 | 50 | True | 0.25319 | 0 | 0 | 2.66094 | 0 | 80.3901 | 8.5447 |
| original_checkpoint | legacy-easy-seed-004 | 16 | False | 0.216301 | 0.193901 | 0.936545 | 1.39109 | 0.0514956 | 61.7736 | 24.214 |
| original_checkpoint | legacy-easy-seed-004 | 50 | False | 0.234737 | 0.158291 | 1 | 1.97468 | 0 | 79.0897 | 25.7733 |
| original_checkpoint | legacy-easy-seed-005 | 16 | True | 0.245796 | 0.0908186 | 0 | 2.37368 | 0.0453225 | 74.2764 | 10.187 |
| original_checkpoint | legacy-easy-seed-005 | 50 | True | 0.262956 | 0.029676 | 0 | 3.06546 | 0 | 93.2343 | 10.467 |
| original_checkpoint | legacy-easy-seed-006 | 16 | True | 0.186566 | 0.0527613 | 0 | 0.66466 | 0.0565699 | 21.7484 | 3.80254 |
| original_checkpoint | legacy-easy-seed-006 | 50 | True | 0.201481 | 0 | 0 | 0.995866 | 0 | 29.947 | 3.05866 |
| original_checkpoint | legacy-easy-seed-007 | 16 | True | 0.221525 | 0.0697068 | 0 | 1.5461 | 0.0412056 | 50.3343 | 8.58959 |
| original_checkpoint | legacy-easy-seed-007 | 50 | True | 0.23407 | 0 | 0 | 1.9518 | 0 | 60.3969 | 7.69827 |
| original_checkpoint | legacy-easy-seed-008 | 16 | False | 0.175094 | 0.253051 | 0.870332 | 0.455295 | 0.0366508 | 35.4602 | 23.1672 |
| original_checkpoint | legacy-easy-seed-008 | 50 | False | 0.187266 | 0.243418 | 1 | 0.678706 | 0 | 43.652 | 25.3269 |
| original_checkpoint | legacy-easy-seed-009 | 16 | True | 0.275364 | 0.0341856 | 0 | 3.62069 | 0.0423778 | 110.248 | 12.4889 |
| original_checkpoint | legacy-easy-seed-009 | 50 | True | 0.291489 | 0 | 0 | 4.41129 | 0 | 132.783 | 13.6777 |
| original_checkpoint | legacy-easy-seed-010 | 16 | True | 0.22273 | 0.0678651 | 0 | 1.58301 | 0.0462174 | 49.7052 | 6.96389 |
| original_checkpoint | legacy-easy-seed-010 | 50 | True | 0.237448 | 0 | 0 | 2.06912 | 0 | 62.2385 | 6.37236 |
| original_checkpoint | legacy-easy-seed-011 | 16 | True | 0.180861 | 0.0443151 | 0 | 0.555609 | 0.0506196 | 18.4805 | 3.47906 |
| original_checkpoint | legacy-easy-seed-011 | 50 | True | 0.19378 | 0 | 0 | 0.816521 | 0 | 24.8648 | 2.81871 |
| original_checkpoint | ref-01-ground-pair | 16 | False | 0.16088 | 0.781345 | 1 | 0.250673 | 0.311887 | 46.7569 | 39.9887 |
| original_checkpoint | ref-01-ground-pair | 50 | False | 0.490273 | 0.703013 | 1 | 20.5653 | 0.00200813 | 655.201 | 99.9382 |
| original_checkpoint | ref-02-facade-pair-and-ground | 16 | False | 0.207549 | 0.277975 | 0.52135 | 1.14973 | 0.148102 | 51.5339 | 20.4912 |
| original_checkpoint | ref-02-facade-pair-and-ground | 50 | False | 0.359159 | 0.189372 | 0.5 | 8.57957 | 0.00102131 | 273.387 | 41.7386 |
| original_checkpoint | ref-03-wide-gap | 16 | False | 0.203793 | 0.281781 | 0.5 | 1.05319 | 0.121131 | 47.8916 | 19.4555 |
| original_checkpoint | ref-03-wide-gap | 50 | False | 0.320093 | 0.244281 | 0.5 | 6.00557 | 0.000707322 | 197.033 | 34.8825 |
| original_checkpoint | ref-04-asymmetric-heights | 16 | False | 0.215054 | 0.255671 | 0.5 | 1.35529 | 0.11416 | 56.993 | 20.4001 |
| original_checkpoint | ref-04-asymmetric-heights | 50 | False | 0.326331 | 0.215782 | 0.5 | 6.38588 | 0.000270124 | 208.133 | 35.7141 |
| original_checkpoint | ref-06-minimal-smoke | 16 | False | 0.12421 | 0.875202 | 1 | 0.00265893 | 0.27792 | 41.0491 | 40.9773 |
| original_checkpoint | ref-06-minimal-smoke | 50 | False | 0.371669 | 0.781663 | 1 | 9.50061 | 0.00157635 | 324.312 | 67.7951 |

## Every training update

| Model | Update | Scene | Objective before update | Norm before clipping | Seconds incl. evidence |
|---|---:|---|---:|---:|---:|
| mapped_30-s0 | 1 | legacy-easy-seed-002 | 164.449 | 693.283 | 3.264 |
| mapped_30-s0 | 2 | legacy-easy-seed-010 | 47.277 | 249.671 | 3.197 |
| mapped_30-s0 | 3 | legacy-easy-seed-003 | 62.7768 | 307.503 | 3.381 |
| mapped_30-s0 | 4 | legacy-easy-seed-011 | 18.8479 | 51.5048 | 3.137 |
| mapped_30-s0 | 5 | legacy-easy-seed-000 | 39.6787 | 294.431 | 3.034 |
| mapped_30-s0 | 6 | legacy-easy-seed-004 | 61.7928 | 281.855 | 3.108 |
| mapped_30-s0 | 7 | legacy-easy-seed-007 | 44.9765 | 250.17 | 3.058 |
| mapped_30-s0 | 8 | legacy-easy-seed-005 | 62.9621 | 446.106 | 2.655 |
| mapped_30-s0 | 9 | ref-06-minimal-smoke | 38.858 | 159.504 | 2.793 |
| mapped_30-s0 | 10 | ref-01-ground-pair | 35.5788 | 173.865 | 2.660 |
| mapped_30-s0 | 11 | ref-02-facade-pair-and-ground | 30.8978 | 219.906 | 2.577 |
| mapped_30-s0 | 12 | legacy-easy-seed-006 | 21.9328 | 115.469 | 2.592 |
| mapped_30-s0 | 13 | legacy-easy-seed-009 | 93.5114 | 631.665 | 2.570 |
| mapped_30-s0 | 14 | ref-03-wide-gap | 29.8501 | 117.186 | 2.596 |
| mapped_30-s0 | 15 | legacy-easy-seed-008 | 34.3282 | 117.061 | 2.509 |
| mapped_30-s0 | 16 | legacy-easy-seed-001 | 35.0251 | 496.803 | 2.662 |
| mapped_30-s0 | 17 | ref-04-asymmetric-heights | 31.6449 | 205.856 | 2.548 |
| mapped_30-s1 | 1 | legacy-easy-seed-001 | 55.8393 | 368.365 | 2.834 |
| mapped_30-s1 | 2 | legacy-easy-seed-010 | 45.7573 | 237.726 | 2.811 |
| mapped_30-s1 | 3 | ref-03-wide-gap | 44.2203 | 1257.32 | 2.566 |
| mapped_30-s1 | 4 | ref-04-asymmetric-heights | 49.8953 | 1266.71 | 2.578 |
| mapped_30-s1 | 5 | legacy-easy-seed-007 | 48.9846 | 185.321 | 2.552 |
| mapped_30-s1 | 6 | ref-01-ground-pair | 37.4939 | 390.718 | 2.611 |
| mapped_30-s1 | 7 | legacy-easy-seed-003 | 61.2817 | 393.136 | 2.503 |
| mapped_30-s1 | 8 | legacy-easy-seed-004 | 58.226 | 352.107 | 2.534 |
| mapped_30-s1 | 9 | legacy-easy-seed-005 | 62.6903 | 488.126 | 2.543 |
| mapped_30-s1 | 10 | legacy-easy-seed-008 | 33.6744 | 126.657 | 2.521 |
| mapped_30-s1 | 11 | legacy-easy-seed-000 | 31.5273 | 331.404 | 2.538 |
| mapped_30-s1 | 12 | legacy-easy-seed-009 | 97.1144 | 627.918 | 2.574 |
| mapped_30-s1 | 13 | legacy-easy-seed-002 | 142.21 | 1100.51 | 2.570 |
| mapped_30-s1 | 14 | ref-06-minimal-smoke | 38.5983 | 42.9897 | 2.823 |
| mapped_30-s1 | 15 | ref-02-facade-pair-and-ground | 28.8969 | 125.471 | 2.834 |
| mapped_30-s1 | 16 | legacy-easy-seed-011 | 16.9292 | 109.276 | 2.762 |
| mapped_30-s1 | 17 | legacy-easy-seed-006 | 19.3328 | 105.02 | 2.537 |
| mass_3-s0 | 1 | legacy-easy-seed-002 | 26.3764 | 43.8729 | 2.831 |
| mass_3-s0 | 2 | legacy-easy-seed-010 | 6.19071 | 12.6474 | 2.503 |
| mass_3-s0 | 3 | legacy-easy-seed-003 | 8.46047 | 8.07348 | 2.560 |
| mass_3-s0 | 4 | legacy-easy-seed-011 | 4.56096 | 102.091 | 2.682 |
| mass_3-s0 | 5 | legacy-easy-seed-000 | 7.39253 | 26.4779 | 2.588 |
| mass_3-s0 | 6 | legacy-easy-seed-004 | 24.1775 | 172.718 | 2.558 |
| mass_3-s0 | 7 | legacy-easy-seed-007 | 6.82498 | 11.5104 | 2.542 |
| mass_3-s0 | 8 | legacy-easy-seed-005 | 9.86241 | 23.7792 | 2.563 |
| mass_3-s0 | 9 | ref-06-minimal-smoke | 41.4045 | 183.2 | 2.682 |
| mass_3-s0 | 10 | ref-01-ground-pair | 40.9552 | 423.087 | 2.555 |
| mass_3-s0 | 11 | ref-02-facade-pair-and-ground | 20.5825 | 260.928 | 2.842 |
| mass_3-s0 | 12 | legacy-easy-seed-006 | 4.88596 | 77.4305 | 2.627 |
| mass_3-s0 | 13 | legacy-easy-seed-009 | 12.8681 | 24.6264 | 2.583 |
| mass_3-s0 | 14 | ref-03-wide-gap | 19.7749 | 205.462 | 2.613 |
| mass_3-s0 | 15 | legacy-easy-seed-008 | 17.3382 | 397.33 | 2.721 |
| mass_3-s0 | 16 | legacy-easy-seed-001 | 8.794 | 29.013 | 2.633 |
| mass_3-s0 | 17 | ref-04-asymmetric-heights | 19.5246 | 193.736 | 2.600 |
| mass_3-s1 | 1 | legacy-easy-seed-001 | 14.953 | 88.2884 | 2.548 |
| mass_3-s1 | 2 | legacy-easy-seed-010 | 6.8205 | 8.51152 | 2.527 |
| mass_3-s1 | 3 | ref-03-wide-gap | 20.6709 | 193.808 | 2.592 |
| mass_3-s1 | 4 | ref-04-asymmetric-heights | 22.303 | 185.188 | 2.587 |
| mass_3-s1 | 5 | legacy-easy-seed-007 | 7.39818 | 20.6038 | 2.549 |
| mass_3-s1 | 6 | ref-01-ground-pair | 39.9174 | 373.776 | 2.533 |
| mass_3-s1 | 7 | legacy-easy-seed-003 | 8.7299 | 13.7648 | 2.535 |
| mass_3-s1 | 8 | legacy-easy-seed-004 | 24.8874 | 18.8093 | 2.736 |
| mass_3-s1 | 9 | legacy-easy-seed-005 | 10.3289 | 58.7578 | 2.488 |
| mass_3-s1 | 10 | legacy-easy-seed-008 | 21.547 | 292.679 | 2.573 |
| mass_3-s1 | 11 | legacy-easy-seed-000 | 6.46141 | 14.4959 | 2.607 |
| mass_3-s1 | 12 | legacy-easy-seed-009 | 12.6423 | 19.8773 | 2.845 |
| mass_3-s1 | 13 | legacy-easy-seed-002 | 17.7638 | 50.8314 | 2.561 |
| mass_3-s1 | 14 | ref-06-minimal-smoke | 39.3554 | 267.703 | 2.647 |
| mass_3-s1 | 15 | ref-02-facade-pair-and-ground | 18.166 | 259.185 | 2.601 |
| mass_3-s1 | 16 | legacy-easy-seed-011 | 4.25663 | 78.6908 | 2.631 |
| mass_3-s1 | 17 | legacy-easy-seed-006 | 3.73293 | 20.1433 | 2.570 |

Different scenes appear at different updates; this table is not a fixed-scene learning curve. Timing includes local contention/evidence overhead and is not a deployment benchmark. Seventeen updates do not establish convergence. No model architecture, production serving defaults or original checkpoint was changed. No paid compute, Drive operation or deployment.
