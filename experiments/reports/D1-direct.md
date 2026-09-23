# D1 full direct-field control

Run `20260923T105246Z_ab0d4a430b4c`. Per-scene raw voxel optimization, not NCA training.

Both recipes start from the same weak0.15 scaffold. No solved W1 initialization. Every case uses its own voxel parameters, Adam0.05 and the fixed nine-family/three-regularizer objective. Optimizer steps are not NCA growth steps and the compute budgets are not matched.

## Initial and final summary

| Recipe | State | Cases | Connected | In3%-12% budget | Connected AND in budget | Mean material/envelope | Illegal / blocked / unsupported voxels |
|---|---|---:|---:|---:|---:|---:|---|
| mapped_30 | initial | 17 | 0 | 15 | 0 | 0.03737935 | 0 / 0 / 0 |
| mass_3 | initial | 17 | 0 | 15 | 0 | 0.03737935 | 0 / 0 / 0 |
| mapped_30 | final | 17 | 17 | 12 | 12 | 0.05411428 | 0 / 0 / 0 |
| mass_3 | final | 17 | 17 | 10 | 10 | 0.08728739 | 0 / 0 / 0 |

Budget tolerance1e-6; connectivity uses material>0.5. Connected-and-in-budget is a limited conjunction, not a full nine-family or architectural success claim.

## Final per-family means

| Recipe | access | coverage | facade | ground | legality | sparsity | spill | support | thickness | Total mapped_30 | Total mass_3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mapped_30 | 0.001642504 | 0.001022727 | 0.0005874099 | 0 | 0 | 0.00214952 | 0 | 0.04116613 | 0 | 0.4588034 | 0.4007664 |
| mass_3 | 0 | 0.0005478797 | 0.01010617 | 0 | 0 | 0.07624381 | 0 | 0.01841098 | 0 | 2.561134 | 0.5025508 |

## Every optimized case

| Recipe | Scene | Updates | Connected | Mass/envelope | Coverage | Access | Sparsity | Total mapped_30 | Total mass_3 | Worker seconds |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mapped_30 | legacy-easy-seed-000 | 32 | True | 0.04372453 | 0 | 0 | 0 | 0.3494425 | 0.3494425 | 12.635 |
| mass_3 | legacy-easy-seed-000 | 32 | True | 0.02985075 | 0 | 0 | 0.0001492538 | 0.007140517 | 0.003110666 | 12.579 |
| mapped_30 | legacy-easy-seed-001 | 32 | True | 0.03870315 | 0 | 0 | 0 | 0.003281014 | 0.003281014 | 12.753 |
| mass_3 | legacy-easy-seed-001 | 32 | True | 0.03890871 | 0 | 0 | 0 | 0.003294341 | 0.003294341 | 12.006 |
| mapped_30 | legacy-easy-seed-002 | 32 | True | 0.03185247 | 0 | 0 | 0 | 0.00638408 | 0.00638408 | 12.544 |
| mass_3 | legacy-easy-seed-002 | 32 | True | 0.03185247 | 0 | 0 | 0 | 0.00638408 | 0.00638408 | 12.541 |
| mapped_30 | legacy-easy-seed-003 | 32 | True | 0.02283412 | 0 | 0 | 0.007165883 | 0.2217721 | 0.02829322 | 12.496 |
| mass_3 | legacy-easy-seed-003 | 32 | True | 0.1648546 | 0 | 0 | 0.3017907 | 9.366885 | 1.218534 | 14.802 |
| mapped_30 | legacy-easy-seed-004 | 32 | True | 0.03368421 | 0 | 0 | 0 | 0.005180113 | 0.005180113 | 13.326 |
| mass_3 | legacy-easy-seed-004 | 32 | True | 0.03368421 | 0 | 0 | 0 | 0.005180113 | 0.005180113 | 12.336 |
| mapped_30 | legacy-easy-seed-005 | 32 | True | 0.03862239 | 0 | 0 | 0 | 0.008040788 | 0.008040788 | 12.253 |
| mass_3 | legacy-easy-seed-005 | 32 | True | 0.03825963 | 0 | 0 | 0 | 0.007988686 | 0.007988686 | 12.127 |
| mapped_30 | legacy-easy-seed-006 | 32 | True | 0.01791145 | 0 | 0 | 0.01208855 | 0.396036 | 0.06964516 | 12.102 |
| mass_3 | legacy-easy-seed-006 | 32 | True | 0.1645178 | 0 | 0 | 0.2972747 | 8.997944 | 0.9715272 | 12.107 |
| mapped_30 | legacy-easy-seed-007 | 32 | True | 0.0487689 | 0 | 0 | 0 | 0.5225794 | 0.5225794 | 12.285 |
| mass_3 | legacy-easy-seed-007 | 32 | True | 0.03338475 | 0 | 0 | 0 | 0.9144268 | 0.9144268 | 12.331 |
| mapped_30 | legacy-easy-seed-008 | 32 | True | 0.08003628 | 0 | 0 | 0 | 1.59359 | 1.59359 | 12.602 |
| mass_3 | legacy-easy-seed-008 | 32 | True | 0.03932584 | 0 | 0 | 0 | 0.8832205 | 0.8832205 | 12.406 |
| mapped_30 | legacy-easy-seed-009 | 32 | True | 0.02624113 | 0 | 0 | 0.003758864 | 0.119913 | 0.01842361 | 12.219 |
| mass_3 | legacy-easy-seed-009 | 32 | True | 0.04062465 | 0 | 0 | 0 | 0.07420378 | 0.07420378 | 12.195 |
| mapped_30 | legacy-easy-seed-010 | 32 | True | 0.02175056 | 0 | 0 | 0.008249441 | 0.2859194 | 0.06318448 | 11.317 |
| mass_3 | legacy-easy-seed-010 | 32 | True | 0.160553 | 0 | 0 | 0.246682 | 8.397345 | 1.73693 | 11.272 |
| mapped_30 | legacy-easy-seed-011 | 32 | True | 0.02472089 | 0 | 0 | 0.005279107 | 0.1642532 | 0.02171737 | 11.516 |
| mass_3 | legacy-easy-seed-011 | 32 | True | 0.1486505 | 0 | 0 | 0.1231281 | 3.725679 | 0.401221 | 11.586 |
| mapped_30 | ref-01-ground-pair | 32 | True | 0.1072894 | 0 | 0 | 0 | 0.008973205 | 0.008973205 | 11.372 |
| mass_3 | ref-01-ground-pair | 32 | True | 0.1045388 | 0 | 0 | 0 | 0.008848749 | 0.008848749 | 11.485 |
| mapped_30 | ref-02-facade-pair-and-ground | 32 | True | 0.06203443 | 0 | 0 | 0 | 0.01050402 | 0.01050402 | 11.400 |
| mass_3 | ref-02-facade-pair-and-ground | 32 | True | 0.06112096 | 0 | 0 | 0 | 0.01060261 | 0.01060261 | 11.830 |
| mapped_30 | ref-03-wide-gap | 32 | True | 0.1176107 | 0.0121379 | 0.02792257 | 0 | 2.496455 | 2.496455 | 11.424 |
| mass_3 | ref-03-wide-gap | 32 | True | 0.15453 | 0.007553039 | 0 | 0.1788477 | 6.335432 | 1.506543 | 11.388 |
| mapped_30 | ref-04-asymmetric-heights | 32 | True | 0.1187301 | 0.005248458 | 0 | 0 | 1.601076 | 1.601076 | 11.436 |
| mass_3 | ref-04-asymmetric-heights | 32 | True | 0.1514401 | 0.001760916 | 0 | 0.1482723 | 4.788318 | 0.7849662 | 11.428 |
| mapped_30 | ref-06-minimal-smoke | 32 | True | 0.08542801 | 0 | 0 | 0 | 0.006259237 | 0.006259237 | 11.454 |
| mass_3 | ref-06-minimal-smoke | 32 | True | 0.08778885 | 0 | 0 | 0 | 0.006382785 | 0.006382785 | 11.337 |

## Preserved K2 and W1 controls on these same scenes

| Model | Scene | NCA growth steps | Connected | Mass/envelope | Total mapped_30 | Total mass_3 |
|---|---|---:|---|---:|---:|---:|
| mapped_30-s0 | legacy-easy-seed-000 | 16 | True | 0.1891457 | 27.09659 | 7.733009 |
| mapped_30-s0 | legacy-easy-seed-000 | 50 | True | 0.2279512 | 53.80644 | 6.609958 |
| mapped_30-s0 | legacy-easy-seed-001 | 16 | True | 0.1964027 | 32.5143 | 8.872937 |
| mapped_30-s0 | legacy-easy-seed-001 | 50 | True | 0.2372635 | 63.27023 | 7.579796 |
| mapped_30-s0 | legacy-easy-seed-002 | 16 | True | 0.2850149 | 127.6635 | 17.3823 |
| mapped_30-s0 | legacy-easy-seed-002 | 50 | True | 0.3269069 | 193.2685 | 19.88605 |
| mapped_30-s0 | legacy-easy-seed-003 | 16 | True | 0.2206329 | 50.91917 | 9.904939 |
| mapped_30-s0 | legacy-easy-seed-003 | 50 | True | 0.2531901 | 80.39006 | 8.544696 |
| mapped_30-s0 | legacy-easy-seed-004 | 16 | False | 0.1950713 | 47.93307 | 25.10848 |
| mapped_30-s0 | legacy-easy-seed-004 | 50 | False | 0.2315789 | 75.77518 | 25.35325 |
| mapped_30-s0 | legacy-easy-seed-005 | 16 | True | 0.2221758 | 53.86328 | 11.58169 |
| mapped_30-s0 | legacy-easy-seed-005 | 50 | True | 0.2600768 | 90.21335 | 10.74629 |
| mapped_30-s0 | legacy-easy-seed-006 | 16 | True | 0.173708 | 16.90469 | 5.222273 |
| mapped_30-s0 | legacy-easy-seed-006 | 50 | True | 0.2014807 | 29.94703 | 3.058659 |
| mapped_30-s0 | legacy-easy-seed-007 | 16 | True | 0.208147 | 40.2686 | 8.800501 |
| mapped_30-s0 | legacy-easy-seed-007 | 50 | True | 0.2340702 | 60.39694 | 7.698268 |
| mapped_30-s0 | legacy-easy-seed-008 | 16 | False | 0.1623398 | 33.28036 | 26.02009 |
| mapped_30-s0 | legacy-easy-seed-008 | 50 | False | 0.1853933 | 42.49464 | 25.17571 |
| mapped_30-s0 | legacy-easy-seed-009 | 16 | True | 0.2576438 | 89.19392 | 12.46341 |
| mapped_30-s0 | legacy-easy-seed-009 | 50 | True | 0.2914894 | 132.7825 | 13.67767 |
| mapped_30-s0 | legacy-easy-seed-010 | 16 | True | 0.2085189 | 39.11071 | 7.376552 |
| mapped_30-s0 | legacy-easy-seed-010 | 50 | True | 0.2374483 | 62.23851 | 6.372365 |
| mapped_30-s0 | legacy-easy-seed-011 | 16 | True | 0.1665335 | 16.08493 | 7.315197 |
| mapped_30-s0 | legacy-easy-seed-011 | 50 | True | 0.1937799 | 24.86479 | 2.818714 |
| mapped_30-s0 | ref-01-ground-pair | 16 | False | 0.08697776 | 36.33255 | 36.33255 |
| mapped_30-s0 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59434 | 35.59434 |
| mapped_30-s0 | ref-02-facade-pair-and-ground | 16 | False | 0.1657244 | 27.77965 | 19.31222 |
| mapped_30-s0 | ref-02-facade-pair-and-ground | 50 | False | 0.1935961 | 38.6046 | 16.66826 |
| mapped_30-s0 | ref-03-wide-gap | 16 | False | 0.1684142 | 27.43701 | 17.94408 |
| mapped_30-s0 | ref-03-wide-gap | 50 | False | 0.1943811 | 38.54551 | 16.13866 |
| mapped_30-s0 | ref-04-asymmetric-heights | 16 | False | 0.179377 | 32.29011 | 18.01129 |
| mapped_30-s0 | ref-04-asymmetric-heights | 50 | False | 0.2060228 | 47.0753 | 17.10564 |
| mapped_30-s0 | ref-06-minimal-smoke | 16 | False | 0.06402721 | 39.31152 | 39.31152 |
| mapped_30-s0 | ref-06-minimal-smoke | 50 | False | 0.07258064 | 37.85109 | 37.85109 |
| mapped_30-s1 | legacy-easy-seed-000 | 16 | True | 0.191296 | 28.24425 | 7.657601 |
| mapped_30-s1 | legacy-easy-seed-000 | 50 | True | 0.2279512 | 53.80644 | 6.609958 |
| mapped_30-s1 | legacy-easy-seed-001 | 16 | True | 0.1984418 | 33.7124 | 8.792292 |
| mapped_30-s1 | legacy-easy-seed-001 | 50 | True | 0.2372635 | 63.27023 | 7.579796 |
| mapped_30-s1 | legacy-easy-seed-002 | 16 | True | 0.2869266 | 130.2177 | 17.36658 |
| mapped_30-s1 | legacy-easy-seed-002 | 50 | True | 0.3269069 | 193.2685 | 19.88605 |
| mapped_30-s1 | legacy-easy-seed-003 | 16 | True | 0.2218577 | 51.91695 | 9.898229 |
| mapped_30-s1 | legacy-easy-seed-003 | 50 | True | 0.2531901 | 80.39006 | 8.544696 |
| mapped_30-s1 | legacy-easy-seed-004 | 16 | False | 0.1962665 | 48.72787 | 25.17073 |
| mapped_30-s1 | legacy-easy-seed-004 | 50 | False | 0.2315789 | 75.77631 | 25.35438 |
| mapped_30-s1 | legacy-easy-seed-005 | 16 | True | 0.2234604 | 54.97798 | 11.62657 |
| mapped_30-s1 | legacy-easy-seed-005 | 50 | True | 0.2600768 | 90.21427 | 10.74721 |
| mapped_30-s1 | legacy-easy-seed-006 | 16 | True | 0.1743068 | 17.05443 | 5.110064 |
| mapped_30-s1 | legacy-easy-seed-006 | 50 | True | 0.2014807 | 29.94703 | 3.058659 |
| mapped_30-s1 | legacy-easy-seed-007 | 16 | True | 0.2085914 | 40.63388 | 8.847691 |
| mapped_30-s1 | legacy-easy-seed-007 | 50 | True | 0.2340702 | 60.39694 | 7.698268 |
| mapped_30-s1 | legacy-easy-seed-008 | 16 | False | 0.1624816 | 33.28291 | 25.97394 |
| mapped_30-s1 | legacy-easy-seed-008 | 50 | False | 0.1853933 | 42.49786 | 25.17893 |
| mapped_30-s1 | legacy-easy-seed-009 | 16 | True | 0.2586198 | 90.34181 | 12.51919 |
| mapped_30-s1 | legacy-easy-seed-009 | 50 | True | 0.2914894 | 132.7825 | 13.67767 |
| mapped_30-s1 | legacy-easy-seed-010 | 16 | True | 0.2093689 | 39.75754 | 7.411005 |
| mapped_30-s1 | legacy-easy-seed-010 | 50 | True | 0.2374483 | 62.23851 | 6.372365 |
| mapped_30-s1 | legacy-easy-seed-011 | 16 | True | 0.1676713 | 16.45631 | 7.252483 |
| mapped_30-s1 | legacy-easy-seed-011 | 50 | True | 0.1937799 | 24.86479 | 2.818714 |
| mapped_30-s1 | ref-01-ground-pair | 16 | False | 0.08691613 | 36.33854 | 36.33854 |
| mapped_30-s1 | ref-01-ground-pair | 50 | False | 0.09761389 | 35.59698 | 35.59698 |
| mapped_30-s1 | ref-02-facade-pair-and-ground | 16 | False | 0.1665915 | 28.05217 | 19.26056 |
| mapped_30-s1 | ref-02-facade-pair-and-ground | 50 | False | 0.1935961 | 38.60432 | 16.66797 |
| mapped_30-s1 | ref-03-wide-gap | 16 | False | 0.1691865 | 27.70075 | 17.90256 |
| mapped_30-s1 | ref-03-wide-gap | 50 | False | 0.1942551 | 38.46233 | 16.13135 |
| mapped_30-s1 | ref-04-asymmetric-heights | 16 | False | 0.1804645 | 32.86253 | 18.05592 |
| mapped_30-s1 | ref-04-asymmetric-heights | 50 | False | 0.2060228 | 47.07685 | 17.10718 |
| mapped_30-s1 | ref-06-minimal-smoke | 16 | False | 0.06393798 | 39.32574 | 39.32574 |
| mapped_30-s1 | ref-06-minimal-smoke | 50 | False | 0.07258064 | 37.8546 | 37.8546 |
| mass_3-s0 | legacy-easy-seed-000 | 16 | True | 0.2107574 | 39.75912 | 6.399637 |
| mass_3-s0 | legacy-easy-seed-000 | 50 | True | 0.2279512 | 53.80644 | 6.609958 |
| mass_3-s0 | legacy-easy-seed-001 | 16 | True | 0.222371 | 49.38431 | 6.941032 |
| mass_3-s0 | legacy-easy-seed-001 | 50 | True | 0.2430859 | 69.73831 | 8.380272 |
| mass_3-s0 | legacy-easy-seed-002 | 16 | True | 0.3077922 | 160.5673 | 17.74027 |
| mass_3-s0 | legacy-easy-seed-002 | 50 | True | 0.3269069 | 193.2685 | 19.88605 |
| mass_3-s0 | legacy-easy-seed-003 | 16 | True | 0.2378083 | 65.04307 | 8.833985 |
| mass_3-s0 | legacy-easy-seed-003 | 50 | True | 0.2531901 | 80.39006 | 8.544696 |
| mass_3-s0 | legacy-easy-seed-004 | 16 | False | 0.2168833 | 59.7506 | 21.73575 |
| mass_3-s0 | legacy-easy-seed-004 | 50 | True | 0.2368421 | 65.47489 | 10.18397 |
| mass_3-s0 | legacy-easy-seed-005 | 16 | True | 0.2463573 | 74.50401 | 9.841042 |
| mass_3-s0 | legacy-easy-seed-005 | 50 | True | 0.2629558 | 93.23083 | 10.46352 |
| mass_3-s0 | legacy-easy-seed-006 | 16 | True | 0.1868163 | 21.73739 | 3.656465 |
| mass_3-s0 | legacy-easy-seed-006 | 50 | True | 0.2014807 | 29.94703 | 3.058659 |
| mass_3-s0 | legacy-easy-seed-007 | 16 | True | 0.2219238 | 50.39764 | 8.324346 |
| mass_3-s0 | legacy-easy-seed-007 | 50 | True | 0.2340702 | 60.39694 | 7.698268 |
| mass_3-s0 | legacy-easy-seed-008 | 16 | False | 0.1773265 | 32.1216 | 18.81197 |
| mass_3-s0 | legacy-easy-seed-008 | 50 | True | 0.1910112 | 27.8652 | 7.442686 |
| mass_3-s0 | legacy-easy-seed-009 | 16 | True | 0.2756061 | 110.3939 | 12.3302 |
| mass_3-s0 | legacy-easy-seed-009 | 50 | True | 0.2914894 | 132.7825 | 13.67767 |
| mass_3-s0 | legacy-easy-seed-010 | 16 | True | 0.2229996 | 49.68459 | 6.718456 |
| mass_3-s0 | legacy-easy-seed-010 | 50 | True | 0.2374483 | 62.23851 | 6.372365 |
| mass_3-s0 | legacy-easy-seed-011 | 16 | True | 0.1810915 | 18.28177 | 3.166492 |
| mass_3-s0 | legacy-easy-seed-011 | 50 | True | 0.1937799 | 24.86479 | 2.818714 |
| mass_3-s0 | ref-01-ground-pair | 16 | False | 0.1429801 | 40.10003 | 37.96129 |
| mass_3-s0 | ref-01-ground-pair | 50 | False | 0.4561582 | 546.5374 | 88.87797 |
| mass_3-s0 | ref-02-facade-pair-and-ground | 16 | False | 0.1992132 | 44.49717 | 19.0845 |
| mass_3-s0 | ref-02-facade-pair-and-ground | 50 | False | 0.3436417 | 240.8402 | 38.27702 |
| mass_3-s0 | ref-03-wide-gap | 16 | False | 0.1979963 | 43.02563 | 18.3878 |
| mass_3-s0 | ref-03-wide-gap | 50 | False | 0.3053224 | 171.1263 | 32.03153 |
| mass_3-s0 | ref-04-asymmetric-heights | 16 | False | 0.2083198 | 50.87374 | 19.28215 |
| mass_3-s0 | ref-04-asymmetric-heights | 50 | False | 0.3184202 | 193.5952 | 34.14429 |
| mass_3-s0 | ref-06-minimal-smoke | 16 | False | 0.1126303 | 39.45462 | 39.45462 |
| mass_3-s0 | ref-06-minimal-smoke | 50 | False | 0.3339843 | 244.9896 | 59.54298 |
| mass_3-s1 | legacy-easy-seed-000 | 16 | True | 0.2120478 | 40.66351 | 6.348695 |
| mass_3-s1 | legacy-easy-seed-000 | 50 | True | 0.2279512 | 53.80644 | 6.609958 |
| mass_3-s1 | legacy-easy-seed-001 | 16 | True | 0.2234112 | 50.16348 | 6.853257 |
| mass_3-s1 | legacy-easy-seed-001 | 50 | True | 0.2416303 | 68.09349 | 8.178102 |
| mass_3-s1 | legacy-easy-seed-002 | 16 | True | 0.3088869 | 162.4173 | 17.92025 |
| mass_3-s1 | legacy-easy-seed-002 | 50 | True | 0.3269069 | 193.2685 | 19.88605 |
| mass_3-s1 | legacy-easy-seed-003 | 16 | True | 0.2385547 | 65.73219 | 8.808519 |
| mass_3-s1 | legacy-easy-seed-003 | 50 | True | 0.2531901 | 80.39006 | 8.544696 |
| mass_3-s1 | legacy-easy-seed-004 | 16 | False | 0.216666 | 60.99895 | 23.1545 |
| mass_3-s1 | legacy-easy-seed-004 | 50 | False | 0.2347368 | 79.08556 | 25.76916 |
| mass_3-s1 | legacy-easy-seed-005 | 16 | True | 0.2463518 | 74.6047 | 9.947371 |
| mass_3-s1 | legacy-easy-seed-005 | 50 | True | 0.2629558 | 93.23509 | 10.46778 |
| mass_3-s1 | legacy-easy-seed-006 | 16 | True | 0.1871405 | 21.89067 | 3.633863 |
| mass_3-s1 | legacy-easy-seed-006 | 50 | True | 0.2014807 | 29.94703 | 3.058659 |
| mass_3-s1 | legacy-easy-seed-007 | 16 | True | 0.2222327 | 50.70332 | 8.374618 |
| mass_3-s1 | legacy-easy-seed-007 | 50 | True | 0.2340702 | 60.39694 | 7.698268 |
| mass_3-s1 | legacy-easy-seed-008 | 16 | False | 0.1755542 | 33.09811 | 20.59874 |
| mass_3-s1 | legacy-easy-seed-008 | 50 | True | 0.1891386 | 28.19873 | 8.839151 |
| mass_3-s1 | legacy-easy-seed-009 | 16 | True | 0.2762162 | 111.195 | 12.36083 |
| mass_3-s1 | legacy-easy-seed-009 | 50 | True | 0.2914894 | 132.7825 | 13.67767 |
| mass_3-s1 | legacy-easy-seed-010 | 16 | True | 0.2235714 | 50.16787 | 6.723406 |
| mass_3-s1 | legacy-easy-seed-010 | 50 | True | 0.2374483 | 62.23851 | 6.372365 |
| mass_3-s1 | legacy-easy-seed-011 | 16 | True | 0.1817129 | 18.55538 | 3.13101 |
| mass_3-s1 | legacy-easy-seed-011 | 50 | True | 0.1937799 | 24.86479 | 2.818714 |
| mass_3-s1 | ref-01-ground-pair | 16 | False | 0.1207294 | 36.65306 | 36.6509 |
| mass_3-s1 | ref-01-ground-pair | 50 | False | 0.4222696 | 449.2254 | 79.18938 |
| mass_3-s1 | ref-02-facade-pair-and-ground | 16 | False | 0.1909464 | 38.51488 | 18.12962 |
| mass_3-s1 | ref-02-facade-pair-and-ground | 50 | False | 0.3311621 | 216.3311 | 35.7439 |
| mass_3-s1 | ref-03-wide-gap | 16 | False | 0.1911871 | 38.06612 | 17.54231 |
| mass_3-s1 | ref-03-wide-gap | 50 | False | 0.29652 | 156.6891 | 30.49387 |
| mass_3-s1 | ref-04-asymmetric-heights | 16 | False | 0.2022708 | 45.80744 | 18.39509 |
| mass_3-s1 | ref-04-asymmetric-heights | 50 | False | 0.3087684 | 176.7035 | 32.38779 |
| mass_3-s1 | ref-06-minimal-smoke | 16 | False | 0.09319687 | 38.66436 | 38.66436 |
| mass_3-s1 | ref-06-minimal-smoke | 50 | False | 0.3032693 | 190.0701 | 54.04012 |
| original_checkpoint | legacy-easy-seed-000 | 16 | True | 0.2108683 | 40.23169 | 6.790633 |
| original_checkpoint | legacy-easy-seed-000 | 50 | True | 0.2279512 | 53.80644 | 6.609958 |
| original_checkpoint | legacy-easy-seed-001 | 16 | True | 0.2222147 | 49.72994 | 7.416137 |
| original_checkpoint | legacy-easy-seed-001 | 50 | True | 0.2423704 | 68.92757 | 8.28075 |
| original_checkpoint | legacy-easy-seed-002 | 16 | True | 0.3079425 | 160.9992 | 17.94353 |
| original_checkpoint | legacy-easy-seed-002 | 50 | True | 0.3269069 | 193.2685 | 19.88605 |
| original_checkpoint | legacy-easy-seed-003 | 16 | True | 0.2377275 | 65.18065 | 9.048592 |
| original_checkpoint | legacy-easy-seed-003 | 50 | True | 0.2531901 | 80.39006 | 8.544696 |
| original_checkpoint | legacy-easy-seed-004 | 16 | False | 0.2163014 | 61.77356 | 24.21402 |
| original_checkpoint | legacy-easy-seed-004 | 50 | False | 0.2347368 | 79.08971 | 25.77331 |
| original_checkpoint | legacy-easy-seed-005 | 16 | True | 0.2457957 | 74.27644 | 10.18703 |
| original_checkpoint | legacy-easy-seed-005 | 50 | True | 0.2629558 | 93.23434 | 10.46704 |
| original_checkpoint | legacy-easy-seed-006 | 16 | True | 0.1865663 | 21.74836 | 3.802535 |
| original_checkpoint | legacy-easy-seed-006 | 50 | True | 0.2014807 | 29.94703 | 3.058659 |
| original_checkpoint | legacy-easy-seed-007 | 16 | True | 0.221525 | 50.33429 | 8.589586 |
| original_checkpoint | legacy-easy-seed-007 | 50 | True | 0.2340702 | 60.39694 | 7.698268 |
| original_checkpoint | legacy-easy-seed-008 | 16 | False | 0.1750936 | 35.46016 | 23.16719 |
| original_checkpoint | legacy-easy-seed-008 | 50 | False | 0.1872659 | 43.65197 | 25.32692 |
| original_checkpoint | legacy-easy-seed-009 | 16 | True | 0.275364 | 110.2476 | 12.48888 |
| original_checkpoint | legacy-easy-seed-009 | 50 | True | 0.2914894 | 132.7825 | 13.67767 |
| original_checkpoint | legacy-easy-seed-010 | 16 | True | 0.2227298 | 49.70519 | 6.963888 |
| original_checkpoint | legacy-easy-seed-010 | 50 | True | 0.2374483 | 62.23851 | 6.372365 |
| original_checkpoint | legacy-easy-seed-011 | 16 | True | 0.180861 | 18.48049 | 3.479058 |
| original_checkpoint | legacy-easy-seed-011 | 50 | True | 0.1937799 | 24.86479 | 2.818714 |
| original_checkpoint | ref-01-ground-pair | 16 | False | 0.1608797 | 46.75688 | 39.98872 |
| original_checkpoint | ref-01-ground-pair | 50 | False | 0.4902727 | 655.2009 | 99.93824 |
| original_checkpoint | ref-02-facade-pair-and-ground | 16 | False | 0.2075492 | 51.5339 | 20.49119 |
| original_checkpoint | ref-02-facade-pair-and-ground | 50 | False | 0.3591593 | 273.3871 | 41.73859 |
| original_checkpoint | ref-03-wide-gap | 16 | False | 0.203793 | 47.89162 | 19.4555 |
| original_checkpoint | ref-03-wide-gap | 50 | False | 0.3200928 | 197.0329 | 34.88252 |
| original_checkpoint | ref-04-asymmetric-heights | 16 | False | 0.2150541 | 56.99296 | 20.4001 |
| original_checkpoint | ref-04-asymmetric-heights | 50 | False | 0.3263312 | 208.1329 | 35.71408 |
| original_checkpoint | ref-06-minimal-smoke | 16 | False | 0.1242102 | 41.04913 | 40.97734 |
| original_checkpoint | ref-06-minimal-smoke | 50 | False | 0.3716692 | 324.3116 | 67.79505 |
| W1_procedural | legacy-easy-seed-000 | static | True | 0.0312076 | 0.002636325 | 0.002636325 |
| W1_procedural | legacy-easy-seed-001 | static | True | 0.03347889 | 0.002788913 | 0.002788913 |
| W1_procedural | legacy-easy-seed-002 | static | True | 0.03185247 | 0.00638408 | 0.00638408 |
| W1_procedural | legacy-easy-seed-003 | static | True | 0.03022163 | 0.007682554 | 0.007682554 |
| W1_procedural | legacy-easy-seed-004 | static | True | 0.03368421 | 0.005180113 | 0.005180113 |
| W1_procedural | legacy-easy-seed-005 | static | True | 0.03454895 | 0.007362612 | 0.007362612 |
| W1_procedural | legacy-easy-seed-006 | static | True | 0.03014278 | 0.01210859 | 0.01210859 |
| W1_procedural | legacy-easy-seed-007 | static | True | 0.05201561 | 0.006168488 | 0.006168488 |
| W1_procedural | legacy-easy-seed-008 | static | True | 0.06367041 | 0.005291354 | 0.005291354 |
| W1_procedural | legacy-easy-seed-009 | static | True | 0.03049645 | 0.007597893 | 0.007597893 |
| W1_procedural | legacy-easy-seed-010 | static | True | 0.03012404 | 0.009979248 | 0.009979248 |
| W1_procedural | legacy-easy-seed-011 | static | True | 0.03030303 | 0.007372456 | 0.007372456 |
| W1_procedural | ref-01-ground-pair | static | True | 0.03904555 | 0.004327589 | 0.004327589 |
| W1_procedural | ref-02-facade-pair-and-ground | static | True | 0.03004926 | 0.008778233 | 0.008778233 |
| W1_procedural | ref-03-wide-gap | static | True | 0.030185 | 0.008458292 | 0.008458292 |
| W1_procedural | ref-04-asymmetric-heights | static | True | 0.03011384 | 0.01235174 | 0.01235174 |
| W1_procedural | ref-06-minimal-smoke | static | True | 0.04301075 | 0.003213205 | 0.003213205 |

No control was rerun or selected by appearance. These are existing development scenes; direct per-scene fitting does not establish learned generalization, physical usability or safety.

## Verification and limits

{
  "saved_field_projections_verified": 1088,
  "checkpoint_boundaries_verified": 1088,
  "initial_final_scores_recomputed": 68,
  "gradient_norms_verified": 1496,
  "controls_reused": 187,
  "scope": "Intermediate projections/checkpoints/norms verified; full objective recomputation at initial/final states."
}

Every update has a raw/projected field, pre-update gradient, objective trace and complete optimizer checkpoint. Field/checkpoint records are AFTER updates; traces describe BEFORE updates. All initial/final objectives and binary metrics were recomputed. Intermediate objectives were not all independently recomputed. No production checkpoint/default, paid compute or cloud operation.
