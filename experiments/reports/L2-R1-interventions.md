# L2 material intervention and R1 recovery evidence

L2 `20260923T075113Z_d95fabaf3776`; R1 `20260923T075727Z_2233c0e51b9a`.

All registered artifacts verified. Saved gradients independently reproduce recorded norms; all 21 hard-forward pairs and frozen fields match exactly. These are local diagnostics, not trained-model quality benchmarks.

## Explicit budget comparison

Counts include 17 feasible scenes and one intentionally sealed reference. Compatibility checks only envelope capacity against minimum mass and full-guide mass against maximum mass; they do not establish simultaneous feasibility of all nine objectives.

| Region | Denominator | Necessary-valid contexts |
|---|---|---:|
| scaffold | site | 0/18 |
| scaffold | envelope | 8/18 |
| radius3 | site | 3/18 |
| radius3 | envelope | 17/18 |
| radius6 | site | 12/18 |
| radius6 | envelope | 17/18 |

Both fractions remain 3%-12%. Changing the denominator changes the physical allowance. Radius-six comparison below uses equivalent fully occupied voxel counts; fractional density is a proxy, not a constructed material specification. Per-scene cubic-metre values are also preserved in raw records.

| Scene | Site voxels | Envelope voxels | Site minimum / maximum | Envelope minimum / maximum | Envelope / site |
|---|---:|---:|---:|---:|---:|
| legacy-easy-seed-000 | 27776 | 737 | 833.280 / 3333.120 | 22.110 / 88.440 | 2.65% |
| legacy-easy-seed-001 | 26972 | 687 | 809.160 / 3236.640 | 20.610 / 82.440 | 2.55% |
| legacy-easy-seed-002 | 26704 | 1193 | 801.120 / 3204.480 | 35.790 / 143.160 | 4.47% |
| legacy-easy-seed-003 | 25478 | 1489 | 764.340 / 3057.360 | 44.670 / 178.680 | 5.84% |
| legacy-easy-seed-004 | 27413 | 950 | 822.390 / 3289.560 | 28.500 / 114.000 | 3.47% |
| legacy-easy-seed-005 | 27952 | 1042 | 838.560 / 3354.240 | 31.260 / 125.040 | 3.73% |
| legacy-easy-seed-006 | 28400 | 1891 | 852.000 / 3408.000 | 56.730 / 226.920 | 6.66% |
| legacy-easy-seed-007 | 27056 | 769 | 811.680 / 3246.720 | 23.070 / 92.280 | 2.84% |
| legacy-easy-seed-008 | 26153 | 534 | 784.590 / 3138.360 | 16.020 / 64.080 | 2.04% |
| legacy-easy-seed-009 | 27776 | 1410 | 833.280 / 3333.120 | 42.300 / 169.200 | 5.08% |
| legacy-easy-seed-010 | 28246 | 1693 | 847.380 / 3389.520 | 50.790 / 203.160 | 5.99% |
| legacy-easy-seed-011 | 25560 | 1254 | 766.800 / 3067.200 | 37.620 / 150.480 | 4.91% |
| ref-01-ground-pair | 24768 | 922 | 743.040 / 2972.160 | 27.660 / 110.640 | 3.72% |
| ref-02-facade-pair-and-ground | 24768 | 2030 | 743.040 / 2972.160 | 60.900 / 243.600 | 8.20% |
| ref-03-wide-gap | 25568 | 2054 | 767.040 / 3068.160 | 61.620 / 246.480 | 8.03% |
| ref-04-asymmetric-heights | 25568 | 2723 | 767.040 / 3068.160 | 81.690 / 326.760 | 10.65% |
| ref-05-sealed-partition | 12288 | 836 | 368.640 / 1474.560 | 25.080 / 100.320 | 6.80% |
| ref-06-minimal-smoke | 29408 | 744 | 882.240 / 3528.960 | 22.320 / 89.280 | 2.53% |

Across all scenes the radius-six allowance becomes 2.04%-10.65% of the site allowance. This is a substantive objective change, selected only for the recovery test.

## Material gradient comparison

Each ordinary arm has 18 cases (three scenes, three seeds, two horizons). Each absent-scaffold arm has three cases (seed zero, four steps). Binary connectivity uses independent six-neighbor traversal at material threshold 0.5. Nonzero derivative counts alone do not demonstrate an effective training objective.

| Scaffold | Arm | Cases | Coverage weight derivative nonzero | Access weight derivative nonzero | Binary connected |
|---|---|---:|---:|---:|---:|
| present | hard_projected | 18 | 18 | 4 | 3 |
| present | hard_preclamp | 18 | 18 | 4 | 3 |
| present | smooth_projected | 18 | 18 | 18 | 3 |
| absent | hard_projected | 3 | 1 | 0 | 0 |
| absent | hard_preclamp | 3 | 3 | 0 | 0 |
| absent | smooth_projected | 3 | 3 | 3 | 0 |

The known ground case (seed zero, four steps) retains raw material -0.001279894 and -0.009346317 at the two failed cells under both hard arms. Their coverage derivatives change from zero to -1/36 with pre-clamp guidance. Its projected access weight derivative is still zero. This changes coverage guidance; it does not repair the access derivative.

Smooth clipping at beta=20 assigns about 0.03466 material at raw zero, before recurrent effects. Its soft access improvement can therefore reflect diffuse background. No straight-through estimator was used.

### Individual gradient and geometry cases

| Scene | Seed | Steps | No scaffold | Arm | Coverage weight norm | Access weight norm | Soft mass | Binary voxels | Connected |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| legacy-easy-seed-000 | 0 | 4 | False | hard_projected | 1.0837207 | 0.90040152 | 45.40374 | 0 | False |
| legacy-easy-seed-000 | 0 | 4 | False | hard_preclamp | 1.0837207 | 0.90040152 | 45.40374 | 0 | False |
| legacy-easy-seed-000 | 0 | 4 | False | smooth_projected | 1.056591 | 0.88450754 | 2264.02539 | 0 | False |
| legacy-easy-seed-000 | 0 | 16 | False | hard_projected | 1.8622848 | 0 | 153.05510 | 165 | True |
| legacy-easy-seed-000 | 0 | 16 | False | hard_preclamp | 1.8622848 | 0 | 153.05510 | 165 | True |
| legacy-easy-seed-000 | 0 | 16 | False | smooth_projected | 1.562942 | 1.2887703 | 14264.90332 | 16688 | True |
| legacy-easy-seed-000 | 1 | 4 | False | hard_projected | 0.9804541 | 0.41530579 | 44.01843 | 0 | False |
| legacy-easy-seed-000 | 1 | 4 | False | hard_preclamp | 0.9804541 | 0.41530579 | 44.01843 | 0 | False |
| legacy-easy-seed-000 | 1 | 4 | False | smooth_projected | 0.9550801 | 0.38293918 | 2266.08496 | 0 | False |
| legacy-easy-seed-000 | 1 | 16 | False | hard_projected | 1.9384458 | 5.3215208 | 155.63277 | 166 | True |
| legacy-easy-seed-000 | 1 | 16 | False | hard_preclamp | 1.9384458 | 5.3215208 | 155.63277 | 166 | True |
| legacy-easy-seed-000 | 1 | 16 | False | smooth_projected | 1.6087251 | 4.643712 | 14282.58887 | 16818 | True |
| legacy-easy-seed-000 | 2 | 4 | False | hard_projected | 1.0509379 | 0.87388641 | 45.60869 | 0 | False |
| legacy-easy-seed-000 | 2 | 4 | False | hard_preclamp | 1.0509379 | 0.87388641 | 45.60869 | 0 | False |
| legacy-easy-seed-000 | 2 | 4 | False | smooth_projected | 1.0226407 | 0.84793611 | 2265.72778 | 0 | False |
| legacy-easy-seed-000 | 2 | 16 | False | hard_projected | 1.3645091 | 0 | 155.40994 | 165 | True |
| legacy-easy-seed-000 | 2 | 16 | False | hard_preclamp | 1.3645091 | 0 | 155.40994 | 165 | True |
| legacy-easy-seed-000 | 2 | 16 | False | smooth_projected | 1.3234074 | 0.024070189 | 14300.43555 | 16757 | True |
| ref-01-ground-pair | 0 | 4 | False | hard_projected | 0.70008162 | 0 | 51.19946 | 0 | False |
| ref-01-ground-pair | 0 | 4 | False | hard_preclamp | 0.8154003 | 0 | 51.19946 | 0 | False |
| ref-01-ground-pair | 0 | 4 | False | smooth_projected | 0.62813819 | 0.42559931 | 2002.80383 | 0 | False |
| ref-01-ground-pair | 0 | 16 | False | hard_projected | 1.9389546 | 0 | 152.71753 | 92 | False |
| ref-01-ground-pair | 0 | 16 | False | hard_preclamp | 2.0984913 | 0 | 152.71753 | 92 | False |
| ref-01-ground-pair | 0 | 16 | False | smooth_projected | 1.8183237 | 0.16729519 | 12395.76562 | 14338 | False |
| ref-01-ground-pair | 1 | 4 | False | hard_projected | 0.59735395 | 0 | 49.11320 | 0 | False |
| ref-01-ground-pair | 1 | 4 | False | hard_preclamp | 0.75809648 | 0 | 49.11320 | 0 | False |
| ref-01-ground-pair | 1 | 4 | False | smooth_projected | 0.57223166 | 0.56850253 | 2003.07581 | 0 | False |
| ref-01-ground-pair | 1 | 16 | False | hard_projected | 2.4032674 | 0 | 149.79961 | 89 | False |
| ref-01-ground-pair | 1 | 16 | False | hard_preclamp | 2.5585922 | 0 | 149.79961 | 89 | False |
| ref-01-ground-pair | 1 | 16 | False | smooth_projected | 2.2326053 | 0.16566458 | 12412.03125 | 14405 | False |
| ref-01-ground-pair | 2 | 4 | False | hard_projected | 0.61031651 | 0 | 50.77608 | 0 | False |
| ref-01-ground-pair | 2 | 4 | False | hard_preclamp | 0.79574847 | 0 | 50.77608 | 0 | False |
| ref-01-ground-pair | 2 | 4 | False | smooth_projected | 0.57310137 | 0.57349104 | 2002.92456 | 0 | False |
| ref-01-ground-pair | 2 | 16 | False | hard_projected | 1.9692421 | 0 | 148.33109 | 89 | False |
| ref-01-ground-pair | 2 | 16 | False | hard_preclamp | 2.1595212 | 0 | 148.33109 | 89 | False |
| ref-01-ground-pair | 2 | 16 | False | smooth_projected | 2.006734 | 0.14913698 | 12413.68945 | 14372 | False |
| ref-06-minimal-smoke | 0 | 4 | False | hard_projected | 0.69581472 | 0 | 37.72796 | 0 | False |
| ref-06-minimal-smoke | 0 | 4 | False | hard_preclamp | 0.79429934 | 0 | 37.72796 | 0 | False |
| ref-06-minimal-smoke | 0 | 4 | False | smooth_projected | 0.58840046 | 0.61338047 | 2395.09521 | 0 | False |
| ref-06-minimal-smoke | 0 | 16 | False | hard_projected | 2.3040543 | 0 | 90.68182 | 54 | False |
| ref-06-minimal-smoke | 0 | 16 | False | hard_preclamp | 2.3548111 | 0 | 90.68182 | 54 | False |
| ref-06-minimal-smoke | 0 | 16 | False | smooth_projected | 2.105632 | 1.0518579 | 15034.48340 | 17619 | False |
| ref-06-minimal-smoke | 1 | 4 | False | hard_projected | 0.71229319 | 0 | 37.34471 | 0 | False |
| ref-06-minimal-smoke | 1 | 4 | False | hard_preclamp | 0.78724517 | 0 | 37.34471 | 0 | False |
| ref-06-minimal-smoke | 1 | 4 | False | smooth_projected | 0.5810757 | 0.43697487 | 2398.06055 | 0 | False |
| ref-06-minimal-smoke | 1 | 16 | False | hard_projected | 2.3268402 | 0 | 88.74525 | 53 | False |
| ref-06-minimal-smoke | 1 | 16 | False | hard_preclamp | 2.4959603 | 0 | 88.74525 | 53 | False |
| ref-06-minimal-smoke | 1 | 16 | False | smooth_projected | 2.3798187 | 0.14955343 | 15058.50488 | 17678 | False |
| ref-06-minimal-smoke | 2 | 4 | False | hard_projected | 0.60719373 | 0 | 37.04744 | 0 | False |
| ref-06-minimal-smoke | 2 | 4 | False | hard_preclamp | 0.78919415 | 0 | 37.04744 | 0 | False |
| ref-06-minimal-smoke | 2 | 4 | False | smooth_projected | 0.57293746 | 0.61329666 | 2395.51709 | 0 | False |
| ref-06-minimal-smoke | 2 | 16 | False | hard_projected | 2.5829344 | 0 | 92.41242 | 55 | False |
| ref-06-minimal-smoke | 2 | 16 | False | hard_preclamp | 2.6720489 | 0 | 92.41242 | 55 | False |
| ref-06-minimal-smoke | 2 | 16 | False | smooth_projected | 2.5667731 | 0.17083925 | 15058.97363 | 17690 | False |
| legacy-easy-seed-000 | 0 | 4 | True | hard_projected | 0.81591931 | 0 | 0.80134 | 0 | False |
| legacy-easy-seed-000 | 0 | 4 | True | hard_preclamp | 0.8962264 | 0 | 0.80134 | 0 | False |
| legacy-easy-seed-000 | 0 | 4 | True | smooth_projected | 0.69727342 | 0.66032239 | 2233.65674 | 0 | False |
| ref-01-ground-pair | 0 | 4 | True | hard_projected | 0 | 0 | 7.34721 | 0 | False |
| ref-01-ground-pair | 0 | 4 | True | hard_preclamp | 0.17664289 | 0 | 7.34721 | 0 | False |
| ref-01-ground-pair | 0 | 4 | True | smooth_projected | 0.25881816 | 0.2693145 | 1972.28394 | 0 | False |
| ref-06-minimal-smoke | 0 | 4 | True | hard_projected | 0 | 0 | 3.88454 | 0 | False |
| ref-06-minimal-smoke | 0 | 4 | True | hard_preclamp | 0.1868363 | 0 | 3.88454 | 0 | False |
| ref-06-minimal-smoke | 0 | 4 | True | smooth_projected | 0.203377 | 0.18258984 | 2371.23096 | 0 | False |

## Separate-process optimizer recovery

Four logical updates; ten executed updates across uninterrupted, prefix, resumed and repeated-resume branches. All seven comparisons pass and are independently rechecked here: full checkpoint trees, traces and field arrays, plus the update-two boundary. CPU Adam, StepLR, model buffers and Python/NumPy/global PyTorch/explicit firing RNG are saved.

| Update | Sampled scene | Steps | Loss | Gradient norm before clipping | Learning rate after step |
|---:|---|---:|---:|---:|---:|
| 1 | ref-06-minimal-smoke | 2 | 2.12298727 | 3.10876298 | 0.00010000 |
| 2 | ref-01-ground-pair | 3 | 2.21099234 | 5.91645241 | 0.00009000 |
| 3 | ref-06-minimal-smoke | 2 | 2.10107470 | 2.96014452 | 0.00009000 |
| 4 | ref-06-minimal-smoke | 3 | 2.14474344 | 2.36361408 | 0.00008100 |

The three-scene sampling pool happened to draw only two scenes in four updates; this does not test legacy-scene optimization. All nine terms used unit weights strictly for recovery mechanics. Loss values from differing scenes/horizons are not a learning curve.

Recovery is tested at a completed-update boundary with orderly process exit. It is not a test of CUDA determinism, Colab disconnection, mid-write power loss, or mid-backward continuation. No paid compute, cloud access or deployment occurred.
