# A1 facade endpoint allowance comparison

Run `20260923T084341Z_e37699e31f26`; source `2d543de6fe0767cc1b01d9f54916785b9f741eaa`.

864 matched target-arm records,144 control-arm records,144 bound-arm records and72 gradient-arm records. All hashes verified. Scene-derived allowances, raw facade values, other-eight-term equality, paired mass/metrics, joint bounds and independent analytical quotient gradients rechecked. No optimizer updates.

## Necessary compatibility on 17 feasible scenes

| Envelope radius | Budget | Original | Endpoint allowance |
|---:|---|---:|---:|
| 3 | site | 3/17 | 3/17 |
| 3 | envelope | 11/17 | 15/17 |
| 6 | site | 12/17 | 12/17 |
| 6 | envelope | 15/17 | 17/17 |

The sealed reference is retained and route-invalid. Passing is necessary, not proof that all losses admit an acceptable common solution. Budgets/regions/fractions did not change.

### Previously conflicting radius-six scenes

| Scene | Arm | Charged mandatory contact | Joint minimum mass | Maximum mass |
|---|---|---:|---:|---:|
| legacy-easy-seed-007 | original | 14 | 93.3333 | 92.2800 |
| legacy-easy-seed-007 | endpoint_allowance | 6 | 40.0000 | 92.2800 |
| legacy-easy-seed-008 | original | 13 | 86.6667 | 64.0800 |
| legacy-easy-seed-008 | endpoint_allowance | 5 | 33.3333 | 64.0800 |

## Controls

| Control | Arm | Cases | Facade penalty range |
|---|---|---:|---:|
| empty | original | 18 | 0 - 0 |
| empty | endpoint_allowance | 18 | 0 - 0 |
| allowance_only | original | 18 | 0 - 0.85 |
| allowance_only | endpoint_allowance | 18 | 0 - 0 |
| facade_blanket | original | 18 | 0.85 - 0.85 |
| facade_blanket | endpoint_allowance | 18 | 0.841361 - 0.85 |
| guide_and_blanket | original | 18 | 0.815595 - 0.847602 |
| guide_and_blanket | endpoint_allowance | 18 | 0.811083 - 0.845204 |

Allowance-only has zero facade penalty by definition; it is not a successful architecture. Ground-only reference scenes have no facade allowance. Empty and attachment-only fields still expose failures through other objectives. Facade blankets remain penalized on all18 scenes.

## Static candidates: zero-term witnesses at radius six / envelope budget

| Candidate | Original | Endpoint allowance |
|---|---:|---:|
| empty | 0/17 | 0/17 |
| guide | 2/17 | 6/17 |
| scaffold | 0/17 | 0/17 |
| radius1 | 1/17 | 5/17 |
| radius3 | 0/17 | 0/17 |
| radius6 | 0/17 | 0/17 |

Simple-route zero-loss examples remain. This intervention fixes one accounting conflict; it does not create minimum thickness, require attachment or demonstrate learned value. The denominator still permits facade-ratio dilution by unrelated material.

## Exact allowance patches

| Scene | Allowed cells | Named patches |
|---|---:|---:|
| legacy-easy-seed-000 | 8 | 2 |
| legacy-easy-seed-001 | 8 | 2 |
| legacy-easy-seed-002 | 8 | 2 |
| legacy-easy-seed-003 | 8 | 2 |
| legacy-easy-seed-004 | 8 | 2 |
| legacy-easy-seed-005 | 8 | 2 |
| legacy-easy-seed-006 | 8 | 2 |
| legacy-easy-seed-007 | 8 | 2 |
| legacy-easy-seed-008 | 8 | 2 |
| legacy-easy-seed-009 | 8 | 2 |
| legacy-easy-seed-010 | 8 | 2 |
| legacy-easy-seed-011 | 6 | 2 |
| ref-01-ground-pair | 0 | 0 |
| ref-02-facade-pair-and-ground | 8 | 2 |
| ref-03-wide-gap | 8 | 2 |
| ref-04-asymmetric-heights | 8 | 2 |
| ref-05-sealed-partition | 8 | 2 |
| ref-06-minimal-smoke | 0 | 0 |

Allowances come from facade-typed entrance geometry intersected with direct building-face neighbors and permitted space. No dilation or target-derived expansion. At0.8m per voxel an entrance block is1.6m per axis; each four-cell face patch is2.56m2. These are geometric patches, not engineered connection specifications.

## Probe facade gradients

| Scene | Budget | Arm | Value | Legal gradient norm |
|---|---|---|---:|---:|
| legacy-easy-seed-000 | site | original | 0.11162663 | 1.157153 |
| legacy-easy-seed-000 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-000 | envelope | original | 0.11162663 | 1.157153 |
| legacy-easy-seed-000 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-001 | site | original | 0.13103089 | 1.2578263 |
| legacy-easy-seed-001 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-001 | envelope | original | 0.13103089 | 1.2578263 |
| legacy-easy-seed-001 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-002 | site | original | 0.046088174 | 0.63190316 |
| legacy-easy-seed-002 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-002 | envelope | original | 0.046088174 | 0.63190316 |
| legacy-easy-seed-002 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-003 | site | original | 0.037065059 | 0.55661782 |
| legacy-easy-seed-003 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-003 | envelope | original | 0.037065059 | 0.55661782 |
| legacy-easy-seed-003 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-004 | site | original | 0.12052581 | 0.884741 |
| legacy-easy-seed-004 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-004 | envelope | original | 0.12052581 | 0.884741 |
| legacy-easy-seed-004 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-005 | site | original | 0.041710392 | 0.6549567 |
| legacy-easy-seed-005 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-005 | envelope | original | 0.041710392 | 0.6549567 |
| legacy-easy-seed-005 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-006 | site | original | 0.009970963 | 0.43398603 |
| legacy-easy-seed-006 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-006 | envelope | original | 0.009970963 | 0.43398603 |
| legacy-easy-seed-006 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-007 | site | original | 0.24514663 | 1.5379997 |
| legacy-easy-seed-007 | site | endpoint_allowance | 0.079813778 | 1.0514939 |
| legacy-easy-seed-007 | envelope | original | 0.24514663 | 1.5379997 |
| legacy-easy-seed-007 | envelope | endpoint_allowance | 0.079813778 | 1.0514939 |
| legacy-easy-seed-008 | site | original | 0.33955365 | 2.5405537 |
| legacy-easy-seed-008 | site | endpoint_allowance | 0.1168994 | 1.5982752 |
| legacy-easy-seed-008 | envelope | original | 0.33955365 | 2.5405537 |
| legacy-easy-seed-008 | envelope | endpoint_allowance | 0.1168994 | 1.5982752 |
| legacy-easy-seed-009 | site | original | 0.031336576 | 0.53374119 |
| legacy-easy-seed-009 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-009 | envelope | original | 0.031336576 | 0.53374119 |
| legacy-easy-seed-009 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-010 | site | original | 0.015085116 | 0.45726528 |
| legacy-easy-seed-010 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-010 | envelope | original | 0.015085116 | 0.45726528 |
| legacy-easy-seed-010 | envelope | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-011 | site | original | 0.022824109 | 0.63907301 |
| legacy-easy-seed-011 | site | endpoint_allowance | 0 | 0 |
| legacy-easy-seed-011 | envelope | original | 0.022824109 | 0.63907301 |
| legacy-easy-seed-011 | envelope | endpoint_allowance | 0 | 0 |
| ref-01-ground-pair | site | original | 0 | 0 |
| ref-01-ground-pair | site | endpoint_allowance | 0 | 0 |
| ref-01-ground-pair | envelope | original | 0 | 0 |
| ref-01-ground-pair | envelope | endpoint_allowance | 0 | 0 |
| ref-02-facade-pair-and-ground | site | original | 0 | 0 |
| ref-02-facade-pair-and-ground | site | endpoint_allowance | 0 | 0 |
| ref-02-facade-pair-and-ground | envelope | original | 0 | 0 |
| ref-02-facade-pair-and-ground | envelope | endpoint_allowance | 0 | 0 |
| ref-03-wide-gap | site | original | 0 | 0 |
| ref-03-wide-gap | site | endpoint_allowance | 0 | 0 |
| ref-03-wide-gap | envelope | original | 0 | 0 |
| ref-03-wide-gap | envelope | endpoint_allowance | 0 | 0 |
| ref-04-asymmetric-heights | site | original | 0 | 0 |
| ref-04-asymmetric-heights | site | endpoint_allowance | 0 | 0 |
| ref-04-asymmetric-heights | envelope | original | 0 | 0 |
| ref-04-asymmetric-heights | envelope | endpoint_allowance | 0 | 0 |
| ref-05-sealed-partition | site | original | 0.22223482 | 1.2594647 |
| ref-05-sealed-partition | site | endpoint_allowance | 0.048015907 | 1.3045735 |
| ref-05-sealed-partition | envelope | original | 0.22223482 | 1.2594647 |
| ref-05-sealed-partition | envelope | endpoint_allowance | 0.048015907 | 1.3045735 |
| ref-06-minimal-smoke | site | original | 0 | 0 |
| ref-06-minimal-smoke | site | endpoint_allowance | 0 | 0 |
| ref-06-minimal-smoke | envelope | original | 0 | 0 |
| ref-06-minimal-smoke | envelope | endpoint_allowance | 0 | 0 |

These are fixed T1 occupancy probes, not model gradients or calibrated weights. A zero gradient can be correct when the capped ratio is below15%. All original probe values/norms reproduce T1.

## Full per-case facade comparison

| Scene | Candidate | Envelope | Budget | Original facade | Endpoint facade |
|---|---|---:|---|---:|---:|
| legacy-easy-seed-000 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-000 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-000 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-000 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-000 | guide | 3 | site | 0.21363637 | 0 |
| legacy-easy-seed-000 | guide | 3 | envelope | 0.21363637 | 0 |
| legacy-easy-seed-000 | guide | 6 | site | 0.21363637 | 0 |
| legacy-easy-seed-000 | guide | 6 | envelope | 0.21363637 | 0 |
| legacy-easy-seed-000 | scaffold | 3 | site | 0.18333334 | 0.13571429 |
| legacy-easy-seed-000 | scaffold | 3 | envelope | 0.18333334 | 0.13571429 |
| legacy-easy-seed-000 | scaffold | 6 | site | 0.18333334 | 0.13571429 |
| legacy-easy-seed-000 | scaffold | 6 | envelope | 0.18333334 | 0.13571429 |
| legacy-easy-seed-000 | radius1 | 3 | site | 0.20443037 | 0.10316455 |
| legacy-easy-seed-000 | radius1 | 3 | envelope | 0.20443037 | 0.10316455 |
| legacy-easy-seed-000 | radius1 | 6 | site | 0.20443037 | 0.10316455 |
| legacy-easy-seed-000 | radius1 | 6 | envelope | 0.20443037 | 0.10316455 |
| legacy-easy-seed-000 | radius3 | 3 | site | 0.13727272 | 0.1081818 |
| legacy-easy-seed-000 | radius3 | 3 | envelope | 0.13727272 | 0.1081818 |
| legacy-easy-seed-000 | radius3 | 6 | site | 0.13727272 | 0.1081818 |
| legacy-easy-seed-000 | radius3 | 6 | envelope | 0.13727272 | 0.1081818 |
| legacy-easy-seed-000 | radius6 | 3 | site | 0.048100397 | 0.037245587 |
| legacy-easy-seed-000 | radius6 | 3 | envelope | 0.048100397 | 0.037245587 |
| legacy-easy-seed-000 | radius6 | 6 | site | 0.048100397 | 0.037245587 |
| legacy-easy-seed-000 | radius6 | 6 | envelope | 0.048100397 | 0.037245587 |
| legacy-easy-seed-001 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-001 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-001 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-001 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-001 | guide | 3 | site | 0.19782609 | 0 |
| legacy-easy-seed-001 | guide | 3 | envelope | 0.19782609 | 0 |
| legacy-easy-seed-001 | guide | 6 | site | 0.19782609 | 0 |
| legacy-easy-seed-001 | guide | 6 | envelope | 0.19782609 | 0 |
| legacy-easy-seed-001 | scaffold | 3 | site | 0.21470588 | 0.16764706 |
| legacy-easy-seed-001 | scaffold | 3 | envelope | 0.21470588 | 0.16764706 |
| legacy-easy-seed-001 | scaffold | 6 | site | 0.21470588 | 0.16764706 |
| legacy-easy-seed-001 | scaffold | 6 | envelope | 0.21470588 | 0.16764706 |
| legacy-easy-seed-001 | radius1 | 3 | site | 0.20365852 | 0.10609755 |
| legacy-easy-seed-001 | radius1 | 3 | envelope | 0.20365852 | 0.10609755 |
| legacy-easy-seed-001 | radius1 | 6 | site | 0.20365852 | 0.10609755 |
| legacy-easy-seed-001 | radius1 | 6 | envelope | 0.20365852 | 0.10609755 |
| legacy-easy-seed-001 | radius3 | 3 | site | 0.15418249 | 0.12376425 |
| legacy-easy-seed-001 | radius3 | 3 | envelope | 0.15418249 | 0.12376425 |
| legacy-easy-seed-001 | radius3 | 6 | site | 0.15418249 | 0.12376425 |
| legacy-easy-seed-001 | radius3 | 6 | envelope | 0.15418249 | 0.12376425 |
| legacy-easy-seed-001 | radius6 | 3 | site | 0.0683406 | 0.056695774 |
| legacy-easy-seed-001 | radius6 | 3 | envelope | 0.0683406 | 0.056695774 |
| legacy-easy-seed-001 | radius6 | 6 | site | 0.0683406 | 0.056695774 |
| legacy-easy-seed-001 | radius6 | 6 | envelope | 0.0683406 | 0.056695774 |
| legacy-easy-seed-002 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-002 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-002 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-002 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-002 | guide | 3 | site | 0.060526311 | 0 |
| legacy-easy-seed-002 | guide | 3 | envelope | 0.060526311 | 0 |
| legacy-easy-seed-002 | guide | 6 | site | 0.060526311 | 0 |
| legacy-easy-seed-002 | guide | 6 | envelope | 0.060526311 | 0 |
| legacy-easy-seed-002 | scaffold | 3 | site | 0.080769226 | 0.060256407 |
| legacy-easy-seed-002 | scaffold | 3 | envelope | 0.080769226 | 0.060256407 |
| legacy-easy-seed-002 | scaffold | 6 | site | 0.080769226 | 0.060256407 |
| legacy-easy-seed-002 | scaffold | 6 | envelope | 0.080769226 | 0.060256407 |
| legacy-easy-seed-002 | radius1 | 3 | site | 0.057792202 | 0.005844146 |
| legacy-easy-seed-002 | radius1 | 3 | envelope | 0.057792202 | 0.005844146 |
| legacy-easy-seed-002 | radius1 | 6 | site | 0.057792202 | 0.005844146 |
| legacy-easy-seed-002 | radius1 | 6 | envelope | 0.057792202 | 0.005844146 |
| legacy-easy-seed-002 | radius3 | 3 | site | 0.047368422 | 0.032330826 |
| legacy-easy-seed-002 | radius3 | 3 | envelope | 0.047368422 | 0.032330826 |
| legacy-easy-seed-002 | radius3 | 6 | site | 0.047368422 | 0.032330826 |
| legacy-easy-seed-002 | radius3 | 6 | envelope | 0.047368422 | 0.032330826 |
| legacy-easy-seed-002 | radius6 | 3 | site | 0.036085486 | 0.029379711 |
| legacy-easy-seed-002 | radius6 | 3 | envelope | 0.036085486 | 0.029379711 |
| legacy-easy-seed-002 | radius6 | 6 | site | 0.036085486 | 0.029379711 |
| legacy-easy-seed-002 | radius6 | 6 | envelope | 0.036085486 | 0.029379711 |
| legacy-easy-seed-003 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-003 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-003 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-003 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-003 | guide | 3 | site | 0.085294113 | 0 |
| legacy-easy-seed-003 | guide | 3 | envelope | 0.085294113 | 0 |
| legacy-easy-seed-003 | guide | 6 | site | 0.085294113 | 0 |
| legacy-easy-seed-003 | guide | 6 | envelope | 0.085294113 | 0 |
| legacy-easy-seed-003 | scaffold | 3 | site | 0.075464189 | 0.054244027 |
| legacy-easy-seed-003 | scaffold | 3 | envelope | 0.075464189 | 0.054244027 |
| legacy-easy-seed-003 | scaffold | 6 | site | 0.075464189 | 0.054244027 |
| legacy-easy-seed-003 | scaffold | 6 | envelope | 0.075464189 | 0.054244027 |
| legacy-easy-seed-003 | radius1 | 3 | site | 0.064814806 | 0.0055555552 |
| legacy-easy-seed-003 | radius1 | 3 | envelope | 0.064814806 | 0.0055555552 |
| legacy-easy-seed-003 | radius1 | 6 | site | 0.064814806 | 0.0055555552 |
| legacy-easy-seed-003 | radius1 | 6 | envelope | 0.064814806 | 0.0055555552 |
| legacy-easy-seed-003 | radius3 | 3 | site | 0.04455252 | 0.028988317 |
| legacy-easy-seed-003 | radius3 | 3 | envelope | 0.04455252 | 0.028988317 |
| legacy-easy-seed-003 | radius3 | 6 | site | 0.04455252 | 0.028988317 |
| legacy-easy-seed-003 | radius3 | 6 | envelope | 0.04455252 | 0.028988317 |
| legacy-easy-seed-003 | radius6 | 3 | site | 0.0098388046 | 0.0044660717 |
| legacy-easy-seed-003 | radius6 | 3 | envelope | 0.0098388046 | 0.0044660717 |
| legacy-easy-seed-003 | radius6 | 6 | site | 0.0098388046 | 0.0044660717 |
| legacy-easy-seed-003 | radius6 | 6 | envelope | 0.0098388046 | 0.0044660717 |
| legacy-easy-seed-004 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-004 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-004 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-004 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-004 | guide | 3 | site | 0.19374999 | 0 |
| legacy-easy-seed-004 | guide | 3 | envelope | 0.19374999 | 0 |
| legacy-easy-seed-004 | guide | 6 | site | 0.19374999 | 0 |
| legacy-easy-seed-004 | guide | 6 | envelope | 0.19374999 | 0 |
| legacy-easy-seed-004 | scaffold | 3 | site | 0.1405983 | 0.10641026 |
| legacy-easy-seed-004 | scaffold | 3 | envelope | 0.1405983 | 0.10641026 |
| legacy-easy-seed-004 | scaffold | 6 | site | 0.1405983 | 0.10641026 |
| legacy-easy-seed-004 | scaffold | 6 | envelope | 0.1405983 | 0.10641026 |
| legacy-easy-seed-004 | radius1 | 3 | site | 0.14729729 | 0.075225219 |
| legacy-easy-seed-004 | radius1 | 3 | envelope | 0.14729729 | 0.075225219 |
| legacy-easy-seed-004 | radius1 | 6 | site | 0.14729729 | 0.075225219 |
| legacy-easy-seed-004 | radius1 | 6 | envelope | 0.14729729 | 0.075225219 |
| legacy-easy-seed-004 | radius3 | 3 | site | 0.094845355 | 0.074226797 |
| legacy-easy-seed-004 | radius3 | 3 | envelope | 0.094845355 | 0.074226797 |
| legacy-easy-seed-004 | radius3 | 6 | site | 0.094845355 | 0.074226797 |
| legacy-easy-seed-004 | radius3 | 6 | envelope | 0.094845355 | 0.074226797 |
| legacy-easy-seed-004 | radius6 | 3 | site | 0.060526311 | 0.052105263 |
| legacy-easy-seed-004 | radius6 | 3 | envelope | 0.060526311 | 0.052105263 |
| legacy-easy-seed-004 | radius6 | 6 | site | 0.060526311 | 0.052105263 |
| legacy-easy-seed-004 | radius6 | 6 | envelope | 0.060526311 | 0.052105263 |
| legacy-easy-seed-005 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-005 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-005 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-005 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-005 | guide | 3 | site | 0.072222218 | 0 |
| legacy-easy-seed-005 | guide | 3 | envelope | 0.072222218 | 0 |
| legacy-easy-seed-005 | guide | 6 | site | 0.072222218 | 0 |
| legacy-easy-seed-005 | guide | 6 | envelope | 0.072222218 | 0 |
| legacy-easy-seed-005 | scaffold | 3 | site | 0.09822695 | 0.069858149 |
| legacy-easy-seed-005 | scaffold | 3 | envelope | 0.09822695 | 0.069858149 |
| legacy-easy-seed-005 | scaffold | 6 | site | 0.09822695 | 0.069858149 |
| legacy-easy-seed-005 | scaffold | 6 | envelope | 0.09822695 | 0.069858149 |
| legacy-easy-seed-005 | radius1 | 3 | site | 0.080158725 | 0.016666666 |
| legacy-easy-seed-005 | radius1 | 3 | envelope | 0.080158725 | 0.016666666 |
| legacy-easy-seed-005 | radius1 | 6 | site | 0.080158725 | 0.016666666 |
| legacy-easy-seed-005 | radius1 | 6 | envelope | 0.080158725 | 0.016666666 |
| legacy-easy-seed-005 | radius3 | 3 | site | 0.048090681 | 0.0289976 |
| legacy-easy-seed-005 | radius3 | 3 | envelope | 0.048090681 | 0.0289976 |
| legacy-easy-seed-005 | radius3 | 6 | site | 0.048090681 | 0.0289976 |
| legacy-easy-seed-005 | radius3 | 6 | envelope | 0.048090681 | 0.0289976 |
| legacy-easy-seed-005 | radius6 | 3 | site | 0.025623798 | 0.017946258 |
| legacy-easy-seed-005 | radius6 | 3 | envelope | 0.025623798 | 0.017946258 |
| legacy-easy-seed-005 | radius6 | 6 | site | 0.025623798 | 0.017946258 |
| legacy-easy-seed-005 | radius6 | 6 | envelope | 0.025623798 | 0.017946258 |
| legacy-easy-seed-006 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-006 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-006 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-006 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-006 | guide | 3 | site | 0.092424244 | 0 |
| legacy-easy-seed-006 | guide | 3 | envelope | 0.092424244 | 0 |
| legacy-easy-seed-006 | guide | 6 | site | 0.092424244 | 0 |
| legacy-easy-seed-006 | guide | 6 | envelope | 0.092424244 | 0 |
| legacy-easy-seed-006 | scaffold | 3 | site | 0.025853008 | 0.0048556328 |
| legacy-easy-seed-006 | scaffold | 3 | envelope | 0.025853008 | 0.0048556328 |
| legacy-easy-seed-006 | scaffold | 6 | site | 0.025853008 | 0.0048556328 |
| legacy-easy-seed-006 | scaffold | 6 | envelope | 0.025853008 | 0.0048556328 |
| legacy-easy-seed-006 | radius1 | 3 | site | 0.044029847 | 0 |
| legacy-easy-seed-006 | radius1 | 3 | envelope | 0.044029847 | 0 |
| legacy-easy-seed-006 | radius1 | 6 | site | 0.044029847 | 0 |
| legacy-easy-seed-006 | radius1 | 6 | envelope | 0.044029847 | 0 |
| legacy-easy-seed-006 | radius3 | 3 | site | 0.0017241389 | 0 |
| legacy-easy-seed-006 | radius3 | 3 | envelope | 0.0017241389 | 0 |
| legacy-easy-seed-006 | radius3 | 6 | site | 0.0017241389 | 0 |
| legacy-easy-seed-006 | radius3 | 6 | envelope | 0.0017241389 | 0 |
| legacy-easy-seed-006 | radius6 | 3 | site | 0 | 0 |
| legacy-easy-seed-006 | radius6 | 3 | envelope | 0 | 0 |
| legacy-easy-seed-006 | radius6 | 6 | site | 0 | 0 |
| legacy-easy-seed-006 | radius6 | 6 | envelope | 0 | 0 |
| legacy-easy-seed-007 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-007 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-007 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-007 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-007 | guide | 3 | site | 0.48636362 | 0.12272727 |
| legacy-easy-seed-007 | guide | 3 | envelope | 0.48636362 | 0.12272727 |
| legacy-easy-seed-007 | guide | 6 | site | 0.48636362 | 0.12272727 |
| legacy-easy-seed-007 | guide | 6 | envelope | 0.48636362 | 0.12272727 |
| legacy-easy-seed-007 | scaffold | 3 | site | 0.22777778 | 0.18333334 |
| legacy-easy-seed-007 | scaffold | 3 | envelope | 0.22777778 | 0.18333334 |
| legacy-easy-seed-007 | scaffold | 6 | site | 0.22777778 | 0.18333334 |
| legacy-easy-seed-007 | scaffold | 6 | envelope | 0.22777778 | 0.18333334 |
| legacy-easy-seed-007 | radius1 | 3 | site | 0.35684928 | 0.24726027 |
| legacy-easy-seed-007 | radius1 | 3 | envelope | 0.35684928 | 0.24726027 |
| legacy-easy-seed-007 | radius1 | 6 | site | 0.35684928 | 0.24726027 |
| legacy-easy-seed-007 | radius1 | 6 | envelope | 0.35684928 | 0.24726027 |
| legacy-easy-seed-007 | radius3 | 3 | site | 0.18962264 | 0.15943396 |
| legacy-easy-seed-007 | radius3 | 3 | envelope | 0.18962264 | 0.15943396 |
| legacy-easy-seed-007 | radius3 | 6 | site | 0.18962264 | 0.15943396 |
| legacy-easy-seed-007 | radius3 | 6 | envelope | 0.18962264 | 0.15943396 |
| legacy-easy-seed-007 | radius6 | 3 | site | 0.058062419 | 0.047659293 |
| legacy-easy-seed-007 | radius6 | 3 | envelope | 0.058062419 | 0.047659293 |
| legacy-easy-seed-007 | radius6 | 6 | site | 0.058062419 | 0.047659293 |
| legacy-easy-seed-007 | radius6 | 6 | envelope | 0.058062419 | 0.047659293 |
| legacy-easy-seed-008 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-008 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-008 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-008 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-008 | guide | 3 | site | 0.61470592 | 0.14411765 |
| legacy-easy-seed-008 | guide | 3 | envelope | 0.61470592 | 0.14411765 |
| legacy-easy-seed-008 | guide | 6 | site | 0.61470592 | 0.14411765 |
| legacy-easy-seed-008 | guide | 6 | envelope | 0.61470592 | 0.14411765 |
| legacy-easy-seed-008 | scaffold | 3 | site | 0.29339623 | 0.21792454 |
| legacy-easy-seed-008 | scaffold | 3 | envelope | 0.29339623 | 0.21792454 |
| legacy-easy-seed-008 | scaffold | 6 | site | 0.29339623 | 0.21792454 |
| legacy-easy-seed-008 | scaffold | 6 | envelope | 0.29339623 | 0.21792454 |
| legacy-easy-seed-008 | radius1 | 3 | site | 0.4726415 | 0.3216981 |
| legacy-easy-seed-008 | radius1 | 3 | envelope | 0.4726415 | 0.3216981 |
| legacy-easy-seed-008 | radius1 | 6 | site | 0.4726415 | 0.3216981 |
| legacy-easy-seed-008 | radius1 | 6 | envelope | 0.4726415 | 0.3216981 |
| legacy-easy-seed-008 | radius3 | 3 | site | 0.27553192 | 0.23297873 |
| legacy-easy-seed-008 | radius3 | 3 | envelope | 0.27553192 | 0.23297873 |
| legacy-easy-seed-008 | radius3 | 6 | site | 0.27553192 | 0.23297873 |
| legacy-easy-seed-008 | radius3 | 6 | envelope | 0.27553192 | 0.23297873 |
| legacy-easy-seed-008 | radius6 | 3 | site | 0.10280898 | 0.087827712 |
| legacy-easy-seed-008 | radius6 | 3 | envelope | 0.10280898 | 0.087827712 |
| legacy-easy-seed-008 | radius6 | 6 | site | 0.10280898 | 0.087827712 |
| legacy-easy-seed-008 | radius6 | 6 | envelope | 0.10280898 | 0.087827712 |
| legacy-easy-seed-009 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-009 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-009 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-009 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-009 | guide | 3 | site | 0.066216215 | 0 |
| legacy-easy-seed-009 | guide | 3 | envelope | 0.066216215 | 0 |
| legacy-easy-seed-009 | guide | 6 | site | 0.066216215 | 0 |
| legacy-easy-seed-009 | guide | 6 | envelope | 0.066216215 | 0 |
| legacy-easy-seed-009 | scaffold | 3 | site | 0.061678827 | 0.04221411 |
| legacy-easy-seed-009 | scaffold | 3 | envelope | 0.061678827 | 0.04221411 |
| legacy-easy-seed-009 | scaffold | 6 | site | 0.061678827 | 0.04221411 |
| legacy-easy-seed-009 | scaffold | 6 | envelope | 0.061678827 | 0.04221411 |
| legacy-easy-seed-009 | radius1 | 3 | site | 0.052614376 | 0.00032679737 |
| legacy-easy-seed-009 | radius1 | 3 | envelope | 0.052614376 | 0.00032679737 |
| legacy-easy-seed-009 | radius1 | 6 | site | 0.052614376 | 0.00032679737 |
| legacy-easy-seed-009 | radius1 | 6 | envelope | 0.052614376 | 0.00032679737 |
| legacy-easy-seed-009 | radius3 | 3 | site | 0.020529792 | 0.0072847605 |
| legacy-easy-seed-009 | radius3 | 3 | envelope | 0.020529792 | 0.0072847605 |
| legacy-easy-seed-009 | radius3 | 6 | site | 0.020529792 | 0.0072847605 |
| legacy-easy-seed-009 | radius3 | 6 | envelope | 0.020529792 | 0.0072847605 |
| legacy-easy-seed-009 | radius6 | 3 | site | 0.010283679 | 0.0046099275 |
| legacy-easy-seed-009 | radius6 | 3 | envelope | 0.010283679 | 0.0046099275 |
| legacy-easy-seed-009 | radius6 | 6 | site | 0.010283679 | 0.0046099275 |
| legacy-easy-seed-009 | radius6 | 6 | envelope | 0.010283679 | 0.0046099275 |
| legacy-easy-seed-010 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-010 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-010 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-010 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-010 | guide | 3 | site | 0.072222218 | 0 |
| legacy-easy-seed-010 | guide | 3 | envelope | 0.072222218 | 0 |
| legacy-easy-seed-010 | guide | 6 | site | 0.072222218 | 0 |
| legacy-easy-seed-010 | guide | 6 | envelope | 0.072222218 | 0 |
| legacy-easy-seed-010 | scaffold | 3 | site | 0.034079596 | 0.014179096 |
| legacy-easy-seed-010 | scaffold | 3 | envelope | 0.034079596 | 0.014179096 |
| legacy-easy-seed-010 | scaffold | 6 | site | 0.034079596 | 0.014179096 |
| legacy-easy-seed-010 | scaffold | 6 | envelope | 0.034079596 | 0.014179096 |
| legacy-easy-seed-010 | radius1 | 3 | site | 0.03918919 | 0 |
| legacy-easy-seed-010 | radius1 | 3 | envelope | 0.03918919 | 0 |
| legacy-easy-seed-010 | radius1 | 6 | site | 0.03918919 | 0 |
| legacy-easy-seed-010 | radius1 | 6 | envelope | 0.03918919 | 0 |
| legacy-easy-seed-010 | radius3 | 3 | site | 0.0018578321 | 0 |
| legacy-easy-seed-010 | radius3 | 3 | envelope | 0.0018578321 | 0 |
| legacy-easy-seed-010 | radius3 | 6 | site | 0.0018578321 | 0 |
| legacy-easy-seed-010 | radius3 | 6 | envelope | 0.0018578321 | 0 |
| legacy-easy-seed-010 | radius6 | 3 | site | 0 | 0 |
| legacy-easy-seed-010 | radius6 | 3 | envelope | 0 | 0 |
| legacy-easy-seed-010 | radius6 | 6 | site | 0 | 0 |
| legacy-easy-seed-010 | radius6 | 6 | envelope | 0 | 0 |
| legacy-easy-seed-011 | empty | 3 | site | 0 | 0 |
| legacy-easy-seed-011 | empty | 3 | envelope | 0 | 0 |
| legacy-easy-seed-011 | empty | 6 | site | 0 | 0 |
| legacy-easy-seed-011 | empty | 6 | envelope | 0 | 0 |
| legacy-easy-seed-011 | guide | 3 | site | 0.043548375 | 0 |
| legacy-easy-seed-011 | guide | 3 | envelope | 0.043548375 | 0 |
| legacy-easy-seed-011 | guide | 6 | site | 0.043548375 | 0 |
| legacy-easy-seed-011 | guide | 6 | envelope | 0.043548375 | 0 |
| legacy-easy-seed-011 | scaffold | 3 | site | 0.059876531 | 0.035185173 |
| legacy-easy-seed-011 | scaffold | 3 | envelope | 0.059876531 | 0.035185173 |
| legacy-easy-seed-011 | scaffold | 6 | site | 0.059876531 | 0.035185173 |
| legacy-easy-seed-011 | scaffold | 6 | envelope | 0.059876531 | 0.035185173 |
| legacy-easy-seed-011 | radius1 | 3 | site | 0.053389817 | 0.0025423616 |
| legacy-easy-seed-011 | radius1 | 3 | envelope | 0.053389817 | 0.0025423616 |
| legacy-easy-seed-011 | radius1 | 6 | site | 0.053389817 | 0.0025423616 |
| legacy-easy-seed-011 | radius1 | 6 | envelope | 0.053389817 | 0.0025423616 |
| legacy-easy-seed-011 | radius3 | 3 | site | 0.040909082 | 0.027272716 |
| legacy-easy-seed-011 | radius3 | 3 | envelope | 0.040909082 | 0.027272716 |
| legacy-easy-seed-011 | radius3 | 6 | site | 0.040909082 | 0.027272716 |
| legacy-easy-seed-011 | radius3 | 6 | envelope | 0.040909082 | 0.027272716 |
| legacy-easy-seed-011 | radius6 | 3 | site | 0.013476864 | 0.0086921751 |
| legacy-easy-seed-011 | radius6 | 3 | envelope | 0.013476864 | 0.0086921751 |
| legacy-easy-seed-011 | radius6 | 6 | site | 0.013476864 | 0.0086921751 |
| legacy-easy-seed-011 | radius6 | 6 | envelope | 0.013476864 | 0.0086921751 |
| ref-01-ground-pair | empty | 3 | site | 0 | 0 |
| ref-01-ground-pair | empty | 3 | envelope | 0 | 0 |
| ref-01-ground-pair | empty | 6 | site | 0 | 0 |
| ref-01-ground-pair | empty | 6 | envelope | 0 | 0 |
| ref-01-ground-pair | guide | 3 | site | 0 | 0 |
| ref-01-ground-pair | guide | 3 | envelope | 0 | 0 |
| ref-01-ground-pair | guide | 6 | site | 0 | 0 |
| ref-01-ground-pair | guide | 6 | envelope | 0 | 0 |
| ref-01-ground-pair | scaffold | 3 | site | 0 | 0 |
| ref-01-ground-pair | scaffold | 3 | envelope | 0 | 0 |
| ref-01-ground-pair | scaffold | 6 | site | 0 | 0 |
| ref-01-ground-pair | scaffold | 6 | envelope | 0 | 0 |
| ref-01-ground-pair | radius1 | 3 | site | 0 | 0 |
| ref-01-ground-pair | radius1 | 3 | envelope | 0 | 0 |
| ref-01-ground-pair | radius1 | 6 | site | 0 | 0 |
| ref-01-ground-pair | radius1 | 6 | envelope | 0 | 0 |
| ref-01-ground-pair | radius3 | 3 | site | 0 | 0 |
| ref-01-ground-pair | radius3 | 3 | envelope | 0 | 0 |
| ref-01-ground-pair | radius3 | 6 | site | 0 | 0 |
| ref-01-ground-pair | radius3 | 6 | envelope | 0 | 0 |
| ref-01-ground-pair | radius6 | 3 | site | 0 | 0 |
| ref-01-ground-pair | radius6 | 3 | envelope | 0 | 0 |
| ref-01-ground-pair | radius6 | 6 | site | 0 | 0 |
| ref-01-ground-pair | radius6 | 6 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | empty | 3 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | empty | 3 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | empty | 6 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | empty | 6 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | guide | 3 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | guide | 3 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | guide | 6 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | guide | 6 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 3 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 3 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 6 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | scaffold | 6 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 3 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 3 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 6 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | radius1 | 6 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 3 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 3 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 6 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | radius3 | 6 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 3 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 3 | envelope | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 6 | site | 0 | 0 |
| ref-02-facade-pair-and-ground | radius6 | 6 | envelope | 0 | 0 |
| ref-03-wide-gap | empty | 3 | site | 0 | 0 |
| ref-03-wide-gap | empty | 3 | envelope | 0 | 0 |
| ref-03-wide-gap | empty | 6 | site | 0 | 0 |
| ref-03-wide-gap | empty | 6 | envelope | 0 | 0 |
| ref-03-wide-gap | guide | 3 | site | 0.0068627447 | 0 |
| ref-03-wide-gap | guide | 3 | envelope | 0.0068627447 | 0 |
| ref-03-wide-gap | guide | 6 | site | 0.0068627447 | 0 |
| ref-03-wide-gap | guide | 6 | envelope | 0.0068627447 | 0 |
| ref-03-wide-gap | scaffold | 3 | site | 0.026715174 | 0.010083154 |
| ref-03-wide-gap | scaffold | 3 | envelope | 0.026715174 | 0.010083154 |
| ref-03-wide-gap | scaffold | 6 | site | 0.026715174 | 0.010083154 |
| ref-03-wide-gap | scaffold | 6 | envelope | 0.026715174 | 0.010083154 |
| ref-03-wide-gap | radius1 | 3 | site | 0.029611647 | 0 |
| ref-03-wide-gap | radius1 | 3 | envelope | 0.029611647 | 0 |
| ref-03-wide-gap | radius1 | 6 | site | 0.029611647 | 0 |
| ref-03-wide-gap | radius1 | 6 | envelope | 0.029611647 | 0 |
| ref-03-wide-gap | radius3 | 3 | site | 0.009653464 | 0 |
| ref-03-wide-gap | radius3 | 3 | envelope | 0.009653464 | 0 |
| ref-03-wide-gap | radius3 | 6 | site | 0.009653464 | 0 |
| ref-03-wide-gap | radius3 | 6 | envelope | 0.009653464 | 0 |
| ref-03-wide-gap | radius6 | 3 | site | 0 | 0 |
| ref-03-wide-gap | radius6 | 3 | envelope | 0 | 0 |
| ref-03-wide-gap | radius6 | 6 | site | 0 | 0 |
| ref-03-wide-gap | radius6 | 6 | envelope | 0 | 0 |
| ref-04-asymmetric-heights | empty | 3 | site | 0 | 0 |
| ref-04-asymmetric-heights | empty | 3 | envelope | 0 | 0 |
| ref-04-asymmetric-heights | empty | 6 | site | 0 | 0 |
| ref-04-asymmetric-heights | empty | 6 | envelope | 0 | 0 |
| ref-04-asymmetric-heights | guide | 3 | site | 0 | 0 |
| ref-04-asymmetric-heights | guide | 3 | envelope | 0 | 0 |
| ref-04-asymmetric-heights | guide | 6 | site | 0 | 0 |
| ref-04-asymmetric-heights | guide | 6 | envelope | 0 | 0 |
| ref-04-asymmetric-heights | scaffold | 3 | site | 0.062634811 | 0.050308153 |
| ref-04-asymmetric-heights | scaffold | 3 | envelope | 0.062634811 | 0.050308153 |
| ref-04-asymmetric-heights | scaffold | 6 | site | 0.062634811 | 0.050308153 |
| ref-04-asymmetric-heights | scaffold | 6 | envelope | 0.062634811 | 0.050308153 |
| ref-04-asymmetric-heights | radius1 | 3 | site | 0.017857134 | 0 |
| ref-04-asymmetric-heights | radius1 | 3 | envelope | 0.017857134 | 0 |
| ref-04-asymmetric-heights | radius1 | 6 | site | 0.017857134 | 0 |
| ref-04-asymmetric-heights | radius1 | 6 | envelope | 0.017857134 | 0 |
| ref-04-asymmetric-heights | radius3 | 3 | site | 0.015714273 | 0.0080952346 |
| ref-04-asymmetric-heights | radius3 | 3 | envelope | 0.015714273 | 0.0080952346 |
| ref-04-asymmetric-heights | radius3 | 6 | site | 0.015714273 | 0.0080952346 |
| ref-04-asymmetric-heights | radius3 | 6 | envelope | 0.015714273 | 0.0080952346 |
| ref-04-asymmetric-heights | radius6 | 3 | site | 0 | 0 |
| ref-04-asymmetric-heights | radius6 | 3 | envelope | 0 | 0 |
| ref-04-asymmetric-heights | radius6 | 6 | site | 0 | 0 |
| ref-04-asymmetric-heights | radius6 | 6 | envelope | 0 | 0 |
| ref-05-sealed-partition | empty | 3 | site | 0 | 0 |
| ref-05-sealed-partition | empty | 3 | envelope | 0 | 0 |
| ref-05-sealed-partition | empty | 6 | site | 0 | 0 |
| ref-05-sealed-partition | empty | 6 | envelope | 0 | 0 |
| ref-05-sealed-partition | guide | 3 | site | 0.34999999 | 0 |
| ref-05-sealed-partition | guide | 3 | envelope | 0.34999999 | 0 |
| ref-05-sealed-partition | guide | 6 | site | 0.34999999 | 0 |
| ref-05-sealed-partition | guide | 6 | envelope | 0.34999999 | 0 |
| ref-05-sealed-partition | scaffold | 3 | site | 0.18333334 | 0.12777779 |
| ref-05-sealed-partition | scaffold | 3 | envelope | 0.18333334 | 0.12777779 |
| ref-05-sealed-partition | scaffold | 6 | site | 0.18333334 | 0.12777779 |
| ref-05-sealed-partition | scaffold | 6 | envelope | 0.18333334 | 0.12777779 |
| ref-05-sealed-partition | radius1 | 3 | site | 0.27857143 | 0.13571429 |
| ref-05-sealed-partition | radius1 | 3 | envelope | 0.27857143 | 0.13571429 |
| ref-05-sealed-partition | radius1 | 6 | site | 0.27857143 | 0.13571429 |
| ref-05-sealed-partition | radius1 | 6 | envelope | 0.27857143 | 0.13571429 |
| ref-05-sealed-partition | radius3 | 3 | site | 0.18333334 | 0.15000001 |
| ref-05-sealed-partition | radius3 | 3 | envelope | 0.18333334 | 0.15000001 |
| ref-05-sealed-partition | radius3 | 6 | site | 0.18333334 | 0.15000001 |
| ref-05-sealed-partition | radius3 | 6 | envelope | 0.18333334 | 0.15000001 |
| ref-05-sealed-partition | radius6 | 3 | site | 0.16100478 | 0.15143541 |
| ref-05-sealed-partition | radius6 | 3 | envelope | 0.16100478 | 0.15143541 |
| ref-05-sealed-partition | radius6 | 6 | site | 0.16100478 | 0.15143541 |
| ref-05-sealed-partition | radius6 | 6 | envelope | 0.16100478 | 0.15143541 |
| ref-06-minimal-smoke | empty | 3 | site | 0 | 0 |
| ref-06-minimal-smoke | empty | 3 | envelope | 0 | 0 |
| ref-06-minimal-smoke | empty | 6 | site | 0 | 0 |
| ref-06-minimal-smoke | empty | 6 | envelope | 0 | 0 |
| ref-06-minimal-smoke | guide | 3 | site | 0 | 0 |
| ref-06-minimal-smoke | guide | 3 | envelope | 0 | 0 |
| ref-06-minimal-smoke | guide | 6 | site | 0 | 0 |
| ref-06-minimal-smoke | guide | 6 | envelope | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 3 | site | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 3 | envelope | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 6 | site | 0 | 0 |
| ref-06-minimal-smoke | scaffold | 6 | envelope | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 3 | site | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 3 | envelope | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 6 | site | 0 | 0 |
| ref-06-minimal-smoke | radius1 | 6 | envelope | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 3 | site | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 3 | envelope | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 6 | site | 0 | 0 |
| ref-06-minimal-smoke | radius3 | 6 | envelope | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 3 | site | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 3 | envelope | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 6 | site | 0 | 0 |
| ref-06-minimal-smoke | radius6 | 6 | envelope | 0 | 0 |
