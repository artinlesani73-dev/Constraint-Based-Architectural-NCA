# A2: access-contract replay and gradient audit

Run `20260923T123641Z_98bf30045a6f`; no optimizer updates.

Candidate component_bottleneck_v2 requires one connected material component to touch every entrance region. It also replaces mean-destination scoring by worst-destination strength and removes the spatial hop limit. The attribution columns isolate these changes. Existing results retain their original definitions.

## Replay summary

| Set | Cases | Old connected | New connected | Old unscorable | Access lower / higher | Source choice improves | Hop removal improves | Worst reduction increases loss |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| fitting | 56 | 3 | 3 | 0 | 8 / 0 | 8 | 0 | 0 |
| sensitivity | 187 | 120 | 120 | 0 | 0 / 30 | 0 | 0 | 30 |
| direct | 34 | 34 | 34 | 0 | 0 / 0 | 0 | 0 | 0 |

Candidate changes are rescoring of saved fields, not learned improvements. W1 controls are included in the sensitivity set. A disconnected fragment cannot pool entrance contact with another component.

## Actual parameter gradients

| Model | Scene | Growth | Access v1 / v2 | Access parameter norm v1 / v2 | Coverage norm | Sparsity norm | Candidate access vs coverage cosine | Total v1 vs v2 cosine |
|---|---|---:|---|---|---:|---:|---|---|
| original | ref-01-ground-pair | 16 | 1 / 1 | 0 / 0 | 2.159521 | 59.77263 | null | 1 |
| original | ref-01-ground-pair | 50 | 1 / 1 | 0 / 0 | 0.1559384 | 1276.081 | null | 1 |
| original | ref-06-minimal-smoke | 16 | 1 / 1 | 0 / 0 | 2.672049 | 4.886703 | null | 1 |
| original | ref-06-minimal-smoke | 50 | 1 / 1 | 0 / 0 | 0.123722 | 298.6038 | null | 1 |
| mapped_30-r0 | ref-01-ground-pair | 16 | 1 / 0.9145257 | 0 / 11.54948 | 5.208994 | 4.573601 | 0.7123908 | 0.1155684 |
| mapped_30-r0 | ref-01-ground-pair | 50 | 1 / 1 | 0 / 0 | 0.07686283 | 270.6502 | null | 1 |
| mapped_30-r1 | ref-06-minimal-smoke | 16 | 1 / 0.7888916 | 0 / 18.69453 | 4.471279 | 7.478035 | 0.8029632 | -0.1736592 |
| mapped_30-r1 | ref-06-minimal-smoke | 50 | 1 / 0 | 0 / 0 | 0.09248851 | 59.96805 | null | 1 |
| mass_3-r0 | ref-01-ground-pair | 16 | 1 / 0.5607875 | 0 / 7.370561 | 5.013884 | 39.65145 | 0.7819933 | 0.3510046 |
| mass_3-r0 | ref-01-ground-pair | 50 | 1 / 0 | 0 / 0 | 0.1359873 | 37.87272 | null | 1 |
| mass_3-r1 | ref-06-minimal-smoke | 16 | 1 / 0.5083662 | 0 / 11.57685 | 4.574538 | 43.9332 | 0.7770444 | -0.05399789 |
| mass_3-r1 | ref-06-minimal-smoke | 50 | 1 / 0 | 0 / 0 | 0.1302139 | 29.66101 | null | 1 |

All forwards match their saved raw/material arrays exactly. Weights remain frozen. Norms are raw derivatives before clipping/Adam; cosines are not optimizer predictions. Zero-vector cosines are null. Topology selection is detached CPU union-find, with a live-tensor critical-voxel derivative and deterministic ties; no GPU efficiency or smoothness claim.

## Every replay case

| Set / model | Scene | Update | Growth | Access old | Fixed worst64 | Fixed worst unbounded | Candidate | Old / new connected |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fitting / mapped_30-r0 | ref-01-ground-pair | 0 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 0 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 1 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 1 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 3 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 3 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 8 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 8 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 16 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 16 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 32 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 32 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 64 | 16 | 1 | 1 | 1 | 0.9145257 | False / False |
| fitting / mapped_30-r0 | ref-01-ground-pair | 64 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 0 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 0 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 1 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 1 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 3 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 3 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 8 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 8 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 16 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 16 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 32 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 32 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 64 | 16 | 1 | 1 | 1 | 0.7888916 | False / False |
| fitting / mapped_30-r1 | ref-06-minimal-smoke | 64 | 50 | 1 | 1 | 1 | 0 | True / True |
| fitting / mass_3-r0 | ref-01-ground-pair | 0 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 0 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 1 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 1 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 3 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 3 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 8 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 8 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 16 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 16 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 32 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 32 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 64 | 16 | 1 | 1 | 1 | 0.5607875 | False / False |
| fitting / mass_3-r0 | ref-01-ground-pair | 64 | 50 | 1 | 1 | 1 | 0 | True / True |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 0 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 0 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 1 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 1 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 3 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 3 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 8 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 8 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 16 | 16 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 16 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 32 | 16 | 1 | 1 | 1 | 0.9880437 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 32 | 50 | 1 | 1 | 1 | 1 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 64 | 16 | 1 | 1 | 1 | 0.5083662 | False / False |
| fitting / mass_3-r1 | ref-06-minimal-smoke | 64 | 50 | 1 | 1 | 1 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-000 | None | 16 | 0.03638673 | 0.03638673 | 0.03638673 | 0.03638673 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-000 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-001 | None | 16 | 0.05145496 | 0.05145496 | 0.05145496 | 0.05145496 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-001 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-002 | None | 16 | 0.1095353 | 0.1095353 | 0.1095353 | 0.1095353 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-002 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-003 | None | 16 | 0.03941011 | 0.03941011 | 0.03941011 | 0.03941011 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-003 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-004 | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | legacy-easy-seed-004 | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | legacy-easy-seed-005 | None | 16 | 0.1212684 | 0.1212684 | 0.1212684 | 0.1212684 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-005 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-006 | None | 16 | 0.02418178 | 0.02418178 | 0.02418178 | 0.02418178 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-006 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-007 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-007 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-008 | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | legacy-easy-seed-008 | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | legacy-easy-seed-009 | None | 16 | 0.0438605 | 0.0438605 | 0.0438605 | 0.0438605 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-009 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-010 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-010 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-011 | None | 16 | 0.1710694 | 0.1710694 | 0.1710694 | 0.1710694 | True / True |
| sensitivity / mapped_30-s0 | legacy-easy-seed-011 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s0 | ref-01-ground-pair | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-01-ground-pair | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-02-facade-pair-and-ground | None | 16 | 0.5893782 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-02-facade-pair-and-ground | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-03-wide-gap | None | 16 | 0.5123857 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-03-wide-gap | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-04-asymmetric-heights | None | 16 | 0.5147315 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-04-asymmetric-heights | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-06-minimal-smoke | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s0 | ref-06-minimal-smoke | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | legacy-easy-seed-000 | None | 16 | 0.03040481 | 0.03040481 | 0.03040481 | 0.03040481 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-000 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-001 | None | 16 | 0.04509223 | 0.04509223 | 0.04509223 | 0.04509223 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-001 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-002 | None | 16 | 0.09054798 | 0.09054798 | 0.09054798 | 0.09054798 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-002 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-003 | None | 16 | 0.03650284 | 0.03650284 | 0.03650284 | 0.03650284 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-003 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-004 | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | legacy-easy-seed-004 | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | legacy-easy-seed-005 | None | 16 | 0.1154357 | 0.1154357 | 0.1154357 | 0.1154357 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-005 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-006 | None | 16 | 0.01971549 | 0.01971549 | 0.01971549 | 0.01971549 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-006 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-007 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-007 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-008 | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | legacy-easy-seed-008 | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | legacy-easy-seed-009 | None | 16 | 0.04417783 | 0.04417783 | 0.04417783 | 0.04417783 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-009 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-010 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-010 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-011 | None | 16 | 0.1665362 | 0.1665362 | 0.1665362 | 0.1665362 | True / True |
| sensitivity / mapped_30-s1 | legacy-easy-seed-011 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mapped_30-s1 | ref-01-ground-pair | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-01-ground-pair | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-02-facade-pair-and-ground | None | 16 | 0.5875062 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-02-facade-pair-and-ground | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-03-wide-gap | None | 16 | 0.5102451 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-03-wide-gap | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-04-asymmetric-heights | None | 16 | 0.5154703 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-04-asymmetric-heights | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-06-minimal-smoke | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mapped_30-s1 | ref-06-minimal-smoke | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | legacy-easy-seed-000 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-000 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-001 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-001 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-002 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-002 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-003 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-003 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-004 | None | 16 | 0.7859593 | 0.7859593 | 0.7859593 | 0.7859593 | False / False |
| sensitivity / mass_3-s0 | legacy-easy-seed-004 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-005 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-005 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-006 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-006 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-007 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-007 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-008 | None | 16 | 0.6586316 | 0.6586316 | 0.6586316 | 0.6586316 | False / False |
| sensitivity / mass_3-s0 | legacy-easy-seed-008 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-009 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-009 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-010 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-010 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-011 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | legacy-easy-seed-011 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s0 | ref-01-ground-pair | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-01-ground-pair | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-02-facade-pair-and-ground | None | 16 | 0.5208275 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-02-facade-pair-and-ground | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-03-wide-gap | None | 16 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-03-wide-gap | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-04-asymmetric-heights | None | 16 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-04-asymmetric-heights | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-06-minimal-smoke | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s0 | ref-06-minimal-smoke | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | legacy-easy-seed-000 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-000 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-001 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-001 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-002 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-002 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-003 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-003 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-004 | None | 16 | 0.880263 | 0.880263 | 0.880263 | 0.880263 | False / False |
| sensitivity / mass_3-s1 | legacy-easy-seed-004 | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | legacy-easy-seed-005 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-005 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-006 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-006 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-007 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-007 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-008 | None | 16 | 0.7317786 | 0.7317786 | 0.7317786 | 0.7317786 | False / False |
| sensitivity / mass_3-s1 | legacy-easy-seed-008 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-009 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-009 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-010 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-010 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-011 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | legacy-easy-seed-011 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / mass_3-s1 | ref-01-ground-pair | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-01-ground-pair | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-02-facade-pair-and-ground | None | 16 | 0.5170482 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-02-facade-pair-and-ground | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-03-wide-gap | None | 16 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-03-wide-gap | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-04-asymmetric-heights | None | 16 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-04-asymmetric-heights | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-06-minimal-smoke | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / mass_3-s1 | ref-06-minimal-smoke | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | legacy-easy-seed-000 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-000 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-001 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-001 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-002 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-002 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-003 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-003 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-004 | None | 16 | 0.9365448 | 0.9365448 | 0.9365448 | 0.9365448 | False / False |
| sensitivity / original_checkpoint | legacy-easy-seed-004 | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | legacy-easy-seed-005 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-005 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-006 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-006 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-007 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-007 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-008 | None | 16 | 0.8703324 | 0.8703324 | 0.8703324 | 0.8703324 | False / False |
| sensitivity / original_checkpoint | legacy-easy-seed-008 | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | legacy-easy-seed-009 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-009 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-010 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-010 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-011 | None | 16 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | legacy-easy-seed-011 | None | 50 | 0 | 0 | 0 | 0 | True / True |
| sensitivity / original_checkpoint | ref-01-ground-pair | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-01-ground-pair | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-02-facade-pair-and-ground | None | 16 | 0.5213503 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-02-facade-pair-and-ground | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-03-wide-gap | None | 16 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-03-wide-gap | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-04-asymmetric-heights | None | 16 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-04-asymmetric-heights | None | 50 | 0.5 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-06-minimal-smoke | None | 16 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / original_checkpoint | ref-06-minimal-smoke | None | 50 | 1 | 1 | 1 | 1 | False / False |
| sensitivity / W1_procedural | legacy-easy-seed-000 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-001 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-002 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-003 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-004 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-005 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-006 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-007 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-008 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-009 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-010 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | legacy-easy-seed-011 | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | ref-01-ground-pair | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | ref-02-facade-pair-and-ground | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | ref-03-wide-gap | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | ref-04-asymmetric-heights | None | None | 0 | 0 | 0 | 0 | True / True |
| sensitivity / W1_procedural | ref-06-minimal-smoke | None | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-000 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-000 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-001 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-001 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-002 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-002 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-003 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-003 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-004 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-004 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-005 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-005 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-006 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-006 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-007 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-007 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-008 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-008 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-009 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-009 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-010 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-010 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | legacy-easy-seed-011 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | legacy-easy-seed-011 | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | ref-01-ground-pair | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | ref-01-ground-pair | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | ref-02-facade-pair-and-ground | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | ref-02-facade-pair-and-ground | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | ref-03-wide-gap | 32 | None | 0.02792257 | 0.02792257 | 0.02792257 | 0.02792257 | True / True |
| direct / mass_3 | ref-03-wide-gap | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | ref-04-asymmetric-heights | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | ref-04-asymmetric-heights | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mapped_30 | ref-06-minimal-smoke | 32 | None | 0 | 0 | 0 | 0 | True / True |
| direct / mass_3 | ref-06-minimal-smoke | 32 | None | 0 | 0 | 0 | 0 | True / True |

## Verification

```json
{
  "registered_hashes_verified": true,
  "source_hashes_verified": 27,
  "candidate_replays_recomputed": 277,
  "binary_bfs_recomputed": 277,
  "gradient_cases_verified": 12,
  "parameter_vectors_verified": 72,
  "last_raw_gradients_verified": 72,
  "cosines_verified": 432,
  "saved_forward_fields_match": 12,
  "optimizer_updates": 0,
  "scope": "Candidate scores/BFS and vector norms/cosines recomputed. Actual backpropagation is recorded, not independently reimplemented."
}
```

All source/hash references, critical coordinates, old source ambiguity, fixed-point distances, gradients and per-recipe totals are retained in A2-evidence.json and the immutable run. Candidate totals replace only access coefficient15; other eight families and regularizers remain unchanged. These are development scenes, not unseen validation. No model/recipe promotion, paid compute, Drive operation, deployment or production default change.
