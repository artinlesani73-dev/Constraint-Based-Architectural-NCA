# Corridor correction findings

Completed locally on 2026-09-23. C1 run `20260923T000945Z_3cbdc3603a12` compares
three target versions across 18 frozen scenes, then runs 108 matched forward
cases with the original checkpoint. No weights were trained or deployed.

The geometry corrections work, but changing targets alone does not repair the
historical model's failures. Correct the training objectives before larger grids
or paid training. The next concrete gate is [LOSS_REPAIR_PLAN.md](LOSS_REPAIR_PLAN.md).

## What changed

- `corridor_bounded_v1` fixes only the cascading vertical expansion. Original
  routing, dilation order and height clipping remain, providing an isolated test.
- `corridor_legal_v1` separately finds exact face-connected paths in permitted
  space using explicit entrance IDs. It preserves a bounded expansion, avoids
  the legacy endpoint-height clip, records paths and reports impossible scenes.
- Original v31 corridor code, notebook, checkpoint and production defaults remain
  available unchanged. The new operators are explicit experimental choices.

## Measured target behavior

| Set | Legacy legal targets connected | Bounded version | Legal router | Legacy forbidden target voxels | Legal router forbidden voxels |
|---|---:|---:|---:|---:|---:|
| Legacy scenes | 12/12 | 12/12 | 12/12 | 1,626 | 0 |
| Reference scenes | 3/6 | 3/6 | 5/6 | 2,518 | 0 |

The sixth reference scene is intentionally sealed. The legal router connects
all 17 scenes with a legal route, flags the sealed case as infeasible and keeps
its partial geometry as evidence. Its mean target size is 278.7 voxels on legacy
scenes and 375.2 on reference scenes, compared with 1,213.8 and 2,143.7 before.
Smaller does not automatically mean better architecture.

The bounded-only fix preserves target connectivity and removes no legacy-set
forbidden target voxels; on reference scenes it removes just one forbidden voxel.
This confirms why a separate legality/routing intervention was needed. Radius-zero
bounded/legacy equality passes on every scene, and the copied pipeline was checked
to differ only in envelope replacement and radius argument validation.

## What the original checkpoint does with these targets

Predetermined seed 0, 50 steps, identical scene inputs and checkpoint:

| Set/profile | Legacy target | Bounded target | Legal target |
|---|---:|---:|---:|
| Legacy / training forward profile | 10/12 | 10/12 | 10/12 |
| Legacy / serving forward profile | 10/12 | 10/12 | 9/12 |
| Reference / training forward profile | 0/6 | 0/6 | 0/6 |
| Reference / serving forward profile | 0/6 | 0/6 | 0/6 |

The legal target loses the serving connection on `legacy-easy-seed-011`. Both
profiles still fail on every feasible reference scene even though their new
targets are connected. The procedural scaffold therefore remains a strong
connectivity control; the old NCA has not demonstrated added value on this measure.

Legal targets substantially reduce output volume: on legacy scenes the training
forward profile averages 276.2 voxels instead of 1,076.2, and serving 208.0 instead
of 806.0. These are material counts, not architectural quality scores. Output
legality remains zero violations because of hard projection. Geometric-support
counts and thickness proxies are recorded, without mechanical safety claims.

## Additional objective conflict found after the target audit

The notebook's actual trainer uses a 3% minimum mass ratio over all non-building
voxels. With occupancy bounded to [0,1], completely filling each new legal target
would use only 0.405%-2.538% of that region. All 18 targets are below the floor,
including the infeasible control (all 17 feasible scenes are below it too).

Therefore zero spill outside these targets and zero lower-volume penalty cannot
both be achieved by reusing the old losses unchanged. These are competing soft
objectives, not a proof that no feasible architectural design exists. This
calculation is explicitly post-hoc; the source notebook/defaults and exact
per-scene values are retained in the volume audit. Do not silently change the
volume floor to hide the conflict. Separate connection guidance from the region
available for architectural material before selecting the revised objectives.

## Verification, evidence and limitations

- 94 regression tests passed, zero failures/errors/skips; checkpoint smoke exit 0.
- 54 target records and 108/108 forward cases completed in 343.54 seconds locally.
- All 36 legacy-arm final fields match saved E0 fields bit for bit.
- All registered artifact hashes verified; fresh report and both audit renders
  exactly reproduce saved files. No unscorable connectivity cases in this run.
- First attempt `20260923T000715Z_1e06516e9763` is retained as failed: Windows
  filename length stopped evidence publication after 54 targets and 7 recorded
  forward cases. Short evidence filenames fixed the retry; full IDs remain in JSON.
- One post-hoc parser attempt rejected an ambiguous notebook constructor because
  it included a standalone loss demonstration. The parser now selects the actual
  trainer constructor. Original parser source/error are preserved locally.

Read the [full per-scene report](reports/20260923T000945Z_3cbdc3603a12-C1.md),
[target audit](reports/20260923T000945Z_3cbdc3603a12-C1.target-audit.json),
[volume audit](reports/20260923T000945Z_3cbdc3603a12-C1.volume-audit.json) and
[verification receipt](reports/20260923T000945Z_3cbdc3603a12-C1.verification.json).
Full arrays/configuration/source snapshots are in the run's `.local-artifacts`
folder, with a local milestone archive and restore instructions.

One rollout seed and development scenes do not establish generalization.
Spatial material connectivity does not establish a walkable route, headroom,
floor/deck geometry or structural engineering safety. The legal router is a
CPU procedural control; larger-grid speed and memory need profiling. No new
constraint family, paid training, Drive access, GitHub push or deployment occurred.
