# NR5 failure diagnosis - 2026-09-28

## Evidence and scope

Recomputed all27 saved validation reports from their original binary fields,
with source NPZ SHA256 verification and exact probability>0.5 equality. All
archived nine-family and reconstruction metrics reproduce exactly. No inference,
training, threshold sweep, TEST access or additional stochastic rollout.
Source: Codex/2026-09-06/cre/outputs/NR5-Single-Trial-Review; target/input source
NL0 run20260925T094341Z_316cff241020 and saved NR3 display inputs.
Raw research score remains17/27. Ten failures: access10, support4, thickness2.

## What actually fails

All27 outputs reach both declared interfaces in raw occupancy and bulk, and
all have zero unreachable bulk cells. All10 failing outputs contain1-6 raw
voxels detached from the interface-reached mass,27cells total.24 are false
additions and3 belong to the reference target. None of these27 cells were
present in the damaged input: the3 target cells are newly reconstructed but
remain disconnected. Therefore NR4's all-detached-cells-are-excess explanation
must NOT be reused for NR5.

Four outputs fail geometric support; two include unsupported target cells.
Two cube-damage outputs fail thickness: bulk fractions0.880980 and0.897647,
below the existing0.90 cutoff. Missing target cells are41 and64 respectively.
These are deficiencies of the binary geometric contract; no mechanics claim.
The access family demands every occupied cell belong to the reached mass, so
one stray voxel can fail it even when both interfaces and the bulk connect.
Better average overlap does not guarantee this global property.

## Diagnostic counterfactuals (not admitted results)

Evaluate three fixed transformations on all27 saved outputs, retain derived
arrays separately, and do not overwrite model output:

| Transformation | All-nine passes | Interpretation |
|---|---:|---|
| Raw NR5 |17/27|Original learned output|
| Remove interface-unreachable occupied cells |26/27|Deletes27cells including3 correct reconstructed target cells|
| Remove every false addition using target |24/27|Oracle, not deployable; missing geometry still matters|
| Restore every missing cell using target |18/27|Oracle, not deployable; most excess islands remain|

Pruning fixes nine of ten failed outputs and introduces no new all-nine failures.
The remaining v16/seed0 cube case still fails thickness. In v16/seed1 cube case,
pruning raises the bulk fraction past acceptance partly by shrinking the
denominator: passing is not evidence that missing target volume was restored.
Thus26/27 is a postprocessed diagnostic score, never an improved NCA score.

## Next design recommendation

Keep the building-mass concept and nine families. Evidence points toward local
birth control/connectivity and volumetric repair, rather than a larger grid or
another global negative-BCE increase. Current NR5 objective is voxel BCE plus
negative and intact weighting; it does not directly optimize the binary access
and bulk-connectivity predicates. This mismatch is observed in code; its causal
contribution has not been isolated by these saved-output probes.

Propose one substantial next design: a connectivity-aware repair update that
conditions additions on an occupied local growth front, trained with supervision
for volumetric neighborhoods as well as voxel reconstruction. Preserve surviving
input in this removal-only damage task; keep its role explicit, since arbitrary
input corruption would require deletions. Distinguish correct-but-disconnected
recovery from false islands, and reward reconnecting volume rather than simply
pruning it. These additions refine access/thickness within existing families.
A local growth-front rule alone cannot guarantee global connectivity or fix
already disconnected inputs; those limits must be covered by the design.

Before paid execution, specify the exact architecture/decoder and losses, verify
its gradients and checkpoint recovery locally, and freeze one bounded training
comparison against raw NR5 and simple closing. Report raw learned and any decoded
outputs separately, including preservation, missing recovery, excess, all-nine
validity and time. This is a proposal, not implemented training or proven benefit.
No automatic NR6, additional seeds or paid job authorized by this diagnosis.

## Preservation and recovery

Script: scripts/diagnose_nr5.py (CLI --review and --output). Output directory is
exclusive; reruns require a new location. Original evidence untouched.
Tracked per-case report: experiments/reports/NR5-failure-diagnosis.json.
Full diagnostic arrays: Codex/outputs/NR5-Failure-Diagnosis-2026-09-28.
Local archive includes script, report, per-case masks and all three derived
fields, manifest hashes and documentation. Same disk, not off-device backup.
No Drive operation, remote push, public deployment or GPU usage.
