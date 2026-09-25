# MG6 frozen physical-scale protocol

2026-09-25, D070. Frozen before new MG6 generator outcomes. Generator MG5 and
MT1 evaluator unchanged; source/config/scene hashes bind the experiment.

Three designed sites per size: compact alignment, offset interfaces with a
partial obstacle, and a full separating-wall control. Exact coordinates are in
experiments/scenes/MG6-scale.json. Seeds6/7, request24%, cube2.4m, route cost12,
same nine-family MT1 settings. Six candidates at48; only a fully passing48 stage
admits the six64 candidates. These are development scale cases, not held-out
population inference. Preserve every failure; no seed replacement or retuning.

Both use0.8m/cell. World boxes are38.4m and51.2m per side. Building gaps increase
from32 to48 cells (25.6m to38.4m); opportunity region remains endpoint bounds plus
the existing6.4m Z/Y padding. Aligned and offset domains grow in physical volume.
Interfaces remain2cells/1.6m, ground band6cells/4.8m, full growth cubes3cells/2.4m.
Building mass means overall volume, with interiors/construction left for later.

Each stage first translates the saved MG5 aligned24%,seed0 field and its scene
in XY into the larger grid, retaining Z. Require identical translated domain,
independent scores and gross volume. This tests representation/evaluation only.
Do not claim seeded generator translation equivariance: grid-sized RNG arrays
change the noise values assigned to a translated location. No generated embedding
case is substituted for the actual larger-site candidates.

Context config explicitly sets grid_size/street_levels from scene, preserving
historical channel definitions. Verify actual state against scene. Weights are
loaded for existing config/provenance but never used in generation. Preserve
the legacy refusal of non-two-cell entrances: finer resolution remains separate.
Save exact masks in lossless NPZ, scenes/config/units in JSON, and hash attachments.

Admission at EACH size: all6 completed,4 nonpartition pass MT1 and requested count
with0..26 extra cells; both partitions finish no_cube_route and fail evaluation;
embedding domain/scores agree; no execution errors, generator timeouts or observed
resource cap breaches. Nonpartition does not imply feasible. Partial/failing
outputs remain evidence; completed scientific negatives are not execution errors.

Local CPU only,2 Torch threads, requested10ms native RSS sampling,2GiB observed
RSS stop. Cooperative per-candidate cap45s at48 and120s at64; study caps420s/900s
include audit/context/evaluation/saving. Check study/memory at case boundaries and
sampled peak after retaining each case. These are cooperative stopping thresholds,
not OS hard memory/deadline enforcement. No paid compute or silent timeout increase.

Measure context construction, generation, growth, setup/routing residual,
independent binary evaluation, saving/hashing, CPU and RSS. A scoped single-call
timing wrapper around grow_coverage restores the original symbol after generation;
its function body/source is unchanged. Residual includes input setup, Dijkstra and
radial initialization: do not label it isolated Dijkstra time. Capture source of
the wrapper. Actual worker/profile/serialization overhead is part of study time.
RSS is entire-process sampling and may miss short peaks. Record lifetime peak
separately. No matched speedup claim from old MG5 timing or from unequal sites.

Independently rebuild contexts, evaluate saved fields/bulk masks, replay completed
non-timeout cases with the unwrapped generator and audit cube unions/deltas and
physical volumes. For time-limited output, retain/audit the actual partial trace;
wall-clock cutoff cannot promise bitwise replay. Source hashes and every registered
artifact must verify. Record valid-only seed diversity with counts; never use
invalid fields to make diversity look better.

Explicit IDs link48 toMG5 and64 toadmitted48. Source snapshots, all raw arrays,
decisions, contexts, resource data and interruptions are immutable. Full project
regression is required. Update findings/decisions/resume, commit locally, then
verify incremental archive and Git-bundle restoration. Preserve prior archives.
No live promotion, new constraints, NCA training, Drive operation, push or hosting.
