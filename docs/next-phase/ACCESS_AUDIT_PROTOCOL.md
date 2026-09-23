# A2: access semantics and actual-gradient audit

Frozen 2026-09-23 before execution. Config:A2-access.json. No optimizer updates.

The candidate contract is that ONE six-connected material component intersects
every entrance region. Disconnected components cannot pool their entrances to
pass. An unrelated source-region fragment is reported but does not invalidate a
different component that itself touches all entrances; this explicitly differs
from binary_v1's rejection of fragmented source regions.

`component_bottleneck_v2` strength is the maximum bottleneck occupancy of such a
component: activate permitted voxels in descending occupancy and merge neighbors
until one component touches every region. The strength is the critical voxel's
value; loss=1-strength. Thus strength>t exactly matches the independent binary
component predicate at material>t, for0<=t<1. Empty material scores loss1. If even
all permitted cells cannot connect the regions, return loss1 and infeasible.

Topology/order selection uses detached CPU values. The critical voxel is gathered
from the live tensor, giving a piecewise-linear derivative almost everywhere.
Ties select lexicographic z,y,x order deterministically; this is a chosen
subgradient, not a smooth function or proof of useful learning dynamics. A zero
critical voxel may still have a blocked derivative through the material clamp.
No GPU performance, recovery or implementation claim is made.

The candidate changes source selection, worst-versus-mean destination reduction,
and finite-versus-unbounded spatial propagation within the same access family.
Do not attribute all differences to the source change. Replay intermediate
definitions (fixed/mean/64, fixed/worst/64, fixed/worst/unbounded) for attribution.
Other eight families, regularizers, coefficients and material budgets stay fixed.

## Fixed matrix

-277 saved fields:56 F1 evaluations,187 K2 evaluations (including original and
 17 W1 controls),34 final D1 fields. Preserve source run/field/scene hashes.
- Check legacy access values against saved values, candidate strength against
  independent binary BFS, fixed-point material, old source fragmentation and
  unrestricted binary distances from the fixed point. Publish all disagreements.
-12 actual-model cases: original checkpoint on both difficult scenes at16/50
 growth steps; all four F1 final checkpoints at16/50. Firing seed2, matching saved
 fields exactly. Loading frozen weights for a diagnostic is not checkpoint resume.
- Store parameter vectors and last-raw-material derivatives for access_v1,
 access_v2, coverage, sparsity, total_v1 and total_v2. Use the member's recipe;
 original cases use mapped_30. Save norms/cosines, complete parameter layout and
 frozen-weight equality. Cosines involving zero vectors are null, never zero.
- CPU two threads, deterministic. Replay worker cap600s; each gradient worker
 cap120s; whole diagnostic cap900s. Failures/timeouts retain partial evidence.

## Preconditions and verification

Regression fixtures cover empty fixed corners, separated multiple origins,
an irrelevant source fragment, empty destinations, paths longer than64 edges,
multiple destinations, finite-difference derivatives, ties and randomized
threshold equivalence against independent BFS. Ground/facade frozen scenes are
included in the277-field replay. Run full regression before the audit.

Before publication check registered hashes/source snapshots, all gradient norms
and cosines from saved vectors, saved forward equality and all replay counts.
No candidate promotion or training follows automatically. Use the findings to
decide whether one access-only intervention is justified and how to profile it.
All prior records remain labelled with their original definitions. No Drive,
paid compute, source checkpoint overwrite, deployment or push.
