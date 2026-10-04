# G4 final review — 2026-10-04

G4 completed successfully, but does not meet the frozen pilot acceptance gates.
Two of nine reused development requests pass all nine families at both64 and128
steps. The next research issue is spatial allocation before capacity is consumed.
MG7 remains the live model. No replacement or new paid run was performed.

## Verified run

User-supplied run20261004T065608Z_176639ac1bc5 completed256updates on Tesla T4.
All1035evidence payload hashes,unique archive membership and exact G4-v2 package
identity verified. Controlled time 180.071s;
worker 173.586s;
peak reserved GPU memory 1446MiB.
The12reference/device cases and differentiable-union backward check passed.
Full-payload and state recovery matched exactly at updates2 and3.

All256saved start fields and hashes match the frozen seed/cube-stage schedule:
128seed starts,128cube teacher stages. Sampler order,all16384step-count accounts,
final optimizer step counts and each retained output's legality,connectivity,
seed retention and complete-cube support verified. Count verification is not
a replay of every training rollout. Full original ZIP and receipt are preserved.
No assistant-launched paid job or automatic retry occurred during this review.

## Frozen comparison

Same nine development requests,final checkpoint256only,CPUfloat32,firing2101.
No postprocessing,threshold search,checkpoint selection or reserved evaluation.

| Check | G3 at64 | G4 at64 | G4 at128 |
|---|---:|---:|---:|
| All nine families pass |1/9|2/9|2/9|
| Access |5/9|2/9|2/9|
| Coverage |7/9|6/9|6/9|
| Thickness |1/9|9/9|9/9|
| Each of facade,ground,legality,sparsity,spill,support |9/9|9/9|9/9|
| Median volume-fraction error,percentage points |4.452|0.152|0.152|
| Maximum volume-fraction error,percentage points |8.255|0.166|0.166|
| Median teacher IoU |0.5354|0.4575|0.4575|

All nine G4fields are identical between64 and128steps; G3 failed the5%stability
gate in all nine. G4passes3of5gates:median/max volume error and stability. It
fails all-nine validity at both horizons. Both passing cases request32%volume
atYoffset variants2and4; the other32%case still fails access.

This is a partial improvement with regressions in access,coverage and teacher
overlap. Complete-cube growth enforces100%bulk support here. The hard cap tightly
bounds mass and explains saturation stability; these are not proof that the
network learned thickness or a self-stabilizing dynamical rule independently.
The integrated change also alters training stages,proposal semantics,firing RNG
and the volume surrogate,so its effects cannot be attributed to one component.

## Why the remaining cases fail

Every output physically touches the west interface and retains the seed. Only
two reach the east interface. All nine exhaust capacity:the first exact cap
hit occurs between steps14and22.
Because additions are irreversible,remaining proposals cannot move already
allocated volume toward the missing interface. Doubling the rollout does not
repair this. All16%requests also underfill the farXthird under the unchanged
coverage metric. Saved geometry supports this diagnosis; it does not identify
which loss or input feature is solely responsible.

| Case | Occupied voxels | First cap step | Legal voxel moves to east | Failure |
|---|---:|---:|---:|---|
| y0-v16 | 876 | 17 | 5 | access, coverage |
| y0-v24 | 1310 | 19 | 2 | access |
| y0-v32 | 1744 | 20 | 2 | access |
| y2-v16 | 874 | 14 | 4 | access, coverage |
| y2-v24 | 1307 | 17 | 1 | access |
| y2-v32 | 1740 | 19 | 0 | PASS |
| y4-v16 | 871 | 16 | 5 | access, coverage |
| y4-v24 | 1302 | 20 | 3 | access |
| y4-v32 | 1733 | 22 | 0 | PASS |

Existing evaluator interface-hit flags measure reachability from alphabetically
first endpoint E_east. If east is untouched,both reported flags are false,even
though west contact exists. allocation-diagnostics.json adds direct contact
counts to avoid misreading that result; the frozen score itself is unchanged.
Graph distances above are diagnostic voxel moves,not whole-cube repair costs.
The all-offer pre-admission candidate is not an unguarded rollout; its seed-step
form can contain multiple cubes. Do not claim it isolates the cap's causal effect.

## Next bounded development step

Retain G4's cube representation and current budget for now. Before another paid
run,design a teacher-independent cue that lets proposals account for the distant
interface early,using the existing access family. A candidate is a legal
cube-origin distance field to the opposite interface,derived from context only.
Audit representability and finite values on TRAIN geometries and check whether
the cue distinguishes useful advance from lateral expansion. Do not encode a
teacher route,hardcode held-out cases,or introduce a new constraint category.

This is a proposed direction,not an implemented or frozen G5protocol. Freeze
one focused change and its effective configuration before training. Keep the
same acceptance thresholds and disclose repeated development-set reuse. Do not
increase grid size,run duration,or add a second seed merely to search for a pass.
Reserved cases remain available for later generalization review.

All evidence,decisions and resumption instructions are local in this folder.
Repository synchronization remains pending:the project checkout's older RESUME
is stale. This archive is on the same disk,not an off-device backup. No Drive,
push,publication or live-model promotion was performed.
