# CGR3 final review — 2026-10-03

CGR3 completed, but failed two frozen acceptance conditions. Keep CGR1 as the
experimental reference and MG7 live. Do not promote this checkpoint or launch
another training trial automatically.

## Verified evidence

Run20261003T102352Z_a40c9f968e66 completed256updates in59.970controlled seconds,
seed1201,T4,peak reserved710MiB,exit0. Verified original receipt hash and all1036
payloads,exact unique membership,final checkpoint digest/identity/cursors and Adam
step counts. Independently matched all256 saved start arrays to the frozen TRAIN
schedule and verified their hashes,per-update traces,source row hashes and visit
history:194original,62intermediate. No incomplete-download or curriculum mismatch.

Runtime is Torch2.11.0+cu130/CUDA13.0/cuDNN92700. The prior17-payload compatibility
run20261003T094720Z_e18428c17993 independently passed full-payload augmented-step
recovery. That is same-runtime recovery evidence, not CUDA12.8/13.0 equivalence.
CGR1/CGR2 were trained on cu128, so runtime is a confound in curriculum attribution.

Frozen evaluation: final256,CPUfloat32,32steps,firing2101,27existing development
examples using original inputs only. No teacher augmentation at evaluation,
threshold tuning,cleanup,TEST or checkpoint/horizon selection. Preserved proposals,
births,states and all case metrics. Verified81prior NR5/CGR1/CGR2 arrays and matched
case/damage/baseline identities. One seed,reused development scenes; not generalization.

## Comparison

| Metric | CGR1 | CGR2 | CGR3 |
|---|---:|---:|---:|
| All nine pass, all27 |24|24|24|
| All nine pass, damaged18 |15|15|15|
| Damaged median IoU |.97504|.97096|.97214|
| Damaged recovered cells |1959|1828|1949|
| Damaged excess cells |245|165|268|
| Damaged median absolute volume error |15.5|23.5|10.5|
| Intact excess cells |117|83|139|
| Intact median IoU |.99426|.99316|.98790|

CGR3 recovers most of CGR1's repaired volume and improves requested-volume error,
but increases excess and worsens intact overlap. Lower volume error can coexist
with worse shape agreement, because missing and excess cells can offset in a
volume total. Closing3 also passes24/27,with damaged IoU.97268,recovered1464,
excess13 and volume error38; CGR3 is not a universal winner over this simple baseline.

## Frozen decision and actual failure changes

Six of eight gates pass. Damaged validity remains15/18 versus17required. Five
intact cases are below.99IoU (every intact case must pass). All9intact cases still
satisfy the nine geometric checks. No surviving input cells removed; no detached
voxels. Input preservation and attached growth are enforced, not learned.

Same three cube5 cases fail, but two individual family failures are resolved:
- v16,s0 now passes access and still fails thickness.
- v16,s2 still fails access and thickness.
- v24,s2 now passes thickness and still fails access.
Access and thickness each improve24/27 to25/27; all remaining families pass27/27.
The curriculum therefore shows a partial change, not full failure resolution.

## Recommendation

Close this bounded curriculum trial as mixed/not accepted. Do not infer that
more voxels,more rollout steps,or another small loss adjustment will solve it.
Before another paid job, consolidate the experiments into a model-design decision:
compare the current irreversible,detached birth rule with a trainable proposal
that can revise its own additions while preserving original input. This is a
candidate for analysis, not an approved architecture or proven solution. Examine
objective versus deployment-rule alignment and keep the nine families fixed.
These trials assess repair of known volumes; they do not demonstrate generation
of new architectural volumes from scene constraints. Keep that distinction in
the larger project roadmap. Retain CGR1 until a replacement meets the frozen gates.

## Records and downloads

User explicitly withdrew the small-review-ZIP proposal. Continue full ZIP exports;
no new split/review export workflow introduced. Original files and all evaluation
evidence are preserved here. Repository writes were not granted, so this record,
review script and pending resume instructions are local and not Git-committed.
No Drive access,new paid training,push or live-model change. Local copies are
same-disk archives,not an off-device backup.
