# CGR1 implementation readiness

## CGR1 implemented and packaged; GPU approval pending - 2026-09-28

Separate nca/connected_repair.py implements D089: binary accepted occupancy plus
seven hidden channels;60->64->8 network; hard synchronous6-face births; monotonic
occupancy; stochastic0.5firing;32steps. Detached decisions, no straight-through
estimator. Per-step frontier loss plus0.25valid3-cube local-volume loss. Original
NR5 and other model implementations untouched. MG7 remains live.
ConnectedSession versions model semantics/objective in checkpoint identity so
same-shaped NR5 checkpoints fail restore. CPU synthetic test verifies next-step
state, trace and complete payload equality after restore, including optimizer,
sampler and RNG. Tests also cover simultaneous attachment, illegal/empty inputs,
no deletion, disconnected-input limitation, finite/nonzero loss/volume gradients,
and target-free inference equivalence.3tests passed in1.968s. Scalar-conversion
warning in test assertion only; no runtime failure. Synthetic checks are not
model-quality evidence. No new test families or acceptance threshold changes.

New package verifier/builder and owned-process runner preserve existing bounded
execution/export pattern. Fresh extraction, source hashes, TRAIN81 only, no
heldout rows. Eight-update CPU rehearsal20260928T095728Z_155b66e1b585 completed
in28.031s controlled time; worker25.687s; zero active owned processes afterward.
All41 exported evidence file hashes and outer receipt verified. Boundary exports
retain proposals at all32steps, birth masks, initial occupancy, final field and
hidden state. Per-update checkpoints, traces and states retained. No GPU quality,
GPU determinism, GPU recovery or600s-fit claim for this new architecture.

Package: C:/Users/artin/Documents/Codex/outputs/CGR1-Connected/
NCA-CGR1-Connected-Package.zip and NCA-CGR1-Connected.ipynb.
ZIP SHA25648310ad07b8a5a2f3f10c3bb07bb210cb7c6fc683a6eecdfc2380b394cebfea6.
Manifest6dc58940d03349c34f50ebafcf26880cf97edbc3e892e565d412e35387a8a3c3.
Readiness details: experiments/reports/CGR1-readiness.json.
The packaged CONNECTED_REPAIR_SPEC.md is the original design-stage specification;
its pending-status paragraph is historical. This entry and readiness report state
the current implemented/prepared status. CGR1-proposal.json likewise remains the
original disarmed proposal, not a run approval record.

NEXT USER ACTION: approve ONE seed1201 GPU run,256updates x32steps, max600controlled
seconds (setup/download/idle extra), on the frozen T4 software stack. Notebook
APPROVED_SEED_JOB remainsFalse until approval. No automatic run/retry/additional
seed, no Drive operations. No GPU launch by assistant. After approval user opens
notebook, uploads ZIP, runs once and returns output ZIP+receipt including failures.
Final review only27development rows at32steps/firing2101; accepted field is output,
NOT proposal>0.5. Preserve both and use frozen criteria in specification.
Full rehearsal and source/docs archive local; same-disk only. No push/publication.
