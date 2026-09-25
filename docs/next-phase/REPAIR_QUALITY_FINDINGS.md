# NR3 readiness: first bounded repair-quality study prepared locally

2026-09-25. Local preparation passes. **No NR3 GPU job or model-quality study has
run.** The Studio remains the procedural MG7 workflow. Existing reports/model
sources and all prior experiment evidence are preserved.

## Concrete proposed experiment

Three fresh models, seeds1201/1202/1203, each256 updates of the unchanged NR1/NR2
architecture/loss/optimizer/16-step rollout. Same81 TRAIN variants,27 teachers,
shuffled sampler and32-cubed resolution. No architecture/objective/resolution sweep.
The verified NR2 T4 stack is the initial hardware/software admission target.

Each model is a separately approved invocation with a600s controlled-job cap;
maximum1800 controlled seconds across3 jobs. The broader proposal is at most60
minutes allocatedGPU time including setup/download/idle time, manually tracked.
This is a ceiling, not a predicted duration/cost or automatically capped bill.
No automatic next seed, retry, Drive mount, installation or long-run continuation.

Return ZIP+receipt after every seed. Verify locally and save/readback that exact
evidence in the sole Drive folder under a separately approved batch before the
next model. The notebook itself also needs separate save/verification and edit
scope. Runtime-only checkpoints can be lost before export; the bounded in-job
loss window and lack of certified cross-VM continuation must be accepted before
execution. This does not claim every intermediate update is already off-device.

## What improves the decision quality

The CPU evaluator requires three completed final-model archives with exact source/
recipe identity and fixes their checkpoint hashes before loading heldout examples.
No quality claim from a lone seed, partial run, attractive sample or reduced loss.
All-nine-family binary metrics, reconstruction errors, request errors and raw fields
are retained. Primary probability threshold>0.5 and horizon32 remain frozen.

Validation diagnostics use all27 rows at checkpoints0/64/128/192/256,32 steps and
firing2101:405 observations. Final evaluation uses all validation/test examples,
16/32/64 steps and three firing seeds:2187 observations. Primary TEST subset486.
The diagnostics schedule is now explicit; no intermediate best-step selection.
Evaluation is CPUfloat32 on the pinned local stack, so its output/timing is not
claimed identical to GPU inference. CPU weights-only inference is separate from
the GPU optimizer's strict same-runtime recovery identity. A CPU benchmark has
its own boundary cap and partial results do not satisfy admission.

For each trained model require median damaged TEST IoU improvement>=0.02 against
both unchanged and closing3, with all-nine pass rate no worse than either. Every
intact primary field must preserve all-nine validity and IoU>=0.99. Every accepted
damaged repair must satisfy absolute request error<=max(8,1% domain cells).
Missing/duplicate observations or a failed model seed cannot disappear in averages.
Firing seeds/related synthetic contexts are not independent model replications.

The archived comparator medians0.9042553191/0.9465109229 leave mathematical room
for the requested0.02 gain. This audits the old proposal using already-known NL0
comparator results; it does not observe learned TEST results or guarantee success.
MG7 regeneration and saved-teacher restoration remain information-rich references;
no NCA speed advantage is claimed. Matched runtime measurements of those references
remain a secondary analysis step before making any comparative performance claim.

## Validation completed

Focused 20260925T140118Z_93161ba36ba8:8 tests pass, including full-case gate accounting,
intact/request-error failures, per-model rejection, CPU evaluator parity, heldout
bundle rejection, disarmed commands and refusing missing final models.
Full regression 20260925T140149Z_2197b0f9fb6b:379 tests,0 failures/errors/skips,original-checkpoint
smoke0. No source changes occurred between these tests and the rehearsal; every
frozen source byte matches all three archived snapshots.

Isolated extracted-package rehearsal 20260925T140804Z_7d74952e7f95:8 CPU updates in 11.80s
of600s. This is explicitly rejected as a completed quality model by the evaluator.
All9 checkpoints0..8 match the preceding NR2 CPU control in model/Adam/RNG/sampler/
cursor/trace state; only the declared study/source identity differs. All8 pre-update
raw rollout states match exactly. Worker cleanup reports zero active descendants.
The integration probe's untrained32-step TRAIN geometry and all MT1/repair metrics
exactly reproduce the archived NL0 unchanged comparator. No heldout forward pass
or256-update job occurred during readiness validation. Before registration, the
first local wrapper failed because the imported Colab run was not a local RunStore
parent. No run was allocated and no training started. The helper now links the
registered NR2 CPU control while retaining external GPU provenance in the recipe.
The original expression and error are preserved in Codex outputs/nr3-parent-link-failure.
This changes no scientific source/package bytes. No registered NR3 experiment failed.

This tests the new driver, packaging and metric integration, not256-step stability,
learning quality or all production Colab UI interactions. The actual GPU job still
needs approval and may fail; preserve every result rather than weakening criteria.

## Files, provenance and next action

Ready directory: Codex outputs/NR3-Quality-Study. Files: NCA-NR3-Quality-Package.zip,
NCA-NR3-Quality-Study.ipynb (disarmed), START-HERE.md and package-receipt.json.
ZIP contains100 payloads plusmanifest,81 TRAIN arrays,zero heldout examples.
ZIP SHA256 fa8278c4b358bbdf3825986b511ee0e3ed9baab81ebfa986238f23ba3f31d61f.
Manifest SHA256 7234b6d0dcc47959b2a91e59141521ca743b153c06f8c4b6a6f732a678ebb6e7.
Notebook SHA256 96a667c23675580b5ae326a38b00e0a9c97f6a061147bbd21fb5ba5439fdf1f4.

Frozen config: experiments/configs/NR3-quality.json; primary rule implementation:
nca/repair_quality.py. Protocol: REPAIR_QUALITY_PROTOCOL.md. Local evaluation command:
project .venv Python scripts/evaluate_repair_quality.py --archives <seed1201.zip>
<seed1202.zip> <seed1203.zip> --manifest-sha256 <receipt manifest hash> --mode
diagnostics|final --output <fresh local folder>. Do not run this until all three
final archives and their adjacent receipts are verified. Diagnostics and final
each preserve their own source snapshot, sealed model list and every observation.

Raw preparation runs: .local-artifacts/runs/<IDs above>. Rehearsal extraction:
Codex outputs/nr3-rehearsals/20260925T140804Z_7d74952e7f95. Summary: NR3-readiness.json.
No cloud operations occurred. First ask for exact new-notebook save/readback scope,
then approval of the concrete seed1201 job and backup arrangement. Subsequent seeds
remain separately gated by verified backups and explicit compute authorization.
All scientific source/control files retained. Local archive receipt is authoritative;
keep NR2 GPU/Drive/full archives and the earlier chain. Same-disk backup is not cloud.
