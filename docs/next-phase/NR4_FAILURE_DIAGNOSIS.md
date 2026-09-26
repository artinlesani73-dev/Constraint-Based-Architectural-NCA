# NR4 saved-field diagnosis

## Current: NR4 failure diagnosis and paired Studio view complete - 2026-09-26

User requested faster work in larger batches and approved diagnosis, comparison
UI and a training revision only if justified. This batch is complete.

All8 NR4 failures retain both raw and bulk interface hits; bulk_unreached=0.
Each has1-2 raw disconnected occupied cells,13 total across8 example observations.
All13 are excess relative to target; none is a target cell. Six examples have
one geometrically unsupported cell each. One cube-damage example has bulk
fraction0.8916667 below0.90. These are per-example counts, not13 independent
spatial events. Main route/bulk connectivity did not fail. Earlier broad wording
about worsened connectivity must be read with this localization. Strict all-nine
failure remains real; do not soften thresholds or relabel these as passes.

Added /static/repair-comparison/index.html with all27 saved cases: input/target,
NR3/NR4 side by side, and closing3. Includes eight-failure filter, NR4 disconnected
cell highlight, common camera/slices/context, recorded metrics and per-case
explanation. Original NR3 study retained; added navigation links only. Download
contains both model identities and source array hashes. MG7 stays default.

Exporter scripts/export_repair_comparison.py validates NR4 raw arrays and
threshold parity, reproduces all27 archived MT1 reports and reconstruction
metrics, and exports diagnosis masks. No model rollout, training, new damage,
TEST evaluation, cleanup operation or parameter search. UI loads without console
errors; eight-option filter and magenta defect highlight visually checked.

D082: no NR5 package yet. Geometry identifies where failure occurs, not the
learning mechanism producing it. Disconnected outliers explain access failures
but not736 false additions on intact examples or sub-.99 intact IoU. Blindly
increasing penalties again is not justified. Deleting islands might address some
symptoms but is not a learned correction and was not performed. No new loss,
architecture, grid size or paid run has been authorized or prepared here.

Next substantial task: review the training design against the preservation
failure before choosing one intervention; distinguish16-step training/32-step
review stability from spatial-error weighting. This batch does not establish
which is causal, and does not authorize a horizon sweep. Continue in coherent
batches, avoid repeated confirmation for local implementation, and still ask
before paid compute or every Drive action. No Drive, push or publication.

Evidence: deploy/static/repair-comparison/diagnosis.json and study.json;
Codex outputs/NR4-Failure-Diagnosis contains preserved diagnosis and verified
batch archive. Old raw NR3/NR4 model evidence is unchanged. Same-disk archive
is not off-device backup. Resume from this entry; prior current entries are history.

