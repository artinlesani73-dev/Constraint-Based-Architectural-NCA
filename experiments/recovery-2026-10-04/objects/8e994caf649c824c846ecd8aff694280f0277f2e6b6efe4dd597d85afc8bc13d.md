# G4 implementation and next run

G4 now proposes complete overlapping3x3x3 cubes. This addresses the thin additions
observed in G3 while keeping overall building mass and the same nine families.
The existing27TRAIN conditions,seed values and target fields are unchanged.
New teacher starts are connected unions of cubes; inference starts from the
same scene-defined single voxel. Origin firing changes the random-number stream.
See PROTOCOL.md for the complete frozen algorithm,losses,budget and review gates.

The differentiable union accounts for overlapping proposals without summing the
same empty voxel repeatedly. It is an independent-proposal surrogate before hard
ranking/cap,not the expected output of the constrained transition. First-seed
steps use only origin classification loss because at most one cube is admitted.
The budget band remains unchanged. A leftover gap too small for a full cube can
stall growth. Enforced thickness,legality and budget do not prove learned quality.

Hard admission currently runs on CPU,with one probability/firing transfer per
step. Measured CPU eligibility+admission median 0.851ms,
max 1.076ms over20synthetic32cubed cases.
These timings exclude GPU transfer and do not predict complete GPU job time.
The fixed600s supervisor remains the limit; partial/failure evidence is retained.

Local verification passed analytic two-cube union probabilities,finite gradients
including extreme logits,correct band-gradient direction,seed-only loss,27
independently reconstructed TRAIN cube graphs,12 reference cases,20 crowded
budget cases,connected/bulk/legal invariant checks and invalid-start rejection.
G3 checkpoints are rejected by identity. Fresh model parameters equal the G3
initialization,so G4 does not silently import trained weights.

The final packaged CPU rehearsal 20261003T214142Z_6c0f9e5d3f06 completed3retained updates
and2full-payload/state recovery replays in 30.718s controlled time.
All23evidence payload hashes verified. All192retained step accounts and saved
start hashes verified. Recovery covers a teacher-cube start and a true seed start.
These are engineering checks,not a G4 quality benchmark. No held-out or reserved
evaluation and no paid GPU work occurred. GPU compatibility is checked inside
the proposed job; it has not yet been demonstrated for this new operator.

The first package/rehearsal is preserved in G4-Block-Training-2026-10-03. It passed; the v2 package
removes an unused inherited intact-negative loss setting from metadata so the
published objective exactly matches execution. No model math changed. Model
weights after each of the3updates match between the two rehearsals. Use v2 only.

Next action: approve one Tesla T4 job,seed1201,256updates64steps,maximum600
controlled seconds. Setup,export and idle time are extra. No automatic retry.
Open NCA-G4-Block.ipynb in Colab and upload NCA-G4-Block-Package.zip from THIS
folder. The notebook approval flag remains False. After explicit approval,
set APPROVED_G4_JOB=True,run once,and return FULL evidence ZIP plus receipt.
Do not upload the rehearsal evidence or previous G3/G4-v1 package in its place.

Results,changes and resumption instructions are saved locally here. Repository
sync remains pending because this session has no granted write access to the
project checkout. Its older RESUME is stale; use this folder's RESUME.json.
The milestone ZIP is a verified same-disk archive,not an off-device backup.
No Drive operation,push,publication or live-model replacement was performed.
