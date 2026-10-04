# G2: one controlled change to seed generation

G1 completed its training but passed 0/9 development cases. A matched TRAIN diagnosis found conservative early growth and substantially better continuation from larger teacher stages. G2 tests one specific hypothesis: equal weighting of correct and incorrect frontier additions may improve seed growth.

## Change and verification

Positive frontier loss coefficient: 0.5 to 1.0. Negative coefficient stays 1.0 and local cube-volume coefficient stays 0.25. The network, initialization seed, all 27 array payloads, sampler, teacher-stage schedule, optimizer, 64-step rollout, 256-update budget and evaluation criteria remain unchanged. Start fresh; do not resume G1 weights.

The local loss check verifies the exact expected derivative change for positive logits, unchanged negative/inactive derivatives and unchanged volume term. It covers mixed classes, positive-only, negative-only and empty eligibility. It also checks identical initial parameter tensors and inference states before training, correct new checkpoint semantics and rejection of a G1 checkpoint by a G2 session. These are engineering checks, not evidence of better generated volumes.

The combined local rehearsal performs three retained updates and two exact checkpoint replays, including a teacher-stage update and a seed-start update. Final results and hash checks are recorded in VERIFICATION.json. GPU recovery is rechecked inside the proposed job, not assumed from CPU success.

## Proposed job

One T4 run, seed1201, 256 retained updates plus two recovery replays, 64 growth steps, batch1 float32. Maximum600 controlled seconds including startup, checkpoint writes and recovery. Setup, export, downloads and idle time are extra. Stop on runtime mismatch, recovery failure, nonfinite values or memory cap. No automatic retry.

Use NCA-G2-Balanced-Growth.ipynb and NCA-G2-Balanced-Growth-Package.zip. Upload accepts any filename but verifies exact bytes and portable row paths. Keep APPROVED_G2_JOB=False until this exact job is approved; after approval set True and run once. Return the full result ZIP and receipt even if the job stops.

PROTOCOL.md freezes the final256, single-seed, 64-step review on the same nine development requests and the additional fixed128-step stability check. All original validity, volume-error and stability gates remain. Higher growth may create excess mass or instability; those failures must be reported. Development has been reused; reserved labels remain unopened. The preserved G1 run is the comparison baseline, so no extra control training job is proposed.

## Scope and next action

No G2 quality result exists yet. No paid GPU run, Drive operation, live-model replacement, repository commit, push or deployment occurred. MG7 stays live. After verification, request approval for this one job; do not add more local rehearsals without a specific change or failure.

All implementation, checks, rehearsal evidence, packaging source and resume records are saved locally. Repository sync remains pending because write access was not granted. The milestone archive is verified but remains on the same disk. Earlier G1 successes and failures are preserved separately.
