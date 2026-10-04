# CGR3 connected constructive repair (DISARMED)

One proposed fresh seed1201,256 updates,32 growth steps each,600s controlled
job cap.32-cubed grid,81TRAIN rows only. Float32 Adam.001. No TEST or validation
rows in package. Hard six-face frontier, monotonic accepted occupancy and per-step
frontier/local-volume supervision: see CGR3_CURRICULUM_SPEC.md in repository.
No reliable learned performance claim; MG7 remains the live generator.

Open NCA-CGR3-Curriculum.ipynb in Colab; upload only NCA-CGR3-Curriculum-Package.zip.
Leave APPROVED_SEED_JOB=False until user explicitly approves this one job.
Verified prior GPU stack required: TeslaT4,Python3.13.15,Torch2.11.0+cu130,
NumPy2.1.3,CUDA13.0,cuDNN92700. A mismatch stops; do not override or retry.
No automatic install, paid launch, retry, extra seed or Drive operation.
Two-update deterministic GPU backward and exact augmented-step recovery verified on this stack; full trial quality is untested.

After approval, set the gate True and run once. Download ZIP AND receipt,
including failures. Return both for verification. Local archives are same-disk;
no off-device backup is claimed. Whole-VM loss before download can lose evidence.
Device-bound checkpoints do not promise cross-runtime recovery. Setup/download/
idle time are outside the600s controller cap. Disconnect idle compute after
verified download. An owned process watchdog stops at the cap; partial evidence
is retained. Do not delete attempt markers to rerun.

Evaluation is separate: final256 only,27 existing development rows,32steps,
firing2101,CPUfloat32. Evaluate accepted occupancy, not proposal>0.5. Retain raw
proposal history, birth masks, hidden state and initial/final occupancy. No
postprocessing or threshold tuning. Compare saved rawNR5 and closing3. Follow
frozen criteria in CGR3_CURRICULUM_SPEC.md; no automatic model admission.

CGR3 changes training starts only versus CGR1. Original-input evaluation is unchanged.
Visit counters and each start hash are saved with checkpoint history; start arrays
are exported per update. The eight-update rehearsal uses first visits (original
starts); the focused recovery test explicitly crosses an augmented second visit.

Runtime update 2026-10-03: compatibility run20261003T094720Z_e18428c17993 passed.
Fresh256-update trial only after approval. Training math,data,seed and curriculum
unchanged; runtime changed versus previous CGR1/CGR2,so comparisons cannot isolate
curriculum effects perfectly. No cross-runtime numerical equivalence is claimed.
Do not resume a compatibility checkpoint or rerun the old cu128 package.
