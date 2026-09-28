# NR5: one horizon-alignment trial (disarmed)

One proposed fresh seed1201,256 optimizer updates,32 NCA steps per update,
600s maximum controlled job time. NR4 used16 steps; final review remains32.
Same loss, architecture,32-cubed grid,81 TRAIN rows,Adam.001 and nine families.
This changes compute per update and stochastic firing consumption; not a matched
FLOP experiment. Approximately twice the rollout compute/memory is possible;
actual GPU time/memory is unmeasured. No AMP, accumulation or automatic fallback.

The code audit establishes a train/review horizon mismatch, not its causal role
in excess growth. Longer rollout supervision is a fixed hypothesis. No claim of
stability beyond32 steps, attractor convergence or reliable new-scene generation.
Fresh training from seed1201, not fine-tuning NR4. Preserve previous models.

Open NCA-NR5-Horizon.ipynb in Colab with a Tesla T4. Upload only
NCA-NR5-Horizon-Package.zip. Keep APPROVED_SEED_JOB=False until this exact single
job is approved. After approval set it True, keep MODEL_SEED=1201, run once.
Download ZIP and receipt, including ordinary failures; send both for verification.
No next seed/retry or deleting markers. Setup/export/idle time is outside timer.
Disconnect idle GPU after download verification. Whole-VM loss before download
can lose the job. Device-bound checkpoints do not promise cross-VM recovery.
No Drive operations/mount/sync or assistant notebook saves authorized; ask per action.

Review only final256 on all27 existing VALIDATION rows at32steps/firing2101 on
CPUfloat32. Compare NR3,NR4,unchanged and closing3. No TEST, additional horizons,
checkpoint selection, threshold tuning or automatic promotion. These cases have
informed the proposal and are development evidence, not independent validation.
Require every intact case IoU>=.99 and all9 valid; damaged medianIoU>=NR4,
all-nine passes>=17/18 (recover NR3 validity), false additions<1354 and median
absolute requested-volume error<=71cells. Report all outcomes; no automatic retry.
