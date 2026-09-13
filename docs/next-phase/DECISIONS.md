# Decision register

Append new decisions or explicit superseding entries; preserve earlier rationale.

## D001 - Keep the original evidence and isolate new implementation

Date: 2026-09-13. Status: accepted. Source: user request.

Work on `next-phase/foundations`. Preserve the original notebook, checkpoint, history and evaluation. Source snapshots include dirty working-tree code so a result can be reconstructed even before a commit. The private report remains ignored. Existing results are historical evidence, not validation of repaired code.

## D002 - Track small records in Git; archive full experiment payloads

Date: 2026-09-13. Status: accepted. Source: user requirement to retain all results and decisions.

Commit summaries under `experiments/records/`; keep full fields, checkpoints, images, logs and snapshots in `.local-artifacts/runs/` or a configured external root. Do not overwrite run metadata or final outcomes. Retry/resume attempts get new IDs linked to the earlier run. Include negative results. Source/control records are append-only by tool convention; file permissions and a remote backup are still needed against manual deletion or device loss.

## D003 - Google Drive plus local archive

Date: 2026-09-13. Status: accepted policy; remote setup pending. Source: user's explicit choice.

Suggested Drive root: `MyDrive/NCA-Next-Phase/`. Keep each complete run folder and verify hashes after copying. Test recovery before paid training. The local before-change snapshot is not an independent backup; cloud synchronization is not assumed to have succeeded merely because a path exists.

## D004 - Do not spend compute before the experiment is defined

Date: 2026-09-13. Status: accepted working rule; numeric allowance pending.

The user has paid Colab, but the pilot cap has not yet been confirmed. Prepare E0 and recovery checks locally. Ask the user to run/sign into Colab only when a concrete notebook and manifest are ready. There is no current paid training job.

## D005 - Independent metrics precede replacement training losses

Date: 2026-09-13. Status: accepted implementation order.

`binary_v1` evaluates explicit boolean arrays with named endpoint IDs and chosen neighborhoods. Empty material yields null material-normalized scores and an explicit nonempty flag. Eroded-core fraction is a voxel-scale proxy, not physical maximum thickness. These primitives are not yet a complete architectural validity gate and do not silently replace historical notebook scores.

## D006 - First serving fix does not silently change the learned rollout

Date: 2026-09-13. Status: accepted.

Fix checkpoint loading and missing optional facade metadata now. Keep current growth schedules, firing/noise, thresholding, legacy corridor operator and UI settings until E0 profiles are explicit. Bounds validation, per-job configuration, actual cancellation and the UI redesign remain planned. This permits attribution rather than combining unrelated behavior changes.
