# G6 handoff

G6 changes how quickly cubes are admitted. It uses G4's model and losses with
the same fresh initialization; no G5 distance inputs or trained warm start.
Read C:\Users\artin\Documents\Codex\outputs\G6-Objective-Audit-2026-10-04\FINDINGS.md for the evidence and limitations.

The fixed allowance is K=max(9,ceil((C-27)/63)) new voxels per step after the
first seed-containing cube. The global volume ceiling and nine families remain
unchanged. Quota stays fixed in128step evaluation. This does not guarantee
connection,facade compliance,requested size or stability.

The independent TRAIN diagnostic improved all-nine validity from2/27to12/27
at64steps,using unchanged G4weights. Facade compliance worsened and only18/27
cases were stable. This is preliminary TRAIN evidence,not trained G6performance.

Local package/trajectory comparison and exact recovery passed. Rehearsal:
20261004T080937Z_e82f41f9d831,3updates plus two replays,33.187controlledseconds.
Initial parameters and row/start/firing schedules match G4. See PROTOCOL.md.

After explicit approval for one T4job,256updates64steps,maximum600controlled
seconds,open NCA-G6-Paced.ipynb in Colab and upload NCA-G6-Paced-Package.zip.
Set APPROVED_G6_JOB=True only after that approval;distributed flag isFalse.
Setup/export/download/idle extra. Run once. Return full evidenceZIP+receipt,
including failures. No automatic retry,Drive operation or model promotion.

Results and resume records are locally saved. Repository synchronization and
off-device backup remain pending. MG7 stays live.
