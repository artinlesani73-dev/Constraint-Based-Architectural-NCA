# G10 — one-sided ranking, ready for approval

1. Open NCA-G10-One-Sided.ipynb in Colab; select Tesla T4.
2. Upload NCA-G10-One-Sided-Package.zip when prompted (196,404 bytes).
3. After approval for this exact job, set APPROVED_G10_JOB=True and run once.
4. Download the complete results ZIP and receipt, including after any failure.
5. Send both here; disconnect the runtime after downloading.

Proposed allowance: ONE fresh seed1201 T4 run,427updates64steps,max600 controlled
seconds. Setup/export/download/idle extra. No automatic retry or cap extension.
G9 took469s; G10 completion within600s is not guaranteed. Runtime mismatch stops
before training; do not bypass it. No Drive mounting.

Only change: ranking raises advancing scores without directly lowering other
teacher-positive scores. Margin1/weight1,base losses,architecture,data,pacing and
training exposure unchanged. This is an explicit semi-gradient, not a guarantee
other scores remain unchanged through shared network parameters.

Local3-update rehearsal passed in40.83s; two exact recoveries.
Initial numerical payload,start choices and random streams match G9; trained
weights differ. Fixed-weight inference parity,loss decomposition,semi-gradient
behavior and finite gradients passed. No model-quality or CUDA parity claim.

Frozen review:57 regression requests plus12 new reserved requests at64/128.
G9 is the primary paired baseline; G8 is the stability reference; both evaluated
on the same fresh12 as G10. All original gates retained; final427 only.

Package SHA256: 62d54b7e0efc1638086027766a6eab149c0134cf360c8dd4958326f2bca4faa3
Manifest SHA256: 193c0cfc5b10cede58a7671453ce26bdd54cd9a11c21fe7fbcd7f5a18926fc07

Documented and archived locally. Repository sync and off-device backup pending.
MG7 remains live. No paid job,Drive,push,publication or live swap occurred.
