# G8 — ready for one approved run

Purpose: keep G7 fixed and increase training to427 updates (9-10 visits per
example), testing the reduced-exposure hypothesis. Improvement is unproven.

1. Open NCA-G8-Exposure.ipynb in Colab; choose Tesla T4.
2. Upload NCA-G8-Exposure-Package.zip when prompted (190,533 bytes).
3. After explicit approval for this run, set APPROVED_G8_JOB=True and run once.
4. Download the full evidence ZIP and receipt, even after a failure. Share both.
5. Disconnect the runtime after downloads complete to avoid idle compute.

Proposed allowance: one fresh seed1201 T4 job,427 updates64 steps,max600
controlled seconds. Setup, export, downloads and idle are extra. No retries.
Strict runtime checks and two recovery replays remain. No Drive mounting.

Package SHA256: 333ea0e2abf4d51350d7be82034e12b59ede70524b3bbe0ddf4d32263de9e6fc
Manifest SHA256: 936feb233294d46445260f7012b01aeaba20fb61105d2f580ec92b714fcab146

Local rehearsal: passed in57.23s; two exact recoveries; G7/G8
update0 andupdate3 numerical payloads identical. This is engineering verification,
not evidence of improved G8 quality or a substitute for CUDA checks.

Review only final427 against33 regression and12 fresh reserved requests at64/128.
Use frozen-review.json; never select a better intermediate checkpoint. Check256
prefix reproducibility separately. MG7 remains live; repository sync is pending.
