# G9 — ready for one approved Colab run

Purpose: teach connection-building cubes to receive higher scores while retaining
G8's volume losses and pacing. Improvement is unproven.

1. Open NCA-G9-Access-Ranking.ipynb in Colab and select a Tesla T4 runtime.
2. Run the upload cell and select NCA-G9-Access-Ranking-Package.zip (196,362 bytes).
3. After approval of this exact allowance, change APPROVED_G9_JOB=False to True
   and run the training cell once.
4. Download the full results ZIP and matching receipt, including after failure.
5. Send both back here and disconnect the runtime after downloads finish.

Proposed allowance: ONE fresh seed1201 job,427updates64steps,maximum600 controlled
seconds; setup/export/download/idle extra. No automatic retry. G8 took336s;
G9 adds work and completion within the cap is not guaranteed. Do not bypass the
runtime guard if Colab changes. No Drive mounting.

Local checks passed:3retained updates, two exact checkpoint recoveries,
unchanged initialization/start schedule/random streams, finite gradients,
six fixed-weight64/128-step TRAIN inference comparisons, three exact
base-loss/forward comparisons. Rehearsal took37.84s.
This verifies engineering behavior, not improved model quality or CUDA parity.
GPU probes/recovery run inside the same bounded job.

Review only final427. Compare against45 prior requests and12 newly frozen cases;
G8 and G9 both run on those same new cases. Keep all old thresholds.

Package SHA256: e8ceb9e4445e10102f72d327109fe34a5d353ab6511fcba7a6852facd6d9bcd7
Manifest SHA256: bca826c8d13b603d5ea82ea481e26675ab9758ee382d5d42984def67df28e6c1

Everything remains local. Repository synchronization and off-device backup
remain pending. MG7 stays live.
