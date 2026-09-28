# CGR1 GPU compatibility correction

## CGR1 GPU failure and pooling compatibility fix - 2026-09-28

User supplied run20260928T113537Z_6f25fe81a893 ZIP+receipt. Verified outer SHA256
b63fada1350d1638e77950a316033562877f040d7de7ae32d844c5d1db4cd230 and all8payloads,
unique/exact archive membership and expected original package manifest. Run failed
at first loss.backward with ZEROcompleted updates: avg_pool3d_backward_cuda has
no deterministic implementation on the supplied stack.10.564s controlled time;
worker7.372s. Initial checkpoint/boundary only; no trained model or quality review.
This is an implementation compatibility failure, not evidence against the model
concept. CPU-only checks could not establish GPU backward compatibility.
No separate approval reply preceded the upload; record user-run execution only.
No new paid launch/retry permission inferred. Original ZIP/receipt/log/result
preserved in Codex outputs/CGR1-Failed-20260928T113537Z; tracked run record added.

Fix: valid3-cube mean via fixed1/27 conv3d kernel replaces avg_pool3d in the volume
loss. Same intended math, small floating-point differences possible. Strict
algorithm determinism stays enabled; no warn_only or nondeterministic fallback.
Semantic version advanced to connected_constructive_repair_v2 so checkpoints
cannot silently cross implementation versions. Original v1 source remains in
its immutable package and Git history. No loss-weight/data/seed/step changes.
Four focused CPU tests passed in1.907s, including value AND gradient parity to
avg_pool3d, exact checkpoint recovery and prior constructive-update invariants.
GPU compatibility of the replacement remains unverified until approved execution.

Corrected disarmed package: C:/Users/artin/Documents/Codex/outputs/CGR1-Connected-v2/
NCA-CGR1-Connected.ipynb and NCA-CGR1-Connected-Package.zip.
ZIP SHA25600fc2cc13b506b5b0962f1085dd29aaa03e441179b67b2eebf86f5e97e5f0a7e.
Manifest467da4659f9534e2491ba727ed9725a1d26b7bad917eae565c381b44e77bda61.
Replaces v1 for future use; do not upload old ZIP or use old notebook checksum.
CPU package rehearsal status will be recorded in CGR1-v2-readiness.json.
Next: request approval for ONE fresh bounded attempt on corrected package,
seed1201,256updates,32steps,600controlled seconds; setup/download/idle extra.
No automatic retries, Drive access, remote push or public deployment.
