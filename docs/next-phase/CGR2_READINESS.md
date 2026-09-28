# CGR2 readiness

## CGR2 objective revision ready; one GPU approval pending — 2026-09-28

D093: separate bulk_repair.py retains CGR1 architecture, seed, data,32steps and
256updates. Add0.5 squared target-full3-cube deficit; intact negative weight3
(previous1.5). Local bulk surrogate, not guaranteed connectivity. Prior code and
checkpoints preserved; semantic restore guard rejects CGR1 weights as resumable
CGR2 state. Five focused tests passed2.570s, including exact CPU recovery.
TRAIN81 gradient audit: finite gradients,54damaged rows nonzero,27intact zero;
no validation/TEST inspected. Eight-update CPU rehearsal completed26.547s,
run20260928T160237Z_2eabf7d1cf3a,41payload hashes verified, no active processes.
These are engineering checks, not trained-model quality or GPU recovery evidence.

Ready files: C:/Users/artin/Documents/Codex/outputs/CGR2-Bulk/
NCA-CGR2-Connected.ipynb and NCA-CGR2-Connected-Package.zip.
Package SHA2564acfe0d207e18f9c58df532a754cebfaebd1769a44689f67f2a402369162316e.
Manifest b3d15f8625637e75ba4b3a59aea8f243facb5d572b0653436acba90afe23b22a.
Upload accepts any filename for exactly one file, while enforcing the checksum.
See BULK_REPAIR_SPEC.md, BULK_REPAIR_PROTOCOL.md, experiments/reports/CGR2-readiness.json.
Next: obtain approval for one Colab T4 seed1201,256updates,32steps,600controlled
seconds (setup/idle/export extra). Gate remains False. Download ZIP+receipt
locally; no Drive operation or off-device backup claimed. No retry/extra seed.
Review final256 accepted occupancy using same27development rows and frozen gates,
report CGR1/NR5/closing3 comparison; no TEST or automatic admission. MG7 stays live.
Adapt review_connected_run.py to CGR2 semantic/module/manifest only after receiving
results; never change the historical CGR1 review or tune gates after evaluation.

