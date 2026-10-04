# G1 portability failure and corrected package

2026-10-03. Returned run: 20261003T172855Z_dafe9fb14084.

Verified original ZIP SHA256, receipt, exact unique membership and all four payload hashes. The request matches the original G1 package manifest. The worker failed while loading its first example, before writing an identity or checkpoint: zero completed updates, 10.739 controlled seconds. No learned-quality or GPU-recovery evaluation is possible. This is a packaging failure, not a model-quality failure. Full original evidence is preserved here.

The dataset metadata used Windows backslashes. On Linux the loader looked for a filename containing those backslashes, although the ZIP members used forward slashes. The Windows rehearsal did not expose the mismatch. This was an assistant-authored packaging mistake.

The replacement package canonicalizes all 27 row paths to forward slashes and strengthens prelaunch verification: row paths must be canonical relative POSIX paths, and each must match a manifest member and digest. Only data.json and nca/generation_package.py changed among the original payloads. The manifest and notebook package checksum were regenerated. All arrays, model code, objective, curriculum, runner and frozen protocol remain byte-identical. Historical packages and evidence are unchanged.

Verification covers all replacement payload hashes, exact ZIP-member lookup for every row independent of Windows path interpretation, rejection of the original bad path style, all 27 rows through the production loader, and notebook-cell syntax. Each input still has exactly one occupied seed. This does not claim an executed Linux or GPU rehearsal. Because training code and arrays did not change, the prior CPU exact-recovery checks remain applicable engineering evidence; the proposed job still repeats recovery on GPU and stops on failure.

Use NCA-G1-Generation-Portable.ipynb with NCA-G1-Generation-Portable-Package.zip after approval for one replacement attempt. Keep APPROVED_G1_JOB=False until approved. Same one-T4, 256-update, 64-step, 600-controlled-second cap with two recovery replays; setup/export/idle additional. No automatic retry, extra seed, Drive operation or deployment. Return full result ZIP and receipt, including failures. Do not upload the prior package.

No new paid run was started. No reserved labels were opened. MG7 remains live. Repository synchronization is pending; this record and resume instructions are saved locally with a verified same-disk archive, not an off-device backup.
