# G1 generation data preparation

2026-10-03. Local procedural dataset and model-input implementation; no learned generation result.

## Completed change

Implemented a seven-channel scene adapter that selects one occupied seed without reading a target. The inference wrapper accepts context, firing seed and rollout length only. It reuses the CGR1 network and connected birth rule, with fresh weights in the smoke check. The training helper builds connected stages using distance through the teacher volume and refuses non-TRAIN requests. These teacher-derived stages are supervision, not inference inputs.

Created a separately versioned copy of MG7's teacher generator. Its initial route starts at a supported interface cube containing the independent seed. All routing, contact accounting and coverage growth otherwise retain MG7 logic. The live generator and existing targets were not edited. No label patching, success-based seed selection or retries were used.

## Dataset result

| Split | Contexts | Requests per context | Teacher passes | MG7 baseline passes |
|---|---:|---:|---:|---:|
| TRAIN | 9 | 3 | 27/27 | 27/27 |
| Development | 3 | 3 | 9/9 | 9/9 |
| Reserved | 4 | 3 planned | Not generated | Not evaluated |

All generated teachers contain the seed and are reachable through their own occupied cells. Maximum teacher path distance is 28 cells across the generated dataset. This is an ideal synchronous growth depth, not a demonstrated learned rollout length; stochastic firing can require more steps.

Requests are 16%, 24% and 32%; generator random seed is fixed at zero before evaluation. There is one teacher per conditioning input. No examples were discarded. The split manifest was written before teacher generation. Every output includes raw arrays, generator report, unchanged nine-family evaluation, seed, teacher distances, procedural baseline and hashes.

TRAIN families are aligned, wide gap and partial obstruction, each with three interface-Y variants. Development uses the historical offset-interface family with the same three variants. Reserved geometry uses raised interface pairs and unequal building heights, two variants each. Entire named families stay in one split; context hashes are unique. These are small, related synthetic families. They do not establish broad architectural generalization, and the reserved geometry itself was constructed and inspected through its masks. Its target volumes and outcomes remain ungenerated.

## Verification and limits

See `verification.json` for executed checks. The suite recomputes all 36 teacher evaluations and distances, verifies all array hashes, checks every teacher growth stage for six-face adjacency, checks family separation, replays an anchored teacher, rejects missing seeds, runs deterministic seed-only inference and checks finite nonzero training gradients. It does not measure trained quality or test checkpoint recovery for a generation training session.

The model adapter is ready for a training-session implementation. The final training-stage sampler, horizon, loss schedule, update/time cap, quality gates and recovery integration are still to be frozen together before requesting one paid run. In particular, inference must always start from the one-cell seed even if training includes teacher-derived stages. No existing repair checkpoint is promoted or silently resumed.

Keep the nine existing constraint definitions and physical scales. The data remains 32 cubed at 0.8 m per voxel, with 2.4 m cube-supported thickness. Validity is the primary generation outcome; exact teacher overlap is only diagnostic. MG7's 36/36 here is the procedural reference, not NCA performance.

## Next action and preservation

Implement one resumable G1 training session with an explicit mixture of seed starts and teacher stages, freeze its complete protocol, and perform one combined local training/recovery rehearsal. Then present one concrete Colab package and budget. No Colab action is required from the user yet.

All work is in this local folder. Repository synchronization remains pending because write access has not been granted; no commit, live deployment, Drive operation, paid compute or push occurred. `source.zip` preserves local implementation dependencies, and the verified milestone ZIP preserves this folder. Both are same-disk copies, not off-device backups. Resume through `RESUME.json`; the repository RESUME ends earlier at D098.

Prior milestone: `C:/Users/artin/Documents/Codex/outputs/Generation-Milestone-2026-10-03/GENERATION-PLAN.md`.
