from pathlib import Path
import json
r=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');out=Path('C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum-Design-2026-09-29')
summary=json.loads((out/'result.json').read_bytes())
(r/'experiments/reports/CGR3-curriculum-feasibility.json').write_text(json.dumps(dict(summary=summary,artifacts=str(out),status='design_audited_not_training_ready'),indent=2)+'\n',encoding='utf-8')
doc='''# CGR3 intermediate-start curriculum specification

D096,2026-09-29. Design and sampler audit complete; training integration and
package are not yet implemented. No GPU job approval or execution implied.

## Hypothesis and controlled comparison

CGR2 rejects reachable missing cells repeatedly. Test whether exposing the CGR1
model to additional partial-completion states improves discrimination on original
damaged inputs. This is one change in training start-state distribution, not a
claim that the diagnosis proves curriculum learning will work.

Use CGR1 v2 architecture, losses (positive0.5,negative1,intact-negative1.5,
local-volume0.25),32steps,Adam.001,seed1201,fresh weights,256updates and firing0.5.
Do not retain CGR2's bulk term or stronger intact penalty. CGR1 is the control;
relative to CGR2 there are multiple differences, so do not describe that contrast
as a single-factor ablation. Same81TRAIN source rows, sorted(case,damage), sampler
seed1203. Same nine families and overall-building-volume meaning; no rooms.

## Exact starting-state rule

Each row has a zero-based visit counter. On even visits (0,2,...) use the original
input. Intact rows always remain unchanged. On odd visits of damaged rows, start
from the original occupancy and perform k=1+(visit//2)%3 synchronous oracle growth
stages. A candidate cell must be empty, face-adjacent to current occupancy, and
inside the binary target. Independently select each candidate with probability0.5.
If a stage would complete the target, stop before accepting that stage. If nothing
was added, record original_fallback. Keep hidden channels zero at initialization.
Then run32 ordinary model-driven steps, with no teacher correction in that rollout.
All original-input rollouts are retained on even visits; intermediate rollouts are
also on-policy after their initialization, but their initial distribution differs.

Use a separate CPU generator seeded by SHA256 of
intermediate_repair_starts_v1|1201|source_array_sha256|visit,
first8bytes big-endian modulo(2^63-1). Draw a full-grid float random tensor per
stage. Do not consume model initialization, row sampler or firing RNG streams.
The concrete helper is nca/repair_curriculum.py; CPU sampling is intentional.

Targets construct augmented TRAIN occupancies, just as they define supervised
labels. This is teacher-derived training data, not a target-free data pipeline.
The model receives only occupancy plus unchanged seven-channel context. Never
pass target masks, damage masks, visit index, stage count, teacher IDs or oracle
frontiers as model inputs. Ordinary inference and every evaluation must bypass
the sampler and use original inputs. Sampler rejects non-TRAIN split requests.
This guard is not a security boundary; package construction must still exclude
all heldout rows. Do not put evaluation transforms in the shared tensors loader.

## Audited schedule and limitations

The exact256-step sampler schedule contains194original and62intermediate starts;
86updates are intact and unchanged,108are original damaged,62intermediate damaged.
No fallback. Every source appears3or4times. Augmentation adds2268cells in total,
maximum84in a start. Audit saved all256start arrays and records with source/start
hashes; deterministic replay matched every state. Inputs remain legal, binary,
nonempty, input-preserving subsets of targets; intermediate damaged starts never
become full targets. The non-TRAIN rejection check passed. No model optimized.

This short schedule reaches only1or2oracle stages; the third stage is defined
for later visits but is not exercised or evidence for this proposed trial.
The curriculum makes some tasks easier and reduces exposure to original damage.
Zero hidden initialization is not the hidden state of a real partial rollout.
There is no promise of better hard-damage repair or exact GPU recovery. Do not
extend the trial or tune state fractions using development results afterward.

## Required implementation before a compute request

Create separate versioned session identity and start_sampler metadata, preserving
all prior sessions. Update visits only for a consumed training row, record chosen
mode,stage,seed and start occupancy hash in each update, and save the start array.
Checkpoint visits and verify they agree with completed count and row history.
Recovery must reproduce next start, firing sequence, loss,optimizer and weights.
Keep evaluation tensor loading untouched; test original-input evaluation and
same-shape CGR1/CGR2 checkpoint rejection. One consolidated CPU correctness and
recovery check plus an eight-update package rehearsal is sufficient if they pass.
Package TRAIN81 only and retain full source/manifest,notebook and result receipts.

## Frozen comparison and proposed compute

Final256 only,CPUfloat32,32steps,firing2101,same27development examples,original
input starts,accepted occupancy, no cleanup or threshold tuning. Compare CGR1,
CGR2,NR5 and closing3; preserve per-case arrays. Original gates remain: all9intact
IoU>=.99 and valid; damaged validity>=17/18,medianIoU>=.9705768039313023,
excess<=325,recovered>=1945,median absolute volume error<=19; no input removal.
Report regressions versus CGR1 even if original gates pass. No TEST or automatic
live admission. MG7 remains live.

Propose ONE fresh Colab T4 job,seed1201,256updates,32steps,600controlled seconds
maximum; setup/export/idle extra. Request explicit approval only once the concrete
package and local verification are ready. No retry,extra seed,Drive,push or launch.

Artifacts: C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum-Design-2026-09-29.
Reproduce with scripts/audit_repair_curriculum.py and a fresh output directory.
Same-disk archive only; no off-device backup claimed.
'''
(r/'docs/next-phase/CGR3_CURRICULUM_SPEC.md').write_text(doc,encoding='utf-8');(out/'SPECIFICATION.md').write_text(doc,encoding='utf-8')
entry='''## D096 — CGR3 curriculum specified and audited — 2026-09-29

ONE proposed start-distribution change versus CGR1 (not CGR2). Same CGR1 losses,
architecture,seed,256updates and32steps. Alternate original/teacher-derived partial
starts by per-row visit, intact unchanged; model receives occupancy+context only.
TRAIN-only deterministic sampler implemented; exact256-start audit:194original,
62intermediate,86intact total,zero fallback. Verified reproducibility,binary/legal
subset,no input removal,no complete augmented damaged target,non-TRAIN rejection.
All256start arrays and hashes saved in Codex outputs/CGR3-Curriculum-Design-2026-09-29.
See CGR3_CURRICULUM_SPEC.md and experiments/reports/CGR3-curriculum-feasibility.json.
This is design feasibility, not model training/quality or a GPU-ready package.
Next implement versioned training session with visit-counter recovery,unchanged
original-input evaluation; one consolidated CPU check and8update package rehearsal.
Then request ONE concrete256update600s T4 job approval. No automatic launch/retry,
Drive,TEST,push or admission. MG7 remains live. Preserve prior results.

'''
for name in ['RESUME.md','PLAN.md']:
 p=r/'docs/next-phase'/name;a,b=p.read_text(encoding='utf-8').split('\n',1);p.write_text(a+'\n\n'+entry+b,encoding='utf-8')
for name in ['DECISIONS.md','CHANGELOG.md']:
 with (r/'docs/next-phase'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+entry.rstrip()+'\n')
