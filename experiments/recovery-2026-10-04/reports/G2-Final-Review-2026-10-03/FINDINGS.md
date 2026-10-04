# G2: growth improved, volume control failed

2026-10-03. Run `20261003T193511Z_56901714f0be`. Final decision: retain as research evidence, do not promote.

## Verified execution and comparison

Training completed all 256 updates in 114.249 controlled seconds (107.100 worker seconds), with peak reserved GPU memory 1,436,549,120 bytes. Both embedded GPU recovery replays report exact full-payload and state equality. All 1,034 archive payload hashes, unique membership and receipt verify. Final checkpoint trace, sampler, optimizer steps and all 256 start selections verify. There are 128 seed starts and 128 teacher-stage starts.

G1 and G2 initial parameter tensors, training array identities, ordered row/stage selections and final firing RNG are exactly equal. Python, Torch, NumPy, CUDA, cuDNN and GPU model versions match. The sole scientific change remains the positive frontier loss coefficient, 0.5 to 1.0. These runs use one model seed and reused development scenes; they do not establish broad robustness. Timing is descriptive across separate jobs, not a controlled speed benchmark.

## Frozen development review

Both use final checkpoint256 only, CPUfloat32, firing2101, the same nine development requests, single-cell starts and no postprocessing. Reserved targets remain ungenerated and unevaluated. All primary64-step and additional128-step outputs are retained; no checkpoint or horizon selection occurred. Exact packaged model code and saved evaluator source were used.

| Measure, primary64 steps | G1 | G2 |
|---|---:|---:|
| All nine families pass | 0/9 | 0/9 |
| Access | 0/9 | 9/9 |
| Coverage | 0/9 | 9/9 |
| Thickness | 0/9 | 1/9 |
| Sparsity / allowed volume range | 0/9, too small | 0/9, too large |
| Other five families | 9/9 each | 9/9 each |
| Occupied domain fraction, range | 1.83–2.25% | 46.73–51.87% |
| Median absolute requested-fraction error | 22.00 percentage points | 25.28 percentage points |
| Maximum absolute requested-fraction error | 30.01 percentage points | 31.04 percentage points |
| Median teacher IoU, diagnostic | 0.0835 | 0.4871 |
| Relative mass change64 to128 steps | 90.57–107.62% | 0.18–0.75% |

Requests were 16%,24%,32%. The unchanged sparsity constraint permits8–40%, so G2 exceeds even the general upper limit in every case, as well as missing the requested amounts. At128 steps G2 still passes0/9 overall, with7 families passing every case and thickness passing1/9. Cube-supported fraction is81.97–90.22% at64steps and81.58–90.13% at128steps, against the90% requirement. More occupied cells do not automatically give adequate thickness everywhere.

G2 passes the mass-stability gate, but fails the other four preregistered gates: all-nine validity at64 and128, median request error<=2 percentage points and maximum request error<=4 points. Stability means the oversized shape has nearly stopped changing; it is not acceptance. MG7 procedural reference stays9/9 on these requests.

## Interpretation and next design step

The controlled weight change demonstrates a substantial response in growth, access and coverage, but does not solve valid generation or volume matching. The result does not support adopting G2 or reporting it as an overall success. It also does not prove that no intermediate coefficient could work; no coefficient search was performed.

Recommend moving from scalar growth-weight adjustments to an explicit volume-budget design. The model currently sees requested fraction in static context, while its auxiliary volume loss is a local cube comparison to a teacher. Neither provides an explicit current-versus-requested whole-volume signal to each local growth decision. A useful next design candidate is a scene-request-derived remaining-budget feature and a corresponding global volume-error term, evaluated together with the existing thickness metric.

Before another paid run, specify and locally verify how remaining budget is calculated, normalized, broadcast and used during training and inference. It must depend only on the scene request and current generated mass, not teacher geometry. Broadcasting a global count makes the system hybrid rather than strictly local NCA; document that architectural tradeoff explicitly. Keep the same nine constraint families: this gives sparsity/volume control a better implementation rather than adding a tenth family.

An irreversible birth rule cannot undo an overshoot. This limitation must be addressed in that design before claiming exact volume control. A hard cutoff can also halt growth before access, coverage or thickness is satisfied. Do not silently crop or truncate outputs and report only the corrected result. The next deliverable is a concrete budget-aware design and its local invariants, not another automatically launched GPU trial. Model promotion still requires the unchanged evaluation protocol and later reserved generalization evidence.

## Preservation

Original ZIP/receipt, final checkpoint, update records, verified start identities,18 scored outputs/states, exact source, review script and G1/G2 per-case comparison are retained. The original ZIP retains all checkpoints and training states. Verification and pairing checks are in imports.json and pairing-verification.json; metrics are in result.json and comparison.json. Full recovery on GPU is evidenced by the supplied worker reports, not independently re-executed here.

Repository synchronization remains pending because write access was not granted. Everything is saved locally and included in a hash-verified same-disk milestone archive; it is not an off-device backup. MG7 is unchanged. No new paid run, Drive operation, push, commit, live deployment or reserved evaluation occurred.
