# G1 bounded generation pilot

Fresh CGR1 network; G1 seed generation semantics. TRAIN27 only from the verified
G1 preparation. One seed1201,256 retained updates plus2 exact-recovery replay
updates,64 steps per update,batch1,float32,Adam lr0.001,gradient clip1.
Alternate seed-only and teacher-stage starts,128 each; stage depth determined by
SHA256(completed-update:row-index) modulo maximum teacher distance minus one.
Teacher starts have zero hidden state. Keep CGR1 loss: positive0.5,negative1,
local3-cube volume0.25; no intact examples. Detached hard births; no topology
gradient. This is a generation baseline,not a controlled repair ablation.

ONE Tesla T4 job capped at600 controlled wall seconds including recovery replay
and output writes. Setup,upload,export,idle are extra. Exact expected stack:
Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Stop on mismatch,failed recovery,nonfinite values or reserved GPU memory>80%.
No automatic retry. Checkpoint every completed update; full evidence ZIP+receipt.
Recovery checks replay both a teacher-stage update and a seed-start update,
comparing the full payload and generated state. Completed-update recovery only;
not mid-rollout recovery or portability across different runtimes.

Freeze review now: final256 checkpoint only,CPUfloat32,firing2101,all9 G1
development requests,seed-only64-step outputs. Also report fixed128-step outputs
as a stability diagnostic,never choose the more favorable horizon or checkpoint.
No teacher stages,cleanup,reroll or teacher input during evaluation. Primary pilot
gate:9/9 pass all unchanged massing_targets_v1 families at64steps; median absolute
requested fraction error<=0.02,and maximum<=0.04. Stability gate:9/9 remain valid
at128steps and each changes occupied count by<=5% of its64-step count. Report
all failures,per-family results,teacher IoU diagnostic,volume error and timings.
These are newly preregistered engineering targets,not established quality claims
or new constraint families. MG7 reference is9/9 valid on these same requests.
No reserved labels or evaluation in this pilot; passing development does not
authorize deployment or establish generalization. MG7 remains live.

Open notebook and upload matching ZIP. Keep APPROVED_G1_JOB=False until this exact
budget is approved. After approval run once,download full ZIP and receipt even
if it fails. No Drive access,external sync,public deployment or model promotion.
