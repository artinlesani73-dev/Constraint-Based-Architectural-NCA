# G3 bounded budget-feedback pilot

Fresh seed1201 model:61->64->8 pointwise network with existing60 inputs copied
from an identically seeded fresh G2-size initialization and the new input weights
zero. No trained checkpoint warm start. Extra channel broadcasts(B-M)/D each step.
B=ceil(request*domain count);C=min(B+8,floor(0.40*D)). Only16/24/32% requests,
32cubed,0.8m spacing,3-cell physical bulk. Reject starts/teachers outside the band.
All27 original TRAIN arrays and row/start schedules unchanged. Same50/50 seed
and teacher stages,256updates64steps,batch1,float32,Adam lr0.001,clip1.
Keep G2 frontier weights1:1 and local cube-volume0.25. Add coefficient1 global
band error:distance of current mass+sum fired-frontier sigmoid proposals from
[B,C],divided byD. Compute loss before hard admission. Hard births/ranking detached.

Intrinsic cap selects at most C-M above0.5 fired legal frontier proposals,descending
probability,stable ascending ZYX index ties. GPU tensor sort; no NumPy conversion
in the admission loop. Global count,broadcast,ranking make this a hybrid NCA.
This is an integrated architectural candidate,not a single-factor ablation.
Budget obedience and stability at capacity are enforced,not learned achievements.
Wrong early additions remain irreversible;geometry may fail before budget is spent.

ONE Tesla T4 run capped600controlledseconds including8 device/reference admission
probes,256retainedupdates,two recovery replays,checkpoint/state/trace writes and
final seed-only diagnostic. Setup/upload/export/download/idle extra. Expected:
Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Stop on mismatch,device-probe or recovery failure,nonfinite values or reserved
GPU memory>80%. No automatic retry. Every completed update saved. FullZIP+receipt.
Recovery replays teacher-stage update2 and seed-start update3;full payload+state
must match. Completed-update recovery only,not cross-runtime or mid-rollout.

Frozen review:final256only,CPUfloat32,firing2101,the same9G1/G2 development requests,
single-seed64steps. Fixed128steps for stability,never best-horizon selection.
Same gates:9/9 all-nine at64;median absolute requested fraction error<=0.02,max<=0.04;
9/9all-nine at128 and mass changes<=5% of64-step count. These are pilot engineering
gates,not generalization or deployment approval. Report all failures,per-family
metrics,teacherIoU diagnostic,volume errors,raw proposed/admitted/rejected counts,
ceiling-hit steps and pre-admission candidates. A pre-admission candidate is not
a full no-guard rollout. No post-hoc clipping,reroll,threshold or checkpoint search.
G2 completed run20261003T193511Z_56901714f0be is existing baseline,no extra control
GPU job. Reused development clearly labeled;reserved targets stay unopened.

Keep APPROVED_G3_JOB=False until this exact one-job budget is approved. After
approval run once,download FULL evidenceZIP+receipt even after failure. No Drive,
automatic retry,extra seed,public deployment or model promotion. MG7 stays live.
