# G6: one paced-growth training pilot

Change only the per-step admission allowance relative to G4. Keep the same fresh
seed1201,61->64->8 network,dataset bytes,teacher stages,losses,origin firing,
threshold and ordering. No G5destination channels and no trained warm start.
Same nine families,32cubed grid,0.8m voxels and overall building-volume meaning.

Global band unchanged:B=ceil(request*domain cells),C=min(B+8,floor(0.40*D)).
Set K=max(9,ceil((C-27)/63)) once per rollout. This formula uses the fixed64step
training horizon;it is unchanged for128step review. First seed-containing cube
uses global C and admits at most one cube. After that,effective per-step ceiling
is min(C,current mass+K). Apply the existing exact sequential overlap admission
to that ceiling. Unused per-step allowance does not carry over. Never trim cubes.
The floor9 permits the maximum new mass of a face-adjacent3cube. This schedule
does not guarantee target volume by64,connection,facade compliance or stability.

The teacher BCE,0.25local cube-volume term and1.0global band term are unchanged.
The differentiable union and global band still use global B,C before hard
admission;the new temporary cap is detached. This is a focused timing experiment,
not a claim that the surrogate equals the expected paced transition. Seed-phase
loss remains BCE only. Keep all training failures and post-cap trace evidence.

Seven count columns:initial_mass,offered_blocks,accepted_blocks,
allowance_rejected_blocks,redundant_blocks,deferred_seed_blocks,added_voxels.
Column3 now records rejections by the effective allowance,not solely global
budget rejection. Save quota and every step ceiling separately. Do not compare
this rejection count to G4's global-only count as if the definitions were equal.

Local prior evidence:existing G4weights with this single fixed schedule improved
TRAIN64 all-nine validity2/27to12/27 and access2/27to22/27,but facade26/27to14/27
and only18/27were stable within5% at128. This is a changed-inference diagnostic,
not trained G6performance or held-out evidence. No quota search was performed.

One Tesla T4 run:256updates64steps,batch1,float32,Adam0.001,clip1,maximum600
controlledseconds. Includes12original admission probes,8paced reference probes,
union backward check,two exact full-payload/state recovery replays,every completed
update checkpoint/start/state/trace,and final seed-only diagnostic. Setup,
upload/export/download/idle extra. ExpectedPython3.13.15,Torch2.11.0+cu130,
NumPy2.1.3,CUDA13.0,cuDNN92700. Stop on runtime/probe/recovery failure,nonfinite
values or reserved GPU memory>80%. No automatic retry or extension. Recovery
is same-runtime and completed-update only. Full evidenceZIP+receipt required.

Frozen review:final256checkpoint only,CPUfloat32,firing2101,same9 reused G1
development requests,single scene seed,64primary and128stability steps with
the same K. Require9/9all-nine atboth,median absolute volume-fraction error
<=0.02,max<=0.04,and each mass change<=5%. Report every case and family,IoU,
global-cap hits,per-step allowance usage and stalls. No threshold,checkpoint,
horizon or quota search. Existing G4run is the comparison;no extra GPU control.
Reserved labels stay unopened. Development reuse must be disclosed. No live
promotion follows automatically. MG7 remains live.

Keep APPROVED_G6_JOB=False until this exact one-job allowance is approved.
No Drive operation,push,publication,extra seed,paid retry or model replacement.
