# G9: access-ranking loss experiment

Only scientific change versus G8: add a TRAIN-only access-ranking loss, margin1,
weight1. All original membership/volume/band terms remain active. Teacher graph
labels never enter inference. Exact specification: G9 access-objective proposal.
Same45 TRAIN payloads,61-64-8 network, fresh paired seed1201,427updates64steps,
Adam0.001,clip1,teacher-stage schedule,firing0.5,threshold0.5,32cubed at0.8m.
Same cube admission, global cap and per-step quota. Nine families unchanged.
No G8 warm start, checkpoint selection, route input, bridge or postprocessing.

One proposed T4 job, maximum600 controlled seconds including startup/probes/
recovery; setup/export/download/idle extra. Runtime must match Python3.13.15,
Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700. Stop on mismatch or timeout;
no automatic extension/retry. G8 took336s; G9 has additional overhead, so this
is not a completion-time guarantee. Full recovery at updates2/3 remains required.
Keep all checkpoints,starts,trace and failures. Evaluate final427 only.

Log ranking loss and every step's supervision phase, fired comparison counts,
active ranking and cap status. Report no-route fallback separately; capped
states can have no route. Cache graphs deterministically; cache is derivable,
not learned state or random-number state. No new random draws in supervision.

Frozen local review:45 exposed regression requests and12 new reserved requests.
Evaluate G8 and G9 final427 on the SAME new12 requests, firing2101,CPUfloat32,
64/128steps. All nine families every case/horizon; cohort median absolute volume
fraction error<=.02,max<=.04; per-case mass change<=5%; visual review required.
Report regressions and individual failures. Fresh geometry and all heldout
labels are excluded from this training ZIP. No heldout inference before return.
No threshold/seed/epoch sweep. Synthetic one-seed result is not broad validation.

APPROVED_G9_JOB=False until this exact one-job allowance is approved. No Drive,
publication,push,automatic retry or live replacement; MG7 remains live.
