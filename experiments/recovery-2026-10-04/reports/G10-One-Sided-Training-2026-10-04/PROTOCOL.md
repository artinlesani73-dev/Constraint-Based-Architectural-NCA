# G10: one-sided access ranking

The only scientific change from G9 is stopping the auxiliary ranking gradient
through the non-advancing teacher-positive reference scores. Numerical ranking
value and advancing logit gradient are unchanged at a fixed state. This is an
explicit semi-gradient, not the full derivative of symmetric ranking. Margin1,
weight1 and all original membership/volume/band losses remain. The other logits
can still change through shared parameters: improvement is not guaranteed.

Same45 TRAIN examples,61-64-8 network,fresh seed1201,427updates64steps,Adam0.001,
clip1,teacher-stage schedule,0.5firing/threshold,32cubed at0.8m,original quota
and global cap. No new inputs,teacher routes at inference or postprocessing.
The metadata seed-loss label is corrected to include the ranking term.
No warm start or checkpoint selection.

One proposed fresh TeslaT4 job,max600 controlled seconds; setup/export/download/
idle extra. G9 took469s; this is not a guarantee G10 finishes in the cap.
Strict runtime:Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700.
Keep full recovery at2/3,finite/memory guards,per-update evidence and all failures.
No automatic retry,extension or runtime-guard bypass.

Frozen review: final427,CPUfloat32,firing2101,64/128steps. Evaluate57 prior
regression requests and12 new reserved requests. Run G9 final427 on the SAME new
12 cases as the primary causal comparison; also run fixed G8 as the stability
reference. No fresh labels in package. Report models/cohorts separately.
All nine families every case/horizon; median absolute volume-fraction error<=.02,
max<=.04; per-case mass growth<=5%; raw geometry visual review required.
No rerolls,coefficient sweep or changing gates after results. One seeded
synthetic comparison is not broad architectural validation.

Keep APPROVED_G10_JOB=False until this exact one-job allowance is approved.
No Drive mount,push,publication or live promotion. MG7 remains live.
