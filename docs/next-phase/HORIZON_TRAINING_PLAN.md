# F3 proposal: train at both short and long growth horizons

Prepared 2026-09-23 from H1; not executed. Read GROWTH_AUDIT_FINDINGS.md and D044.

## Question and isolated intervention

Does alternating short and long growth during optimization improve joint entrance
connectivity and material-budget compliance compared with F2's 16-step training?
H1 shows late-state material gradients are available, but does not prove learning
will use them successfully.

Propose four models: the same ground-pair and minimal-smoke scenes, existing
mapped_30 and mass_3 recipes, original Model C initialization, training seed 0,
weak 0.15 scaffold, and 64 updates per model. Use component_bottleneck_v2 access
and all other F2 terms, coefficients, optimizer/scheduler settings and precision.
Odd numbered updates grow 16 steps; even numbered updates grow 50. Restart from
the same prescribed seed scaffold each update, as in F2. No state pool, curriculum,
new constraint family, architecture change, budget relaxation or fitted warm start.

This holds optimizer-update count fixed, not compute: 32*16 + 32*50 = 2112 growth
steps per model, versus F2's 1024, before evaluation. Different horizons also
consume different amounts of firing RNG. Record both facts; this is a training
schedule comparison, not equal-compute evidence for a universally better method.
An equal-compute comparison would be a later explicitly planned experiment.

## Gates before the proposed full comparison

1. Implement a separately versioned horizon schedule and include its identity,
   next update, RNG, optimizer and scheduler state in strict recovery metadata.
   Historical F2 source snapshots remain immutable. A constant-16 compatibility
   path must exactly reproduce registered F2 traces/checkpoints for all four
   members before those models serve as controls. Never bypass source guards.
2. Verify the actual mixed-horizon loop in fresh processes: a three-update
   16/50/16 sequence run uninterrupted versus update 1 plus resumed updates 2/3.
   Compare full checkpoints, optimizer/scheduler/RNG states, losses, evaluated
   fields and next-update horizon exactly. Exercise both transitions and preserve
   failed attempts. CPU recovery does not certify Colab/GPU recovery.
3. Profile a fixed small pilot that includes 50-step backpropagation, evidence
   writes, intermediate evaluations and the complete final evaluation workload.
   Freeze configuration, numerical gates, worker caps and overall allowance
   before the full run. Admit using time/resource measurements only, not quality.
   Apply explicit elapsed checks even when a worker exits successfully.
4. Do not silently shorten horizons, drop a recipe/scene/seed, loosen thresholds
   or reduce evaluations if the allowance is exceeded. Preserve the attempt and
   record a new protocol revision before an alternative run. Paid Colab requires
   an explicit spending allowance and its own recovery verification.

## Measurements fixed before training

Use F2 boundaries 0, 1, 3, 8, 16, 32, 64, evaluated at both 16 and 50 growth steps
with firing seed 2. Save all checkpoint trees and training traces. Final models
also receive H1's six horizons (16, 24, 32, 40, 50, 64) and three firing seeds
(0, 1, 2), allowing comparison against the existing 72 final F2 fields. Count
duplicate boundary/final evaluations once when reporting unique cases, while
retaining provenance for any actual repeat. Establish original-field parity.

Report both access definitions, all nine loss families, three regularizers,
continuous mass, strict binary connectivity, geometric violations, saturation,
gradient/clipping statistics, training and evaluation times. Primary outcome:
connectivity AND the existing 3%-12% mass budget on the same field. Present every
scene/recipe separately, short/long results and the complete final horizon grid;
do not select a successful stopping horizon or firing seed after seeing results.
Retain comparisons against F1/original where already available in H1.

## How results determine the next decision

A gain on these two training scenes would justify replication and later fresh
holdouts; it would not justify deployment or a generalization claim. Report
connectivity regressions alongside mass improvements. If joint feasibility is
still absent, inspect actual optimizer changes and saved term/gradient traces
before isolating one next intervention (for example loss margin or state pool).
Do not automatically increase training duration or combine changes to hide a
negative result. No architecture or project-concept change is established by H1.

All attempts and decisions remain local and archived, with the private next-phase
report Git-ignored. Every Drive operation needs its own exact approval within the
project folder; this plan authorizes no cloud access, paid run or remote push.
