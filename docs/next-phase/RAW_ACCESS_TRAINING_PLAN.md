# Next proposed learning comparison: F4 raw access

Prepared 2026-09-23 from A3, D048. Proposal only; no training executed under F4.

## One change and a fixed control

Compare research-only raw_component_bottleneck_v3 with the constant16 F2
component_bottleneck_v2 objective. Replace only the access-family term. Retain
pre-clamp coverage, all nine families and three existing regularizers, both
mapped_30/mass_3 coefficient sets, original Model C initialization, scenes
ref-01-ground-pair/ref-06-minimal-smoke, training seed0, fire rate/update scale,
64 updates per model,16 rollout steps per update and F2 optimizer settings.
Do not continue training the failed F3 weights. Both arms have1024 recurrent
training steps/model. This is a small learning comparison, not model selection
on fresh holdouts or evidence across training seeds.

## Implementation gates before any full run

1. Add an explicit objective identity to metadata and checkpoint recovery.
   Preserve historical F2 source and resume guards. Baseline mode must replay
   F2's early actual updates, losses, gradients, RNG and checkpoint trees exactly,
   except declared experiment identity. Use saved F2 controls after verified
   hashes; do not repeat its full training just for a cleaner timing result.
2. Test the actual candidate loop: uninterrupted versus saved/restarted execution
   across several updates, full model/optimizer/scheduler/firing/global RNG
   equality, and both evaluation horizons. Also test a restart from a nontrivial
   trained boundary, preferably through final63/64. Register failures and use
   new IDs for linked recovery, never override a source mismatch.
3. Profile a fixed small pilot covering both scenes/recipes, training and all
   final evaluation horizons. Freeze cases, timeout allowance and timing-only
   admission before inspecting outcomes. Pilot work is a separate run; do not
   silently extend its optimizer exposure into the full study.
4. Run full regression before freezing scientific code. CPU execution only if
   the measured allowance is reasonable. Paid Colab needs a concrete job and
   approved GPU-hour allowance; any Drive operation separately needs exact
   folder-scoped approval. No such operation is required for preparation.

## Evaluation and decision

Save every update's scalar losses, weighted gradient norm before clipping,
checkpoint and RNG state. At fixed boundaries0,1,4,8,16,32,64 score16/50 steps
with firing seed2. At final64 evaluate horizons16,24,32,40,50,64 and seeds0,1,2.
Reuse exact overlapping evaluation records and count unique fields explicitly.
Retain both projected and raw access values, so a changed numeric loss cannot
masquerade as better geometry.

Primary result: projected component connectivity AND existing3–12% material
budget in the same evaluation. Report every scene/recipe/horizon/seed, counts
of F2 connections retained/lost/gained, mass changes, legality, blocked ground,
support, coverage and all remaining family/regularizer terms. Check beyond the
training horizon; a16-step success that breaks or overgrows later is not stable.
Preserve the original9-family semantics and disclose all proxy limitations.

Improved raw loss alone is failure to establish a geometric gain. Zero joint
success leaves the candidate unpromoted. A joint success justifies a separately
planned multi-seed/fresh-scene test, not immediate deployment. If failure again
comes from non-firing cells behind earlier clamps, choose ONE explicit update-
signal or conditioning change for the next comparison; do not bundle it with
reweighting or scale. A3 does not establish that changing architecture is needed.

## Preservation

Write the executed protocol and exact IDs to RESUME.md as each gate starts or
finishes. Keep historical evidence, negative results and source snapshots.
Produce a verified full local archive after the milestone. Reports remain
Git-ignored where required; archive includes the private reports. A local copy
on the same disk is not the pending off-device backup.
