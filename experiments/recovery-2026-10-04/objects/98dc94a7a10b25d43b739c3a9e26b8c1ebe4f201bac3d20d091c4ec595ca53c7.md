# G1 — From a seed to building volume

2026-10-03. Preparation milestone; no trained generation result or paid run.

## Decision

Close the current reversible-repair branch. Its paired trial achieved the same 24/27 development passes as CGR1, with lower damaged-case recovery (1818 versus 1959 cells) and four failed acceptance gates versus two. Keep all evidence, CGR1 as a research reference, and MG7 as the live procedural generator. Neither learned repair model meets its acceptance criteria. This is evidence against adopting this candidate, not against every reversible NCA.

The next learned task is to generate overall building mass from a small seed and scene constraints. Occupancy means building volume; rooms, construction and internal cavities are not required. Preserve meaningful exterior gaps. Keep the existing nine constraint families.

## What the local audit established

`seed-audit.json` records all 27 unique intact TRAIN targets, their verified source-array hashes and a target-independent candidate seed. These come from only three scene contexts, not 27 independent sites. No validation or TEST arrays were opened by this audit.

The proposed seed is one permitted domain cell at the lowest-X interface: select its lowest-X plane, take the cell nearest that plane's centroid, and break ties in z/y/x order. Existing and protected cells are excluded. The rule reads only context channels. It is orientation-specific and is not claimed to be rotation invariant or suitable for every future scene.

- The seed lies inside 26/27 teacher volumes. Reusing all targets with an immutable seed would therefore create one impossible reconstruction target.
- All nine identical scene/request input groups contain multiple distinct teacher volumes. Training deterministic reconstruction against these without a variation input can give conflicting supervision; this audit does not prove that it caused earlier failures.
- Every teacher cell is within 27 six-neighbor steps through the legal domain. This optimistic geometric bound does not establish that stochastic learned growth, or growth restricted to the teacher, can complete in 32 steps.

The study file also contains historical TEST baseline aggregates, and its opening portion was read during discovery. Do not describe that historical split as wholly unseen. Its arrays were not used here. New generation generalization claims require a newly frozen split with a clear access history.

## Input and output contract

Use the existing seven context channels: domain, permitted, existing, protected, support boundary, interfaces and requested volume fraction. Initialize occupied mass from the scene-defined seed only. No damaged teacher, teacher-picked seed, teacher route, complete procedural mass, or target-dependent distance field may enter model inference. Keep labels outside the inference API and verify output equality when only labels change.

The first baseline is deterministic conditional generation. Use one procedural teacher per scene/request, selected by a fixed generator seed before scoring. Preserve alternative teachers for later diversity work; do not count three damage variants as three generation examples. Fix firing seeds before comparisons. A later explicit variation input and diversity experiment must be versioned separately.

An anchored teacher builder must include the independently selected seed by construction and retain failed construction requests. Do not silently union a seed into old labels or discard unfavorable examples. Re-evaluate the complete teacher field against all nine families. If the existing procedural generator cannot accept this anchor without substantial redesign, record that finding and choose an explicit seed contract before training.

Retain CGR1's local network dimensions and irreversible connected growth as the initial architecture reference. Start fresh weights and version the generation wrapper, seed semantics, training schedule and objective. Reversible deletion is not adopted. CGR1's repair sampler and frontier-only loss must not be copied blindly: early seed growth needs supervision on successive reachable fronts. Build teacher-restricted growth trajectories from the context-defined seed, then specify their sampling and generation rollouts together. These implementation details remain to be frozen before any paid run.

## Evaluation and data

Begin at 32 cubed and 0.8 m voxels to separate learning progress from scaling changes. The 2.4 m thickness scale therefore remains three cells. Increasing grid size at fixed spacing enlarges the physical site; increasing resolution at fixed site size requires updating physical-to-voxel conversions. Treat these as separate experiments.

Primary quality measure: fraction of all requested final volumes passing all nine unchanged `massing_targets_v1` checks. Report every family separately, empty outputs and all construction failures. Access includes connected raw and bulk mass touching interfaces; thickness requires at least 90% cube-supported mass. Geometric support is not structural safety.

Report requested-volume error, per-scene outcomes, runtime, peak memory and rollout stability alongside validity. Teacher IoU is a reconstruction diagnostic, not the definition of a good design: multiple different volumes can satisfy the same request. Diversity becomes a separate measured objective once variation is explicitly conditioned. Do not add it as a tenth constraint family.

Compare against the live MG7 procedural baseline on identical requests, with the same recorded hardware where timing is compared. No best-of-N selection or hidden repair after generation. If any postprocessing is later proposed, score raw and processed outputs separately.

Use the three historical TRAIN contexts for engineering preparation and label the repeatedly used offset-interface scene as development. Build a broader context-disjoint dataset for generation: split geometry/context hashes before creating teachers, keep near-duplicate scene families together, and retain an untouched final evaluation manifest. Define blocked/infeasible requests separately; necessary feasibility checks do not prove feasibility. No numbers from the old repair split become generation acceptance thresholds automatically.

Before training, freeze exact scene counts, requests, teacher seed, firing seeds, rollout horizon, update count, loss, checkpoint-selection rule, time cap, numerical acceptance gates and failure policy. Set these using task requirements and procedural baseline evidence, before observing learned generation outputs. This document is the milestone specification, not a completed frozen training protocol.

## Compact execution sequence

1. Build the seed-only input adapter, anchored deterministic teacher dataset and context-disjoint manifest in one local preparation milestone. Check label independence, seed compatibility, teacher validity and physical thickness together. Retain all failures and exact hashes.
2. Freeze one bounded generation comparison and verify checkpoint recovery in that implementation. Reuse the existing archive and recovery infrastructure. Present one concrete Colab package and budget for approval; do not start another chain of repair trials.
3. Review the returned full ZIP against the frozen generation criteria. If it misses them, diagnose the dominant failure before proposing another run. If it meets them, test larger and more varied sites with the same nine families.
4. Integrate an accepted generation model into the polished live interface, showing provenance, constraints and real growth. Keep procedural and learned modes clearly labeled until a learned candidate qualifies.

The immediate next action is step 1. No Colab upload is needed now. GPU quality, scaling performance and completion dates remain unproven.

## Preservation and resuming

Prior evidence: `C:/Users/artin/Documents/Codex/outputs/RGR1-Paired-Review-2026-10-03/`. Repository records currently stop at D098; later CGR3/RGR1 evidence lives under Codex outputs. This milestone is also saved there because repository write access was not granted. Synchronization into tracked decisions, changelog, experiment records and RESUME remains pending. No repository commit, model replacement, Drive operation, push or publication occurred.

The saved audit script and source fingerprints make the audit repeatable. A verified local ZIP is preservation on the same disk, not an off-device backup. Keep full experiment ZIP exports as requested. Resume from this document and `RESUME.json`, together with the paired review; do not restart earlier repair work from the stale repository RESUME alone.
