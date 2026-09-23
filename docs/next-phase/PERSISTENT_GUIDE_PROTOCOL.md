# F5 frozen local protocol: persistent scaffold conditioning

2026-09-23, D052. User requested finishing the local investigation. This protocol
tests one architecture change against completed F4, without changing the nine
families, three regularizers, raw-access objective, original initialization,
scene distribution, recipes, training seed, optimizer, horizon or material budget.

## Architecture and primary outcome

Version persistent_scaffold_v1 retains all eight state channels (four frozen,
four evolving) and original backbone tensors. Cache identity and three Sobel
features of exactly item['scaffold'] = corridor_legal_v1, with scene, seed,
scaffold, configuration, dtype/device and cache-integrity guards. A bias-free
zero-initialized 4->96 projection adds384 weights to the first preactivation.
Projection is recomputed once per rollout with autograd, shared across its steps.
Static features contain no autograd graph. Cache never lives in global job state.
Construction consumes no extra RNG beyond the original backbone. Disabled
control uses the original model without extra parameters. All trainable parameters
are clipped together under the unchanged threshold; the new gradient contributes
to that norm. Log guide gradient before clipping and weight norm after updating.

Primary outcome remains strict>0.5 component connectivity AND continuous material
within3-12% of envelope (report tolerance1e-6). Retain all family metrics, F4/F2
references and every output. No loss-only/model promotion. This is development
evidence: two scenes, two recipes, one training seed,64 updates/member at16 growth
steps (1,024 recurrent steps/member). Fresh scenes/seeds required after promise.

## Ordered local gates

1. Full regression: zero-branch forward/backbone-gradient/RNG parity; actual
guide derivatives; batch2; repeated fresh backward graphs; immutable/correctly
bound context; malformed input; migration and metadata guards. Commit source.
2. F5B: original-architecture path,3 updates/member;12 updates/24 boundary
evaluations must match saved F4 traces, arrays, optimizer/checkpoint/RNG states
except declared source/architecture metadata. Worker180s,total900s.
3. F5R: conditioned mass_3 ground-pair whole0->3, prefix0->1, resume/repeat1->3;
full state/trace/fields and evaluation exact. Worker180s,total900s.
4. F5P: two conditioned updates/member and full final grid. Worker240s,total1200s.
Require nonzero guide gradients for the eight pilot updates. Timing-only full
admission1.5*(max(5,max(setup)+3)+64*max(update)+7*max(eval_pair)+max(extra_grid))
per member. Each estimate<=900s and total<=3600s. The3600s full cap is frozen
before F5 outcomes to allow added architecture work; not a relaxation of F4.
5. F5 study only after verified matching-source gates. Worker900s,total3600s.
64 updates for each of four members. Boundaries0,1,3,8,16,32,64 at16/50 steps,
firing seed2. Final grid16,24,32,40,50,64 x seeds0,1,2:72 records, eight reused
boundary fields.56 boundary+72 grid=120 unique evaluations. Save every pre-update
field and post-update optimizer/RNG checkpoint and cursor. CPU2 threads.
6. F5L: all four trained update62 checkpoints replay63/64 and final16/50
evaluations; exact state/trace/fields, worker120s,total360s. No added exposure.

Reject metadata/source drift, all runtime overruns (including elapsed suspension),
incomplete evidence or guard bypasses. Failed runs remain registered. A failed
timing gate closes this attempt without enlarging its cap. GPU/AMP and interrupted
file-write recovery are not certified by CPU update-boundary recovery.

## Closure

Verify every saved field, source hash, cursor, control and eight final checkpoint
rollouts. Compare F4 connections retained/gained/lost and material changes; inspect
all other family scores and long-growth behavior. No joint success means no
promotion. If this bounded conditioning comparison fails, close local incremental
loss/conditioning tests and write a representation/NCA-role decision before any
further training. If promising, freeze a new fresh-scene/multi-seed validation plan.
The whole product upgrade, deployment redesign and generalization remain later.

Write findings/decision/resume, local commit and verified full archive, preserving
all previous failures and exact snapshots. Private report stays ignored. No paid
Colab, Drive operation, publication, production-default replacement or remote push.
