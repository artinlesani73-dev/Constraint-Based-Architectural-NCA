# F4: one raw-access learning change

Frozen 2026-09-23 before any F4 run. User approved proceeding after A3.
Candidate raw_component_objective_v3 changes only the access-family score to
relu(1-b_raw). Baseline mode is component_objective_v2 and must exactly replay
historical F2 training. Nine constraint families, three existing regularizers,
material budget, architecture, original initialization, two scenes, both recipes,
training seed0,64 updates and constant16 growth all remain fixed. Both arms use
1024 recurrent training steps/model; scoring overhead can differ.

The draft plan's update4 evaluation is corrected to frozen F2's actual update3.
Boundary evaluations are0,1,3,8,16,32,64 at16/50 growth, firing seed2. Full final
grid: horizons16,24,32,40,50,64 times seeds0,1,2 for all four models. Reuse the
eight overlapping boundary fields explicitly:56 boundaries plus72 grid records
represent120 unique evaluations. Save every update's pre-update fields and loss,
post-update full checkpoint, optimizer/scheduler state and RNG, including cursor.
All evaluations retain old, projected-component and raw-component access scores.

## Ordered gates and exact allowances

1. Foundation regression, then freeze source commit/snapshot.
2. F4B: three actual projected-objective updates/model. Require12 update records
   and24 evaluations to equal F2 exactly in historical trace fields, forward
   fields and full checkpoint/RNG trees except declared protocol/source identity.
   Additional raw evaluation fields do not change historical fields. Worker180s,
   coordinator900s. Original F2 interrupted parent/timing issue remains recorded.
3. F4R: raw objective, mass_3 ground-pair. Whole0->3, prefix0->1, resumed1->3 and
   repeated1->3 in fresh processes. Compare all fields, full checkpoint/RNG
   trees and evaluation traces. Worker180s, coordinator900s. At least11 exact
   checks. Every run records objective identity and rejects mismatched recovery.
4. F4P: two raw-objective updates for each of four models, complete final grid.
   Worker240s, coordinator1200s. Timing-only admission:
   per_member =1.5*(max(5,max(setup)+3)+64*max(update16)+7*max(pair)+max(extra_grid)).
   Require per_member<=900s and four*per_member<=2400s. No outcome-based gate.
5. F4: conditional full64 updates/model only after all verified gates and exact
   metadata/code equality. Worker900s, coordinator2400s. An elapsed overrun is
   a protocol violation even if process.wait reports a successful exit. Preserve
   partial evidence and stop; never silently change caps or restart from zero.
6. F4L: preregistered supplementary trained-state recovery from all four completed
   update62 checkpoints through63/64, comparing full checkpoints/trace/fields
   and final16/50 evaluations exactly. Worker120s, total360s. No extra learned
   exposure. This verifies trained-state recovery after the bounded study; early
   real-loop restart is the pre-study gate. GPU/AMP/abrupt-write recovery remains
   uncertified. The late wrapper is included in the frozen source identity.

Reporters rescore every saved field, verify checkpoint cursor/metadata and source
snapshot hashes, compare original outputs, verify56 F2 boundary/72 H1 F2 final
controls, and replay eight final checkpoint rollouts. They share objective
implementations; binary component BFS is independent of maximin selection.

## Decision and preservation

Primary geometry outcome is joint strict>0.5 component connectivity and existing
3–12% material budget (report tolerance1e-6 explicitly). Report F2 connections
retained/lost/gained, mass changes and every other family/metric per case. Loss
reduction alone cannot promote a model. No joint success means no promotion;
any success still needs separately planned training-seed and fresh-scene testing.
Do not combine pools, loss reweighting, larger grids or a new update rule here.

Runs use CPU two threads and deterministic algorithms. Failed attempts get new
linked IDs; snapshots, all results and decisions remain. Update RESUME at each
gate. Finish with a local commit and verified full archive. No paid compute,
Drive operation, deployment or remote push is authorized by this protocol.
