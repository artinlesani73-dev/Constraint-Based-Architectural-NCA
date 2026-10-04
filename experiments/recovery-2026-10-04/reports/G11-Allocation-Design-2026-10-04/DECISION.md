# G11 allocation design decision — 2026-10-04

## Decision

Select one local, fixed-weight HYBRID capacity-reservation prototype before
further paid training. Keep learned reversible growth as a separate research
option. This is a change to admission architecture, not a new constraint family,
and not proof that NCA alone learned to plan connections.

G10 passes stability on 69 cases but has 18 persistent access/coverage failures
at exhausted capacity. That motivates allocation control; it does not establish
that reversibility or reservation will solve the full task.

## Comparison

| Dimension | Learned reversible occupancy | Capacity reservation |
|---|---|---|
| Mechanism | Learn additions and removals to relocate mass | Reject additions that consume capacity needed for a feasible completion |
| Existing weights | New removal behavior is untrained | Existing proposal scores can be retained |
| Connectivity | Removal can split a connected mass | Monotone connected cube additions retain connectivity |
| Thickness | Voxel removal can cut thin fragments; cube-union representation or safeguards needed | Existing full-cube admission retains depth |
| Budget | Add/remove arbitration must account for overlap | Count exact occupied union with reserved witness |
| Locality | Learned rule can be local; exact global checks still are not | Explicit global planning and admission; hybrid |
| Main risk | Oscillation, destructive removals, difficult credit assignment | Planner dominates shape, rigid routes, planning cost |
| Evidence needed first | Safe reversible transition and recovery; then new training | Geometry feasibility and paired fixed-weight local rollout |
| Decision | Defer rather than combine architecture and training changes | First bounded prototype |

These are engineering assessments of this implementation, not a literature
benchmark or a claim that reversible NCAs are generally inferior.

## Completed TRAIN-only feasibility audit

Verified all 45 input NPZ hashes against the earlier recorded TRAIN manifest.
Read only the seven-channel condition arrays; teacher target and teacher
distances were not used. No model inference, optimizer update or held-out
evaluation occurred.

For each condition:
1. Build a graph of legal 3x3x3 cube origins, with unit face-neighbor edges.
2. Breadth-first search from origins whose cube intersects the east interface.
3. Select a shortest-hop reachable seed-covering cube, with deterministic
   lexicographic ties, and follow decreasing distances.
4. Save the exact cube union, route origins, distances and input context.
5. Verify legal occupancy, seed contact, east contact, voxel connectivity and
   full-cube thickness.
6. Count route volume and remaining cap, plus optimistic missing coverage
   voxels in each fixed site third using the existing 8% requirement.

Results: 45/45 legal thick routes fit the cap. Route volume is 144–189 voxels;
minimum spare capacity is 365 voxels. In 45/45, route plus the sum of coverage
voxel deficits also fits. These are 45 correlated requests from the existing
training scenes, not 45 independent sites.

The saved route is an actual access witness, not a shortest-volume proof.
The coverage calculation is only a lower bound: whole-cube overlap,
connectivity, facade and support may require extra volume. It is NOT a joint
nine-family feasibility certificate. No learned-model quality claim follows.
No 64-step completion or stochastic-firing guarantee was tested.

## Concrete next prototype: G11-R1

Scope: a new isolated inference adapter with fixed G10 weights and unchanged
nine-family evaluator, requests, firing seed, horizons and global cap.
Keep original G10 raw inference unchanged. Label every derived artifact
'hybrid capacity reservation'; include parent checkpoint and adapter hashes.

Implement before any paid training:
- Build a connected full-cube completion witness from site geometry. The first
  route implementation is available in audit.py. Extend the witness to meet
  existing coverage and check ALL nine families plus requested volume error.
  Save any unsuccessful search as 'no certificate found'; do not call it proof
  that the scene is impossible, and do not return an incomplete route as success.
- Let learned scores propose additional eligible cubes. For candidate union F',
  require |F' union W| <= C, where W is the remaining certified completion
  witness. This exact union count is an upper bound for that chosen completion,
  unlike a shortest-path distance or a coverage voxel lower bound.
- Check other constraints on the prospective union as well: preserving capacity
  alone does not preserve facade ratio, support or every acceptance condition.
- Reserve progress slots within the existing per-step quota. Do not defer all
  bulk growth until contact: the earlier G9 audit already showed that 37/45
  particular connection-first oracle trajectories could not reach requested B
  by 64 even with optimistic remaining quota.
- Initially keep one deterministic witness and record its cost and tie-breaking.
  Replanning, route diversity and resolution scaling are later changes.
- Explicitly separate learned accepted additions from planner additions.
  Record their fractions, blocked proposals, planning time and remaining
  reserved volume at each step.
- If the witness schedule cannot finish within the original quota/horizon,
  record a failure. Do not silently increase quota, change firing, extend
  horizon, relax constraints or force a final postprocessing repair.

A planner completing mandatory witness cubes is a procedural component.
If such completion bypasses learned score/firing, declare that change explicitly
in the adapter protocol and paired comparison, before execution. It cannot be
described as unchanged NCA inference. At this design milestone that scheduling
choice is not implemented or validated.

## One consolidated local acceptance exercise

First exercise geometry and transition invariants on TRAIN, including blocked
routes, cap shortage and a critical connectivity case; keep all failures.
Then compare raw G10 and G11-R1 on exactly the same 45 TRAIN inputs, 64/128
steps and firing seed 2101. Preserve states, birth provenance, case metrics,
runtime and planning overhead. Evaluate all nine families, volume error and
stability. A connection gain with coverage loss is not success.

No new paid run is necessary for that comparison. Do not retune repeatedly on
the 69 exposed evaluation cases. Only after selecting/fixing the adapter should
a new frozen held-out set be created for a single subsequent assessment.
Do not imply held-out success from this TRAIN geometry audit.

## Scope and preservation

Building volume with meaningful depth remains the target; rooms, cavities,
construction and structural certification remain outside this phase.
MG7 remains live; no model is promoted. No Drive operation or paid run occurred.
The original review/report is untouched. Repository synchronization remains
pending; these output records supersede its stale D098 resume for this work.

Input contexts, exact route evidence, source snapshot, audit code, results and
this decision are retained in a manifest-verified local archive. A same-disk
archive is not an off-device backup. Read RESUME.json here to continue.
