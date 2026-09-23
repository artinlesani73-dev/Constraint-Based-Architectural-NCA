# T1_v1 target compatibility and occupancy-gradient protocol

Frozen before outcomes on 2026-09-23. CPU only, no model optimization or change to
production objectives. Inputs: hash-verified C1 legal guides/scaffolds, 18 frozen
scenes, unchanged geometry_losses_v1 and budget_contract_v2.

432 target records: 18 scenes x six binary candidates (empty, guide, C1 scaffold,
legal graph radii1/3/6) x envelope radii3/6 x site/envelope mass denominators.
All nine terms, context validity, physical mass, contact fraction, eroded cores,
64-hop proxy scores and independent binary metrics are recorded. Empty is a
negative control. Infeasible scenes remain labeled, never silently dropped.
No assertion that these candidates are architecture or walkable circulation.

72 necessary-bound records: 18 scenes x two envelopes x two budgets. Add the
coverage-implied facade contact bound to previous capacity checks: required mass
at least mandatory facade-contact count / 0.15, and at most budget/capacity.
Also check whether available non-facade capacity can dilute mandatory contact.
Passing is necessary only; failing disproves simultaneous zero values of those
four particular losses under the declared context, not feasibility of every
possible architectural compromise. No bound is silently changed to pass.

36 occupancy-gradient records: all 18 scenes x two budgets, radius6 envelope.
Deterministic per-scene local generator seed1000+sorted-scene-index; occupancy
0.70-0.90 on guide and 0.01-0.05 on remaining legal envelope, zero elsewhere.
Use projected coverage for geometry audit (raw pre-clamp guidance is not defined
for static geometry). Save all nine direct occupancy gradients, legal-tangent
norms and pairwise cosines. A legal-tangent vector zeroes forbidden coordinates;
it does not imply an NCA parameter derivative. Invalid scenes remain diagnostic.
All coefficients one only to expose raw scales; no optimizer or calibrated weights.

Thickness is eroded-core fraction, which penalizes bulk when minimized; it is
not minimum structural thickness. Radius2 uses a five-voxel cube, 4m at0.8m/voxel.
Facade currently penalizes contact above15% of mass, not absence of contact.
Access follows material connectivity, not free-space walkability. Preserve these
meanings and expose shortcomings before proposing a versioned semantic change.

Archive source, protocol, scene/checkpoint hashes, fields, every record, errors and
summary. Independently recheck saved norms/cosines and report counts. Inspect
results before defining E2 gates; do not choose loss weights by inverse norm alone.
