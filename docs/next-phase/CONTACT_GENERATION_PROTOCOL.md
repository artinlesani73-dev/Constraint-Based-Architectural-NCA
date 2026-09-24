# MG2: full contact-aware procedural comparison

Frozen 2026-09-24 before execution. Follows D064/MD1. User authorized continuing.
Use MG2-contact.json and run_contact_generation.py. No generator or evaluator change.

Read all five saved MG1 contexts, masks, domains and original fields from run
20260924T113023Z_24298393f2a5. Match scene, request16/24/32% and seed0/1/2 exactly:
45 paired cases. The four MD1 pilot members are included, not new independent
confirmations. These remain development scenes, not held-out generalization.

Use contact_cube_route_growth_v1, cube2.4m, cost12 in route and growth, the same
legal domain and unchanged MT1. Verify source hashes against MD1. No reroll,
weight search, threshold tuning, repair or larger grid. Preserve every field,
route, selected cube trace, original comparator and per-family outcome. Rescore
all45 originals before comparison. Four prior analytical probes need no rerun
because neither generator version changes evaluator semantics.

CPU, two Torch threads,15s cooperative candidate cap and600s cooperative study
cap. Boundaries check time; a timed-out candidate retains partial geometry/status.
Log raw generation/evaluation costs separately. Persist failures and incomplete
matrices with unique IDs; retries link their parent. No paid compute or Drive.

Studio integration admission is deliberately stronger than the four-case pilot:
all36 nonblocked cases must pass MT1, preserve all27 original passing cases,
meet their requested voxel count with less than one cube of overshoot, and have
no timeouts. All nine blocked cases remain in the denominator and are reported
separately. Request fidelity is an operational goal, not a tenth constraint family.
Failure of this gate does not authorize a new coefficient search or a weaker gate.

Report per-context/request pass counts, baseline regressions, repaired cases,
all family failures, request errors, contact fraction, runtime and valid-only
diversity. Preserve both generator versions and MD1's negative direct result.
If the gate passes, integrate a separately labeled, experimental mass-generation
workflow with saved evidence and explicit failure states. It must not relabel old
material records or promise validity for arbitrary edited scenes.
