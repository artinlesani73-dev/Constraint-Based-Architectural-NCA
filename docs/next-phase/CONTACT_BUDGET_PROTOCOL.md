# MG3: contact-budget growth protocol

Frozen2026-09-24 before the45-member comparison; user authorized proceeding.
Implements CONTACT_BUDGET_NEXT_PLAN as budgeted_contact_growth_v1. No old generator
or MT1 threshold changed. Seven focused mechanics tests pass before execution.

Keep MG2 routes and frozen cost12 ranking. Assemble the full initial route first;
retain and label route_contact_budget_exceeded if its global contact fraction is
already above MT1's15% limit (+same1e-10 evaluator tolerance). Do not apply the
growth rule to route prefixes. No reroute or weight search is introduced.

At each growth proposal count unique newly occupied and newly contacting cells.
Accept only when the resulting total-contact/total-mass ratio stays within15%.
Deferred origins retain their priority. Requeue them only after accepted positive
geometric growth. Zero-delta accepted origins expand once, without requeuing
deferred entries. Unique queued-origin membership prevents repeat expansion.
When the heap is empty before the request, return contact_budget_stalled if any
origins remain deferred, otherwise component_exhausted. Preserve partial fields,
all acceptance/rejection records and pending deferred origins. This is finite
greedy search, not an existence/optimality proof. No clipping or new family.

Exactly45 original MG1 contexts/requests/seeds and saved MG2 counterparts. Rescore
both prior sets; retain original and previous comparisons. Same masks,2.4m cubes,
16/24/32% requests, seeds0/1/2, unchanged MT1;CPU,two threads.15s cooperative
candidate cap,600s study cap. Timeouts retain partial evidence. Source and failed
attempts archived, unique linked IDs; no concurrent benchmark/regression timing.

Admission unchanged in intent: all36 nonblocked outputs must pass MT1 and request
fidelity (<one cube overshoot); no regressions against MG1 or MG2, no timeouts.
Keep all9 blocked cases in the full denominator. Report unchanged geometry versus
changed outcomes, valid-only diversity, deferred growth and request shortfall.
Contact validity is now partly by construction; still independently score all nine.

This experiment tests the revised generator. If admitted, the next product phase
integrates it as an explicit experimental mass workflow with durable jobs and
versioned records; no existing material record is relabeled. No learned model,
paid Colab, Drive action, remote push or publication is part of this experiment.
