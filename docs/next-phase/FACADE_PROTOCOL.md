# A1_v1 facade endpoint allowance protocol

Frozen before experiment outcomes, 2026-09-23. User accepted proceeding with
architectural material/form generation, usability evaluated separately. This
comparison changes only facade accounting, preserving all nine families.

facade_endpoint_v1 allowance: cells inside explicitly typed facade entrance blocks
that are six-neighbor face-adjacent to declared existing buildings and permitted.
No dilation; no ground-entrance exemption; no guide/scaffold/model input. Freeze
sidecar annotations with exact cells, scene hash and mask hash for all18 scenes.
These are allowed attachment locations, not a requirement to fill every patch.
The original facade region is a26-neighbor shell; only exact face-contact cells
in the named endpoint patch receive the allowance, not its diagonal surroundings.

Old facade term: relu(sum(p*facade)/max(sum(p),1)-0.15).
New facade term: relu(sum(p*facade*~allowance)/max(sum(p),1)-0.15).
The denominator remains ALL material. This explicit semantic change still permits
ratio dilution; it does not add attachment, circulation or mechanical guarantees.
All budgets, fractions, envelopes, eight other terms and original code unchanged.

A1 pairs every432 T1 target/context record with the revised facade term (864 arm
records). Reproduce baseline terms from saved geometry. Recompute72 necessary
bounds per arm (144 records) using mandatory non-allowlisted facade contact.
Keep the sealed reference invalid. Never call a necessary bound sufficient.

Additional controls, radius6/envelope: four per scene (empty, allowance-only,
legal facade blanket, guide plus legal facade blanket), both arms =144 records.
Blankets with chargeable contact must remain penalized; empty is not successful.
These retain other-family values and binary metrics. An allowance-only field can
pass facade while failing other objectives; report that honestly.

36 T1 occupancy probes x two facade arms =72 gradient records. Save facade
occupancy derivatives, values and norms; finite differences tested independently.
Identical geometry and unchanged other eight terms isolate the intervention.
All weights remain diagnostic; no optimizer updates or production switch.

Acceptance for continuing calibration preparation: all matched baselines agree,
eight other terms unchanged, budget/geometry unchanged, finite gradients, targeted
conflicts measured, unwanted facade blankets penalized, annotations independently
reconstructed from scene metadata. Whether to adopt remains an evidence-based
research decision; success here does not establish learned architectural quality.
