# MO1 - Massing residual endpoint and gradient admission audit

Frozen before audit, 2026-09-24. R2-B preparation after MG1. No optimizer or NCA
training in this study. massing_residuals_v1 is CPU float32/64, with per-scene terms
and independent batch evaluation. All nine MT1 families and thresholds stay fixed.

## Definition

Occupancy p is in[0,1]. m=sum(p), d=max(m,1), V=fixed domain cells. Legal/domain
projection q is used for bulk/connectivity only; raw p remains in all-volume counts
and violation accounting. Soft bulk is a min over complete physical-scale cubes,
then a max over all cubes containing each cell. At binary endpoints this equals
MT1's cube opening, including even widths and empty exterior padding.

Access sums raw and bulk interface deficits (1 minus component_bottleneck_v2
strength), raw and bulk component-excess/d, and outside-available occupancy/d.
Component-excess uses descending six-neighbor activation: each merging younger
peak contributes birth minus merge level; disconnected surviving domain peaks
except the global eldest contribute birth. At binary endpoints this is k-1 for
k>0 occupied components and0 for empty. Thus a connected interface component cannot
hide satellites. No multi-origin flood is used to merge disconnected candidates.

Coverage=max over fixed X thirds of relu(.08 - bulk_fraction_in_third).
Facade=relu(non-exempt-contact/d - .15). Ground/legality/spill are respectively
blocked-protected/illegal/outside-domain occupancy divided by d.
Sparsity=relu(.08-m/V)+relu(m/V-.40).
Thickness=relu(.90-sum(bulk)/d).
Support=relu(1-m)+sum(relu(p-exact_widest_path_strength_from_fixed_support))/d.
Fixed support transmits1 matching the historical geometric-support flood.

The domain/thresholds/contact masks are prepared once from scene context. A context
must have nonempty X thirds. No weighted objective or requested-volume term is
selected by this audit. At binary endpoints, residual<=1e-10 should agree with each
MT1 family verdict, including empty, invalid and blocked controls. Continuous zero
does NOT certify a thresholded field, and the nine residual scales are not equal.

Graph sorting/path selection is detached, with live tensor gathers supplying
piecewise derivatives. Exact ties are nonsmooth, with deterministic choices;
do not claim a unique derivative there, global smoothness, broad nonzero-gradient
coverage, or an efficient GPU training implementation. Finite differences are
tested on untied continuous inputs. This first CPU implementation favors correctness.

## Frozen evidence and checks

Replay all48 MT1 controls from20260924T102755Z_e89550a24d8d and all49 MG1 fields
from20260924T113023Z_24298393f2a5:97 fields,873 family comparisons. Compare bulk masks
to archived masks. Do not change fields/thresholds if a mismatch occurs.
On MG1 aligned/partial_obstruction v24 seed0, use soft p=.005+.99*binary_field as
an explicit gradient probe (including background, not a valid generated design).
Record each family residual and gradient norm, finite values, and a centered
finite-difference check of the facade derivative along its normalized negative
gradient at epsilon1e-5. These are derivative probes, not optimization results.

Run focused operator tests and the complete regression. Study CPU2threads,
cooperative wall cap600s checked between cases (a case or snapshot may overrun
before the next boundary). Save all partial results and finalize interrupted on
cap. Register every failure with unique parent-linked attempt. Save source hashes,
effective config, source field IDs/hashes, all residuals and gradient summaries.

## Exit

If endpoints/bulk/gradient checks pass, prepare a bounded direct-optimization pilot
with explicit initialization, coefficients, requested-volume policy, binary check
schedule, recovery and timing admission. Do not launch it without those mechanics.
MG1's facade failures motivate the comparison but do not justify a weaker limit.
Include a contact-aware procedural comparator to avoid overstating optimization's
benefit against a generator that does not account for contact. User liked MG1 forms;
retain them as baseline, not as proof of final architectural quality.

No changes to historical model/losses, MT1 evaluation, MG1 generator or gallery.
No paid compute, Drive access, publication or push.
