# Loss repair: next implementation gate

Prepared 2026-09-23 after the corridor code audit. This is a concrete engineering
plan, not a completed training implementation. Keep the notebook, historical
fine-tuner and checkpoint intact. Introduce a separately versioned shared loss
package and a diagnostic runner before launching any optimizer experiment.

## Start by checking compatible objectives

The legal shortest-path scaffold is a procedural baseline. It is not yet a
complete replacement training recipe or an architectural design envelope.
The historical trainer instantiates SparsityLossV31 with its default 3% lower
and 12% upper mass ratio, and divides by all non-building voxels. Coverage asks
for filled target volume while spill discourages material outside it. Therefore
check each target's maximum zero-spill volume against that lower budget before
reusing those losses. The post-hoc C1 budget audit records this calculation from
the experiment's archived notebook and arrays. Do not silently lower the mass
budget or widen targets to make the numbers pass.

Maintain two declared quantities: the routed centerline/scaffold used for
connection service, and the region in which additional material may develop.
Coverage/spill semantics need explicit versioning. A low-volume scaffold result
is useful evidence; full scaffold filling is not the final research objective.
Do not introduce new objective families or present material paths as walkways.

## Preserve nine families with tested meanings

| Family | Work before training | Required diagnostic |
|---|---|---|
| Legality | Share the declared permitted field; keep hard material projection. Separate pre-projection violations from post-projection validity. | Frozen context unchanged; zero forbidden material after projection; do not require a nonzero post-projection legality gradient. |
| Coverage | Use feasible, nonempty targets, retain infeasibility explicitly, and version route-service versus full-fill objectives. | Missing legal target material changes the objective in the right direction; no target in forbidden cells; empty target is not a successful case. |
| Spill | Declare the material envelope and normalize per scene before averaging. | Outside-envelope material increases spill; no hidden batch-size scaling; inspect conflict with mass floor. |
| Ground openness | Use the protected street region, independently of the fraction of ground material lying in the target. | Empty protected region is open; obstruction worsens the metric. Hard projection may make the associated training penalty redundant. |
| Thickness | Replace sigmoid background leakage with zero-background occupancy; specify erosion padding and physical units. | Empty material is invalid overall, not maximally thick; thin/thick synthetic forms rank correctly; finite gradients at nondegenerate occupancies. |
| Sparsity | Retain lower and upper budgets in a declared denominator; diagnose incompatibility before choosing any new limits. | Under/within/over-budget cases, different scene volumes, and single-versus-batched agreement. |
| Facade contact | Preserve contact intent with a declared neighborhood and per-scene normalization. | Contact/non-contact fixtures; no empty-denominator reward counted as a successful design. |
| Access | Keep binary six-neighbor material connectivity as a spatial diagnostic; separately resolve street-void and elevated material/clearance semantics before claiming architectural access. | Designated source must reach other IDs; separated entrances fail; endpoint height is respected. A soft surrogate must not seed every endpoint then score its own seeds. |
| Load path | Name and test geometric support; prevent empty-space transmission. Preserve explicit support boundary. | Floating/attached components and diagonal cases; no structural-safety claim. |

Cantilever remains a support-related surrogate; density/binarization and TV
remain regularizers. Do not import the fine-tuner's additional porosity/surface
objectives or changed budgets as if they preserved the original task.

## Shape, gradient and recovery acceptance

1. One tensor contract: occupancy [B,D,H,W]; add exactly one channel axis for
   pooling. Preserve B for B=1 and B>1. Assert shapes instead of broad squeeze.
2. Return per-scene terms and validity flags before reduction; disclose empty or
   impossible cases. Mixed batches must agree with separately evaluated samples.
3. Use continuous surrogates for optimization and independent thresholded fields
   for reporting. A hard comparison cannot be relied on for occupancy gradients.
4. Test finite forward/backward and intended gradient direction on asymmetric,
   non-saturated fixtures. Use finite-difference checks away from max/min/ReLU
   ties; record zero gradients when mathematically expected. Test contribution
   through several real NCA updates as well as direct occupancy gradients.
5. Record term magnitudes, occupancy/model gradient norms and pairwise gradient
   alignment before changing weights. Keep model architecture/scene distribution
   fixed during this gate. Do not use adaptive weighting to hide incompatible
   definitions.
6. Only then prepare a tiny local optimizer/recovery smoke test, saving model,
   optimizer, scheduler, RNG, sampled scenes and update count. Compare interrupted
   and uninterrupted continuations before a Colab job.
7. Colab requires an explicit compute cap and a user-approved artifact procedure.
   Drive access is still disallowed without approval for the exact operation.

No paid training, new deployment defaults or larger-grid experiments are part
of the current corridor milestone. Product work follows stable result contracts
and should show the real measured limitations, including infeasible scenes.
