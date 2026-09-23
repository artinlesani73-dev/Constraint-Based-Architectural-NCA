# K1_v1 model-gradient calibration diagnostic

Frozen before K1 outcomes,2026-09-23. No optimizer updates, no architecture change,
no paid compute. Original checkpoint/weights stay immutable. The experimental
objective is hard_preclamp coverage, radius6/envelope3%-12% material budget and
facade_endpoint_v1. All nine families retained. Hard legality/ground projection
means their parameter gradients may correctly vanish.

71 model cases:17 feasible frozen scenes x firing seeds0/1 x horizons4/16 =68;
plus legacy000,ground-pair,minimal at seed0/horizon50 =3. These extra cases probe
original-training-scale horizon, not all-scene long-run generalization. Exclude the
sealed reference from optimization diagnostics explicitly and record its ID.
Save13 per-term parameter gradients: nine families, notebook density-binarization,
notebook TV sum, new boundary-aware cantilever and original cantilever diagnostic.
Both cantilever variants are recorded, never silently summed as one recipe.

51 direct material-budget probes:17 scenes x occupancy0.015/0.075/0.20 throughout
the legal radius-six envelope. These set mass ratio below/inside/above3%-12%.
Record all raw term/regularizer values and direct sparsity gradients. Use projected
coverage for static probes. No adaptive coefficients or optimization. This matrix
activates budget branches that T1's sole probe missed.

Pre-clamp diagnostics: record guide raw fractions below0,within[0,1),at/above1,
maximum overshoot, missing coverage, and raw coverage-gradient norm. Expected
hinge saturation above1 is not corrected by a straight-through estimator.

For each model case save material/raw fields, all parameter vectors/names, norms,
pairwise cosine matrices, binary metrics and frozen/legality checks. Use explicit
firing RNG, two CPU threads, preserved historical0.15 scaffold seed. Unit values
are diagnostics, not selected training weights. Check weights unchanged at end.

Regularizer audit source: notebook cell19 and trainer cell24. Checkpoint's saved
weights exactly match the notebook table. Density=p(1-p) mean; TV=sum of three
axis means (fine-tuner differs). Historical cantilever uses previous3 layers,
3x3 horizontal max, sigmoid10*(support-0.3), skips bottom3 layers and ignores fixed
support context. Retain that exact path only as diagnostic. New geometric proxy
uses all layers, explicit fixed support, continuous maximum strength below, and
zero out-of-grid support; it is not a structural load calculation or span limit.

Acceptance: finite measured gradients/terms, every scene/case retained, intact
frozen context, no forbidden material, immutable weights, preserved archive hashes.
Analyze actual scales/opposition/zeros before selecting a fixed small sensitivity
recipe. Numerical consistency is established by W1; learned value remains untested.
