# Retained regularizer audit

Primary sources: notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb cell19
(definitions) and cell24 (trainer), plus v31_fixed_geometry.pth saved weights.
K1 records source hashes, exact extracted definitions and checkpoint coefficients.
Checkpoint coefficients match the notebook table exactly. This supports recipe
attribution, not reconstruction of every historical optimizer step.

| Term | Original notebook | Later fine-tuner | New diagnostic path |
|---|---|---|---|
| Density | mean(p*(1-p)), weight3 | upper-density penalty | notebook-faithful per-scene binarization |
| TV | sum of z/y/x mean absolute differences, weight1 | divides sum by3 | notebook-faithful per-scene sum |
| Cantilever | sigmoid support from prior3 layers and horizontal3x3, weight5 | compares directly below; different formula | retain exact notebook diagnostic plus explicit boundary-aware candidate |

The original cantilever ignores the bottom3 layers, ignores declared building/
ground support, gives positive support strength sigmoid(-3) in empty neighborhoods,
and never represents a true maximum horizontal span despite its name. The later
fine-tuner is not its faithful replacement. Keep both historical sources intact.

regularizers_v1 ports density,TV and original cantilever with per-scene outputs.
Tests compare values and gradients against extracted notebook classes at batch2.
Boundary-aware cantilever is a separately named geometric proxy: every layer
participates, fixed support can contribute strength1, max material/support over
previous3 layers and a3x3 horizontal stencil supplies continuous support strength,
outside-volume support is0, and current fixed-boundary cells are supported.
Loss is mean(p*(1-strength)). This does not prove load-bearing safety or allowable
cantilever length and is not introduced as a tenth constraint family.

Numerical tests cover finite differences, batch independence, exact notebook
parity, floating/ground-supported examples, no wraparound and empty material.
Verification20260923T092058Z_bba721fc054d:137 tests pass, smoke exit0.
K1 measures both cantilevers independently. No recipe silently sums them or uses
historical coefficients as if corrected terms had identical scales.
