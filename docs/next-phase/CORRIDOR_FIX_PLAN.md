# The next corridor correction

Prepared during E0_v1 on 2026-09-23. This is an implementation design, not a
completed correction or a new experiment result. The running baseline still uses
the original operator from its archived source snapshot.

## Exact defect and scope

In `deploy/model_utils.py::compute_corridor_target_v31`, the vertical-envelope
loop reads from `corridor_dilated` and writes each new row back into that same
tensor while scanning upward. An activated row can therefore activate the next
row, which activates another beyond the requested radius. A final height-band
clamp using `corridor_z_margin` limits the result afterward; the defect is
unbounded propagation within the intermediate volume, not a claim that the
final output always fills the whole grid. The checkpoint margin is 3 voxels.

Repair only this step in a new version. Preserve centroid extraction, MST/nearest
neighbor edge choice, distance-field neighborhood, corridor slack, initial
isotropic dilation, fallback with fewer than two centroids, height-band clamp
and final existing-building mask. The width parameter already affects dilation
in all three axes; the vertical envelope is an additional operation. Do not
quietly turn the width dilation into a horizontal-only operation.

## Proposed operator

Read the fully dilated tensor from an immutable input. For each z, return the
maximum over original rows [z-radius, z+radius], clipped to the volume bounds.
A single depth-only max-pool with kernel (2*radius+1, 1, 1), stride 1 and matching
depth padding implements this on nonnegative corridor fields without scan-order
dependence. Radius zero must return an independent copy with identical values.
Apply the existing height-band clamp and existing-building mask afterward.

Do not obtain this by calling the old function with radius zero and then
expanding its output: that output has already been clamped and masked, changing
the order of operations. Extract or copy the minimal pre/post envelope code with
explicit version provenance and preserve the old callable as the replay oracle.

## Acceptance evidence

1. A one-row impulse expands by exactly the declared radius, including at both
   volume boundaries, and has no propagation beyond that radius.
2. Reversing the z axis, applying the operator and reversing back gives the same
   result; batch elements and channels do not bleed into one another.
3. Compare against a simple out-of-place sliding-window oracle on small random
   nonnegative tensors and several radii. Radius zero preserves values.
4. Target generation with radius zero agrees with the legacy target function.
   Both functions preserve shape/device/type, input state and existing masks.
5. Record target differences for all 18 frozen scenes: added/removed voxels,
   height ranges, entrance contact and legality. Keep raw old/new targets.
6. Only afterward run matched rollouts with old/new targets, preserving settings,
   seeds and measurement definitions. Do not call a smaller target a better
   architecture without that evidence.

The distance field currently grows using a full 3x3x3 neighborhood (26 neighbors)
while the independent endpoint metric uses six face neighbors. That mismatch is
another explicit design question, not something to change inside the vertical
bug fix. No additional constraint family is needed to make either decision.
