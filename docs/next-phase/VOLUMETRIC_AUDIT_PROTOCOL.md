# VA1: volumetric form and void audit

2026-09-24. The user clarified that this experiment concerns volumes with spatial
depth and voids within/between their parts, without predefined rooms, shelters or
architectural functions. SP1 narrowed that intent to a flat platform. Preserve
SP1 as a diagnostic experiment; it is superseded as the design target. The later
assistant suggestion of a prescribed room/roof/wall assembly is also superseded.

## Corrected target and boundary

Investigate three-dimensional material organization and the spatial voids it
forms. Material, empty space and overall form are distinct. A thick solid block
has extent but no internal void; a few distant cells can fake a large bounding
box. A flat platform with air above it is not sufficient evidence of volumetric
organization. Open voids are allowed; full enclosure is not a required goal.
This does not introduce a tenth constraint or a prescribed architectural program.

VA1 is a non-learning diagnostic audit. Keep the original nine-family objective,
its material-budget denominator and all historical results unchanged. Do not
optimize new metrics, promote a model, scale grids or start Colab in this audit.

## Fixed comparison

Use the same32³,0.8m aligned context from SP1 for every field. Keep its route-derived
radius6 permitted envelope fixed. Proposed analysis domain is x=[8,24),y=[8,24),
z=[6,19); it is fixed before examining outcomes and is not a budget denominator.
Axes are z,y,x in arrays. Shapes are hand-designed probes, not generated NCA forms.

Probe empty field, W1 scaffold, SP1 platform, a512-cell slab, a512-cell solid block,
a512-cell hollow form with two open ends, that form with a25-cell side aperture,
a610-cell closed shell, and eight separated corner cells with the hollow form's
bounding extents. The equal512-cell fields test material arrangement at fixed mass.
They do not hold facade contact, support, coverage or spill constant; report those
differences rather than interpreting the comparison as a causal training ablation.
The script fixes exact cell coordinates and archives all inputs/outputs.

## Diagnostics, not a new aggregate objective

- Material count and volume, bounding extents, and extent balance. These describe
  size/flatness only; the corner-cell control demonstrates why extent is insufficient.
- Empty cells bracketed by proposed material along two or three coordinate axes:
  a cell has material somewhere on both sides of each counted axis. Existing
  buildings do not supply those brackets. Counts are restricted to the fixed domain.
  This detects some open spatial voids, but is axis-dependent and can be fooled by
  disconnected panels; it is not a general measure of architectural space.
- Free-space flood fill from all six outer grid faces using six-neighbor adjacency.
  Report sealed free cells once with proposed material alone and once with existing
  context included. Flood the full grid BEFORE restricting reported cells to the
  domain so the analysis-box boundary cannot create artificial cavities. Sample
  boundaries and context can affect topology; distinguish their contribution.
- For bracketed free-cell centers, count full1³,3³,5³ free cubes. These are descriptive
  clearance samples at0.8,2.4,4.0m cube widths, not human dimensions, walkability or
  acceptance thresholds. The full cube must lie in the grid and outside both solids.
- Material connected-component count. No scalar 'space quality' or winner score.

Show unchanged access, coverage, thickness, sparsity, facade, legality, spill,
ground and support penalties for every field. Specifically inspect whether the
old thickness proxy penalizes filled bulk rather than the overall extent of a
thin-walled volume; do not blame it without measuring. Audit the3–12% cap and
route coverage on actual volumetric probes without changing the denominator.

## Implementation and validation

Separate evaluator, repeatable probe constructor, frozen recipe and append-only
runner. Register the run, seed, effective config, checkpoint-source hash, code
snapshot and each scene/field/metric. Independently test an analytic closed shell,
opened shell, solid block, plane, empty field, bounding-box decoy, rotation of axis
labels, context closure, region-boundary leakage and grid-edge clearance.

Expose actual candidate slices and cutaway geometry in a local volume-study page.
Show the material separately from diagnostic void masks; masks are derived from
each saved field and are not generated architecture. Keep both W1 and SP1 available
as historical controls. Findings determine the next representation/objective task;
this protocol preselects neither room types nor a new loss recipe.
