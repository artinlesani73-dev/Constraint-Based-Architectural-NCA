# Building mass first; interiors later

Accepted 2026-09-24, D058. This is the current research brief.

## Confirmed meaning

The user suggested filling the blue cells after generation or during it. Asked
whether they meant "generate the overall building mass now, and leave its internal
spaces and construction for later", the user answered "yes".

Future occupied voxels represent building volume, not solid construction material,
walls or floors. Their physical measure is gross occupied voxel volume in cubic
metres, not material quantity or floor area. This does not establish habitability.
Interiors, construction and architectural programs remain later work.

The target still requires three-dimensional depth; a thin path or platform is
insufficient. Internal cavities are no longer required to demonstrate potential
interior space. Exterior gaps, setbacks and separation can remain meaningful.
This decision does not require filling every empty cell or the entire site.

## Preserved evidence

SP1 and VA1 retain their original definitions and results. VA1 green represents
material in analytical probes; blue marks axis-bracketed empty cells, including
some exterior-connected space. Blue is not an interior classifier or a filling
instruction. The current gallery is unchanged and is not a newly trained model.
Do not relabel old fields or reuse old acceptance claims under the new semantics.
More bracketed void is no longer a primary success criterion for the new brief.

## Next bounded local implementation

1. Define a versioned massing contract: building occupancy, fixed scene-derived
   3D opportunity region, world units, context exclusions and declared interfaces.
   Keep historical scene/material contracts readable.
2. Save original fields alongside derived massing candidates. Include closed-shell
   cavity filling and open-form controls. Closed free components are geometrically
   identifiable; open forms need an explicit completion rule. Do not fill every
   blue cell. Record parent field, operation, parameters and every added cell.
3. Label original material probes versus proposed building mass in the viewer.
   Filling must be labeled derived postprocessing, not learned generation. Save
   and evaluate both results; keep the original study unchanged.
4. Audit the same nine families before changing losses. Legality/spill must detect
   additions outside the domain or inside context. Review access as connection to
   declared interfaces without claiming internal circulation. Coverage should
   express spatial distribution. Sparsity needs a fixed-domain building-volume
   budget; the old material budget is not automatically suitable. Review thickness
   against volumetric depth rather than rejecting all filled bulk. Retain old
   ground/facade/support reports while checking their suitability. Geometric support
   does not certify structural safety.
5. Freeze acceptance examples: thin sheet, narrow path, solid block, disconnected
   fragments, context collision, excessive site fill and articulated mass. A solid
   block is not invalid solely because it lacks cavities, or good solely because
   it occupies three dimensions.
6. Compare procedural and direct-optimization massing controls before specifying
   an NCA trial. Compare direct mass generation with a documented completion step,
   using equivalent final-output evaluation and separately retained raw outputs.

These steps are planned, not implemented. Filling is not yet the chosen generation
method. No numerical budget, depth threshold, loss, channel count or training
configuration is frozen here. Keep nine families; no tenth constraint.

Preserve all old fields, metrics, source snapshots and archives. New derived fields
get new records with parent links. Resume from D058 and this brief. No automatic
paid training, Drive access, remote push or publication follows from this decision.
