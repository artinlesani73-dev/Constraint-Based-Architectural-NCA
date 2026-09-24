# SP1: floor, usable surface and clear volume

2026-09-24. Authorized local prototype following SPATIAL_BRIEF and the user's
approval to proceed. No training or changes to historical objectives. The recipe
is `experiments/configs/SP1-platform.json`; evaluator version is
`spatial_platform_v1`. These are deliberately chosen development examples, not
unseen-site validation or evidence of learned generalization.

## A concrete architectural target

Two aligned external facade approach regions connect across a level deck with a
wider shared landing. Existing building boxes remain solid context. The approach
markers designate external destinations; their 2-cube extents do not specify real
doors. Interior rooms, openings and routes inside buildings are not modeled.
This is an outdoor spatial study, not a complete habitable building.

| Quantity | Cells | World dimension |
|---|---:|---:|
| Voxel edge | 1 | 0.8 m |
| Deck surface elevation | 8 | 6.4 m above origin |
| Floor depth | 1 | 0.8 m, coarse concept geometry |
| Connecting deck width | 3 | 2.4 m |
| Shared landing | 5 × 5 | 4 × 4 m; 16 m² |
| Required clear height | 3 | 2.4 m |
| Site grid | 32³ | 25.6 m on each side |

These dimensions are a research brief, not code requirements or an engineered
slab design. There are no railings, ramps, stairs, drainage or mechanical analysis.
In particular, the one-cell depth must not be interpreted as a practical slab
thickness. Finer geometry can follow after the semantics are established.

For the aligned example, the floor occupies z=[7,8), the surface is the plane z=8,
and clear volume is z=[8,11). All coordinates below are half-open voxel extents.

```text
PLAN (looking down)                SECTION (through the passage)
west context       east context    z=11  ───── top of required clear zone
████│   ┌─────┐   │████                    clear volume; no material/context
████├───┤4 × 4├───┤████            z=8   ═════ usable floor surface
████│   └─────┘   │████            z=7   █████ proposed floor material
       2.4 m strips                       open space below
```

The interactive Studio spatial page provides actual voxel plan and section
drawings from the saved run. Clearance is a translucent derived volume, not
additional generated material. It must never be counted in the material budget.

## Representation and independent evaluation

Keep one boolean material grid for this bounded prototype. Derive a floor mask
where all declared depth cells below the specified surface plane are occupied
by proposed material. Derive clearance from BOTH material and existing context.
A surface cell qualifies only when the floor exists and the complete required
column above it is empty. Missing grid cells are not treated as empty headroom.

A 3×3-cell square footprint must fit wholly on those clear surface cells. Erode
the surface in XY to get allowed footprint centers. Four-neighbor connectivity
of those centers must link both external approach regions. The erosion treats
outside-grid cells as unavailable. This is conservative square-footprint traversal
on one horizontal plane, not a disk-shaped person, arbitrary slope or vertical path.

The 5×5 landing must be fully clear and its interior footprint centers reachable
from the same source component. Require nonempty material, no context collision,
an entirely open street band and geometric attachment of all material to existing
context. Attachment is not load capacity. The stricter street-band check here has
no anchor exception; historical ground metrics remain separately reported.

The reported spatial gate combines those checks only. Legacy budget compatibility
and the unchanged nine penalties are reported separately; spatial gate success
does not mean all nine old objectives or a complete architectural brief pass.

## Same nine families; no silent replacement

| Existing family | SP1 treatment |
|---|---|
| access | Additional versioned footprint traversal and clear-height diagnostic; retain old occupied-entrance access score |
| coverage | Explicit usable landing coverage/reachability; retain historical guide coverage score |
| thickness | Declared floor depth and route footprint width; retain old bulk penalty |
| support | Geometric attachment only, with original support penalty also reported |
| legality | Context collision and historical permitted-region audit |
| ground | Clear street band diagnostic and unchanged historical ground penalty |
| facade | Unchanged historical facade penalty/allowance; no invented safety interpretation |
| sparsity | Unchanged historical 3–12% fixed-envelope budget audit |
| spill | Unchanged historical spill penalty |

No new differentiable objective, coefficients, budget percentage or trained model
is introduced. Legacy material access asks for occupied entrance blocks; the new
surface interpretation asks for an empty approach above a floor. Both can disagree
on the SAME geometry. Report the conflict rather than filling the passage with
material to improve the old loss or declaring the new score numerically better.
The envelope is derived from the original scene route at radius6, never enlarged
in response to the proposed floor to make the material ratio pass.

Derived masks suffice to represent this one-level example. That does not establish
that one material channel is sufficient for multilevel rooms, stairs, roofs or
heterogeneous materials. Explicit surface/void channels remain a later comparison.

## Frozen construction and negative examples

Construct straight strips between aligned approaches and a centered 5×5 landing.
Keep the full proposed field even if it collides; never clip a collision into a
claimed success. Unsupported approach levels return an empty field plus an
explicit unsupported-layout status. That is not a proof of global infeasibility.

SP1 pairs each of six cases with W1 on the same scene:

1. Aligned approaches: expected spatial pass.
2. Narrow strip away from landing: width failure, despite material continuity.
3. Missing floor cross-section: access failure.
4. Context beam leaving1.6m clearance: headroom failure.
5. Full-height partition through the span: collision/blocked-route failure.
6. Different approach heights: unsupported level-deck construction.

Narrow-strip and missing-floor cases are deliberate candidate ablations; they
change geometry, not scene context. All candidates, failures, masks and paired
old scores are retained. The expected labels are frozen in the config before
running; evaluate outcomes without tuning dimensions to fit observed scores.

Run `.venv/Scripts/python.exe scripts/run_spatial_prototype.py` from the project.
Use `--parent-run <id>` for a linked new attempt after a failure; never overwrite.
The runner preserves source, configuration, scene hashes, seed, environment,
individual geometry and diagnostics in a RunStore. Source snapshots include
working bytes; checkpoint weights are not used, only authoritative configuration.

Next admission gate: review the spatial example, clarify interior/door/vertical
access requirements and audit incompatible old families before any learning.
Do not start Colab merely because a procedural example passes this spatial gate.


## Scope correction - D057, 2026-09-24

This earlier platform/refinement direction is superseded as the research target.
The user clarified volumetric forms with spatial depth and voids, without prescribed
rooms, shelters or functions. Preserve this document as diagnostic/history. Current
scope and next work: VOLUMETRIC_AUDIT_FINDINGS.md and VOLUMETRIC_NEXT_PHASE.md.
