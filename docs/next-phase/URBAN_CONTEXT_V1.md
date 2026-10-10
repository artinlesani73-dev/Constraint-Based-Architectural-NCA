# Urban context v1: representation milestone

Completed2026-10-10. Implements the first step of MULTI_SITE_NEXT_PHASE.md; model generation has not been adapted yet.

Source nca/urban_context.py. Fixture experiments/scenes/urban_context_v1/block-01.json. Four regression tests in tests/test_urban_context.py passed. Archived scene/check log/summary .local-artifacts/urban-context-v1-2026-10-10. Tracked receipt experiments/reports/Urban-context-v1-2026-10-10.json. Check command .venv/Scripts/python.exe -B scripts/check_urban_context.py; refuses existing output, use a new attempt path for continuation. Direct regressions: .venv/Scripts/python.exe -B -m unittest discover -s tests -p test_urban_context.py -v .

## Meaning and conventions

Independent urban_context_v1 schema, not a silent extension of historical scene_v1. Integer XYZ shape, ZYX arrays, Z upward, half-open voxel bounds, explicit world-origin XYZ metres and positive isotropic voxel edge. World cell centre is origin+(index+0.5)*voxel_size_m. Existing buildings and generated occupancy remain distinct; generated cells mean gross building volume. Maximum grid caps are representation/memory safeguards, not architectural constraints.

Axis-aligned fixed buildings, non-overlapping opportunity boxes, protected boxes, named facade interfaces and undirected requested links. Interfaces support X,Y or Z faces, either side, and use a one-cell-thick adjacent exterior patch whose tangential spans fit the declared building face. Entire patch must belong to its declared opportunity region and avoid fixed/protected geometry. This is a geometric attachment, not a door, walkable connection or structural joint. Multiple interfaces and links need not be an all-to-all connected graph. Connection requests state desired relationships; they are not route certificates.

Allowed domain = union(opportunity boxes) minus fixed buildings minus protected boxes. Opportunity boxes may include context but their allowed remainder must be nonempty. Protected geometry expresses explicit exclusions under future legality/spill handling; no tenth constraint family was introduced. No implicit street band, ground/facade/support exemption or learned condition channels are inferred. Their next-phase meaning must be specified before evaluation/training.

Fixture block01:96x96x40cells at0.8m,76.8x76.8x32m extent;3 buildings,2 adjoining opportunity regions,3 interfaces,2 requested links and a protected courtyard. 48,192allowed cells,24,674.304m3 opportunity capacity, not requested/generated volume. 368,640total domain cells:5.76times40-cube cell count. Dimensions are one authored representation fixture, not a frozen training/scale prescription or unseen evaluation set.

## Checks and visual

Tests cover axes/world size, input immutability, independent validated copies, all interface patches allowed, collision/protected exclusions, invalid reference/face bounds/bool dimensions, overlapping buildings/opportunity regions and hash sensitivity to origin/connection changes. Scene hash fcc672b7307460ebfafda452d38a64bc3dc7c4a04ce6fcff27b1e50631679574. Hash covers scene representation including origin and requested links; no generation seed exists at this stage.

Plan diagram scripts/render_urban_context.py saved C:/Users/artin/Documents/Codex/outputs/NCA-Urban-Context-2026-10-10/block-plan.png and visually inspected. Yellow lines are relationship diagrams, not valid routes. Project environment lacked matplotlib; preserved probe failure in this record and rendered with existing bundled Pillow without installing dependencies. Renderer requires bundled Pillow and Windows fonts. This is a static plan, not a new studio renderer.

## Boundaries and next work

No changes to old contract, frozen neural code, checkpoints, studio package, current research store or historical scenes. No architecture/training claims, no paid compute, Drive, cloud, push or MG7 replacement. All9 families remain; there is no evaluator yet for this schema. Schema-valid does not establish feasible connectivity or desirable urban form.

Next define the multi-site evaluation/budget protocol in the existing nine families, separating whole-block resource allocation from individual-region behavior. Add cheap explicit feasible/blocked geometric controls before adapting the generator. Profile actual state/capture memory, decide global-context fields and then compare procedural/neutral/learned alternatives. Do not pass this fixture through the old cubic two-facade adapter or label old thresholds as validated for blocks. Keep one bounded verification milestone at a time.