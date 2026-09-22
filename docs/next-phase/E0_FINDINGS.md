# E0 findings and the next engineering decision

Date: 2026-09-23. Recorded run: `20260922T230120Z_76f3b4677e8f`.
[Full tables and method](reports/20260922T230120Z_76f3b4677e8f-E0.md).

The next priority is a consistent, usable target and corrected training
objectives. Increasing voxel count or network size would not resolve the
specific target conflicts measured here. Keep the NCA concept under evaluation;
this is evidence for repairing and testing the pipeline before deciding whether
to retain, specialize or replace the learned model.

## What is now measured

270 fixed-weight forward cases completed on the local CPU; no paid compute,
optimizer updates, Drive access or deployment. All case records, full states,
input/corridor fields and source/config hashes are retained. The run archive
passed integrity verification. 82 regression tests passed before the experiment;
the failed notebook-oracle setup attempt and its linked retries are retained.

- **Settings materially change what is evaluated.** On the 12 frozen historical
  easy scenes, training-style forward dynamics connected all entrances in 10/12
  scenes at each of three seeds; serving also achieved 10/12 at each seed. The
  historical evaluation profile achieved 0/12 and averaged 27.7 material voxels,
  compared with 1,076.2 for training and 802.2 for serving. These are results on
  our frozen set, not a recalculation of the original 50-scene published scores.
- **The designed cases expose a real generalization problem.** None of the main
  profiles connected all entrances on any of the six reference scenes. One is an
  intentionally impossible negative control. The other five all have a route in
  the permitted voxel region, so their failures are not explained by that control.
- **The target itself needs repair.** Both ground-only reference targets lose
  connectivity after applying legality, although a permitted route exists.
  Their raw targets contain 452/660 and 180/388 forbidden voxels respectively.
  All five nonempty reference targets and 8/12 legacy targets include forbidden
  voxels. Clipping a target to legality does not automatically restore a route.
- **A limited procedural comparison already matters.** The legal part of the
  corridor target connects all entrances in all 12 legacy scenes, while the NCA
  connects only 10 at the main settings. This is one geometric criterion, not a
  full comparison of architecture, thickness, material use or compute cost. E2
  must assess those criteria before claiming the NCA adds value.

## What the single-seed ablations suggest

These pairs all use seed 0 and the same scenes. They are preliminary, not
estimates of robust effects across random seeds.

- Removing serving noise loses connectivity on 10/12 legacy scenes. The
  deployed small-seed configuration depends on its noise injection, even though
  the original training forward loop had no such noise. Noise should not simply
  be switched off without reconciling the rest of the configuration.
- Removing the serving step mask adds about 16,099 material voxels per legacy
  scene and 13,259 per reference scene on average, without gaining endpoint
  connectivity. More filled volume is not progress.
- Raising serving seed scale from 0.005 to 0.15 gains connectivity on one legacy
  scene (11/12 instead of 10/12) but none of the reference scenes. This is a
  candidate for further checking, not an approved replacement default.
- Changing training firing rate from 0.65 to 1 produces no connectivity gain.
  Increasing update activity alone does not solve the recorded failures.

## Next work, in order

1. Implement a versioned bounded vertical-envelope operator with an immutable
   input, preserving the old operator for replay. Follow CORRIDOR_FIX_PLAN.md.
2. Address target legality and routing explicitly. Reconcile street protection,
   ground entrances, path neighborhood and endpoint-based height clipping.
   Compare legal routing with merely clipping targets; do not hide disconnected
   targets behind improved legality scores. Keep all existing constraint families.
3. Repair objective definitions, tensor shapes and gradient paths, then check the
   corrected NCA against scaffold-only, procedural and direct-optimization controls.
4. Prepare the short Colab recovery/pilot notebook only when those checks pass.
   Confirm the specific compute cap and checkpoint backup procedure at that time.
5. Use the stable scene/result records to build the improved design workspace;
   a better renderer must expose these diagnostics rather than imply trained
   quality that has not been established.

All connectivity here is voxel adjacency, not walking clearance. Support is
geometric, not mechanical. The five feasible reference cases above are feasible
only in the limited sense of a route through the permitted voxel region. One
checkpoint, two development sets and these seeds do not establish broad
architectural generalization. Nothing has been trained or published in E0.

The historical `access_reach` score and the new through-material endpoint metric
answer different questions. Their numerical values must not be compared as if
they were the same measure.
