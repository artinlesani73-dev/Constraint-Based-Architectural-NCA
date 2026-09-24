# Next: a volumetric objective contract, before training

2026-09-24, after VA1. User target: volumetric forms and spatial voids, without
predefined functions. The rectangular audit probes are measurement controls,
not a new shape template or a target dataset for copying rooms or boxes.

## What should change first

Keep the existing material occupancy representation initially. VA1 demonstrates
that it can represent both solid and hollow fields. More channels or a new NCA
backbone are not yet justified by an inability to represent a void. Hidden-state
capacity, conditioning and training remain hypotheses to evaluate subsequently.

The immediate work is a versioned interpretation of the nine existing families
for volumetric growth. Separate three concepts: permissible material region,
amount of material, and the voids articulated by that material. A required occupied
route and its narrow dilation should not silently define all three.

## Bounded next implementation

1. Define a candidate-independent 3D opportunity region in world units from the
   scene. Compare it with the historical route-derived envelope on the SAME saved
   fields. Declare how it behaves at different voxel resolutions. Never expand
   a denominator in response to a candidate merely to turn failure into success.
2. Audit feasible material-budget ranges for that region using volumetric controls
   of several scales and apertures. Retain actual material count, physical volume,
   original denominator and original scores. VA1's40–50% ratios are observations
   for particular coarse probes, not a proposed new target percentage.
3. Separate material support/connectivity from free-space connectivity within the
   access family. Establish what access endpoints mean without inventing physical
   door openings inside the solid context boxes. Allow open spatial voids; sealed
   chambers cannot be the only rewarded outcome. No walkability or room program
   follows from free-space topology alone.
4. Replace the mandatory occupied centerline interpretation of coverage with an
   explicit volumetric distribution target or bounded descriptor task, while
   testing trivial cheats. Candidate targets require counterexamples: solid fill,
   empty field, isolated fragments, a sparse tall lattice, detached panels, sealed
   unreachable pockets, and a single planar sheet. VA1's axis-bracket measure is
   diagnostic only; do not adopt it uncritically as a differentiable loss.
5. Retain legality, ground, facade and support semantics unless the audit exposes
   a specific conflict. Keep thickness as a separately tested control of material
   bulk/member scale; it is not the same as overall volumetric depth. Keep sparsity
   and spill tied to an explicit, fixed physical domain. Version changes, report
   all nine families and avoid hiding failures in one weighted score.

Freeze the revised objective contract and acceptance examples only after these
comparisons. Then compare a procedural volumetric control and direct material
optimization before proposing a learned model. The next NCA trial must answer a
specific question about useful volumetric growth, variation or recovery, with
held-out scene families and a defined compute allowance. The earlier proposed
platform-repair task does not carry forward as the accepted research objective.

## Explicit representation alternative

If deriving spatial descriptors from one material field proves unreliable or
non-differentiable for the chosen task, compare explicit material/void channels as
a named ablation. Define their consistency inside the opportunity region; a
predicted void must actually be empty and meet the independent geometric checks.
Otherwise the model could improve a void-channel score without changing geometry.
Do not change representation, backbone, scene distribution and budget together.

No additional training, budget adjustment, production model switch, Drive action
or paid Colab run is authorized by this proposal alone. Local contract design and
saved-field analysis can proceed from the documented evidence.
