# MT1: binary massing target contract, before optimization

2026-09-24. Implements D058/D059 next work. Occupancy is building volume; interiors
and construction remain deferred. This is a provisional geometric contract and
counterexample audit, not a differentiable loss or architectural-quality score.

Freeze recipe MT1-targets.json before the run. Four declared development contexts:
aligned, wider gap, vertically/laterally offset interfaces, and a full blocking
partition. Twelve analytical fields per context. Two positive controls (compact
and articulated mass); ten negatives (empty, thin sheet, fragmentation, thin neck,
unsupported mass, context collision, ground intrusion, satellite, spill, full fill).
All fields remain unclipped. Expect both positives to pass in three open contexts
and all candidates to fail in the blocked context. Retain unexpected outcomes;
never tune thresholds silently to match the proposed control labels.

## Pilot definitions within the existing nine families

- access: all declared interfaces and all occupied cells share one legal in-domain
  six-connected component. Additionally all bulk-qualified cells and interfaces
  share one bulk component. No one-cell connector can establish substantial access.
  This is geometric mass connection, not interior circulation or walkability.
- coverage: bulk-qualified volume occupies at least8% of each of three fixed X
  thirds of the opportunity region. This is an explicit distribution proxy for
  these gap-facing contexts, not universal site orientation or all-cell coverage.
- facade: retain old non-allowlisted facade-contact/all-occupancy fraction <=15%,
  with the same26-neighbor facade shell and6-neighbor typed-interface exemptions.
  This retains its known dilution limitation; the upper volume cap bounds but
  does not solve dilution. Check numerical parity against old scores.
- ground: no occupancy in the existing protected ground mask; anchor exceptions
  remain explicit. A stricter binary acceptance check, not a hidden old-loss edit.
- legality: zero occupied cells outside historical permitted space.
- sparsity: ALL occupied volume / fixed region volume in[8%,40%]. No cropping of
  invalid cells in the numerator. These are declared pilot values, not budgets
  inferred from MA1's successful shape or established architectural requirements.
- spill: no occupancy outside the MA1 physical opportunity region. MT1 explicitly
  adopts this scene-derived region as its distribution boundary inside the existing
  spill family. It does not turn the region top into a new legality/height family.
- support: every occupied cell connects to the historical geometric support
  boundary. No load, force, cantilever safety or construction feasibility claim.
- thickness: at least90% of occupied cells belong to at least one fully occupied
  axis-aligned cube of side>=2.4m. Use ceil(side/voxel_size) with numerical tolerance
  and the union of all complete cubes (binary opening), including even widths.
  Full cubes retain boundary cells; single sheets and narrow necks do not qualify.
  This replaces the material-bulk penalty only in this new binary contract.

Report all nine pass/fail checks separately and their explicit conjunction, with
context necessary-feasibility flags. No weighted sum. Passing means meeting this
pilot contract, not being a good building. Minimum-scale geometry is axis dependent
and can be gamed by volumetric grids; three-bin coverage cannot ensure useful
architectural articulation. The90% allowance can permit small thin appendages.

Opportunity region remains candidate-independent, physical interface bbox plus
padding(Z,Y,X)=(6.4,6.4,0)m intersected with historical legality/context exclusions.
Retain original nine-family numerical scores for every field. Context feasibility
tests connectivity of the region's cube-supported subset, a necessary condition
only, not proof that a budget-compatible candidate exists.

## Validation and preservation

Run the full regression suite and save exact source/config/scenes/masks/fields.
Preregister sensitivity: cube sizes1.6/2.4/3.2m crossed with max budgets25/40/55%;
other settings fixed. Save all432 evaluations, including shifts in control labels.
Check an aligned physical2x refinement for the cube operator separately; do not
claim general resolution invariance for diagonal/curved or non-aligned boundaries.
No automatic search, differentiable optimizer, NCA trial, larger-site training,
paid compute, Drive operation or remote push. A failure leaves its run and source
intact and triggers a documented review rather than automatic retuning.

Final review correction: context feasibility must examine every available component
touching the source interface. Choosing only its first cell can falsely reject a
context where a later component connects all interfaces. Added a counterexample
test, including disjoint paths that must not be merged. Preserve first audit
20260924T101444Z_3d8c5504cb9e and link a fresh run; thresholds/control labels unchanged.
