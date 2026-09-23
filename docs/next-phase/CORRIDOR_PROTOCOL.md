# Corridor correction protocol C1_v1

Frozen before the comparison run on 2026-09-23. No training or deployment.

## Separate interventions

1. `legacy_v31`: original operator, unchanged replay oracle.
2. `corridor_bounded_v1`: only replace the ascending in-place vertical scan
   with a bounded, out-of-place depth maximum. Keep the original 26-neighbor
   distance propagation, 64-iteration cap, centroid extraction, MST/kNN edges,
   path slack, isotropic width dilation, endpoint-height clip and building mask.
   It deliberately retains target-legality conflicts for causal isolation.
3. `corridor_legal_v1`: new procedural target over the existing permitted field.
   Explicit entrance IDs replace extracted centroids. Exact six-neighbor BFS
   finds shortest paths between legal entrance regions with no iteration cap.
   A deterministic Kruskal minimum spanning forest selects edges by distance
   then IDs. Include legal endpoint regions to join paths incident on a region.
   Apply isotropic width dilation, bounded depth expansion, and the legal mask;
   discard thickened components unreachable from the routed centerline. Remove
   the endpoint-based height clip: a ground entrance must reach above the street
   band before a horizontal connection is legal. Config's corridor_z_margin is
   intentionally unused in this version. Infeasible cases retain partial forests
   and explicit failure-to-connect status. Ambiguous disconnected legal endpoint
   regions are rejected. Existing nine constraint families remain unchanged.

The third arm is a **routing package intervention**, not an attribution to one
individual line. Its shortest-path forest is a procedural control, not a learned
architectural planner, a minimum-volume Steiner tree, or a walking/structure test.
CPU graph traversal is detached and sequential per scene. Profile it before
large grids or online training; this implementation makes no GPU speed claim.

## Frozen comparisons

- Both frozen sets: 6 reference + 12 legacy scenes; original checkpoint/config.
- Target audit: all three arms, width 1 and vertical envelope 1. Record raw
  targets, seeds, legal region, centerlines where available, entrance contact,
  material/illegal counts, z ranges, added/removed voxels against legacy and
  the bounded arm, legal-target connectivity and router diagnostics. Verify
  bounded radius-zero parity on all 18 scenes, recording outcomes.
- Matched forward comparison: all 18 scenes x three arms x historical-training
  and historical-serving profiles = 108 cases. Predetermined RNG seed 0 only,
  50 steps, schedule position 60, same model weights/config, CPU two threads,
  deterministic algorithms. Reinitialize Python/NumPy/torch RNG before each
  case. No historical-evaluation arm: that profile does not consume a corridor.
- Use `binary_v1`, material >0.5 and six-neighbor connectivity; retain threshold
  material counts at 0.3 and 0.7, legality, ground, support and thickness proxies.
- New result directories and immutable registered arrays/records per case.
  Record source snapshot, checkpoint/manifest hashes, profiles, versions,
  timings, failures and infeasible cases. Retry creates a new linked run.
- Compare the legacy arm to E0 seed-0 saved fields as a replay check where
  available. A mismatch is evidence to investigate, never overwrite E0.

This is a development diagnostic of one historical checkpoint with one rollout
seed, not retraining, a held-out study, or a claim of improved architecture.
Targets can improve while this checkpoint fails to use them. No settings are
promoted to production from this experiment alone. E2 still needs material/cost
and quality comparisons against procedural and direct-optimization controls.
