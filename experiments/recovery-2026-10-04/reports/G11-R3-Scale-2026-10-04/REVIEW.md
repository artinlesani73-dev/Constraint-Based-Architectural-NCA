# Larger-domain CPU probe — 2026-10-04

Original R3 successfully ran at40³ on one expanded physical site with unchanged weights, evaluator, planner and admission policy. This is feasibility evidence, not broad validation or a studio release. The approved visual skins are retained unchanged.

| Grid | Physical edge | Per-run checks pass | Rollout128 | Measured stages total | Peak working set |
|---|---|---|---|---|---|
| 32³ | 25.6m | True | 3.94s | 4.41s | 255MiB |
| 40³ | 32.0m | True | 8.67s | 9.41s | 312MiB |

Measured stages are context construction, route, witness/certificate and128-step rollout. Their sum excludes interpreter/model loading, saved-output serialization and final scoring. Peak working set is Windows process-wide peak resident memory since startup, including imports, model and trajectory capture; not incremental model memory, GPU VRAM or training memory. Both separate workers used CPU float32, two threads, same seed2102 and24% request. No warm-up or repeated timing trials; these values are observations, not guaranteed latency. Each worker had a180-second cap and completed normally.

40³ has1.953 times the cells of32³. Voxel size stayed0.8m: physical domain edge grew25.6m to32m. Building X/Y coordinates/extents and connection Y scaled1.25 with integer rounding; connection X was reattached to its facade. Building heights, entrance extent, street band, physical opportunity padding and2.4m cube thickness stayed fixed. Therefore this is not finer resolution, a uniform3D enlargement or the same physical problem. The allowed-domain fraction and volume cap recompute for the changed scene. The same RNG seed on different tensor shapes does not mean identical per-cell firing.

Both grids passed all nine families at64 and128, absolute volume error<=4pp and late growth<=5%; both had zero late growth. Finite trajectory/state fields, unique voxel births, provenance and state/field agreement verified.32³ final fields exactly match the archived parent at both horizons. Pre-run frozen input/source hashes unchanged. Both final views inspected at the same drawing scale; renderer boundary handling was adapted to the actual grid dimensions (original renderer assumed32).

Decision:40³ is a viable next experimental studio scale for this case; do not extrapolate to64³ costs or claim larger-grid generalization. Next prepare a versioned40³ studio path with dimension-aware rendering/validation and clear separation of larger-domain versus finer-resolution modes, then one focused integration check. Keep32³ as the reference. Increasing resolution at constant physical size would require explicit thickness/interface/padding semantics and is not authorized by this result alone.

All scenes/configs, contexts, certificates, route/witness fields, weights/source, trajectories, traces, runtime versions, metrics, controller limits and logs are preserved. No paid training or GPU run. Original studios including skins8018 and MG7 unchanged. Repository synchronization/off-device backup pending; no Drive, push or publication. Same-disk archive is not off-device backup.
