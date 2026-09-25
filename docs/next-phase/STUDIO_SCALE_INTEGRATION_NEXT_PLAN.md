# Next proposal: versioned Studio integration

2026-09-25. MG7 timing admission: PASS. This document is a
proposal for the next milestone, not an implemented Studio change.

1. Proceed to a bounded integration specification.
   Keep MG3/MS1 records and imports supported. Add explicit dispatch by supported
   generator/record version rather than replacing the existing import replay.
   Retain exact source bytes and evaluator/config identities in every new job.
2. Introduce the optimized generator first on existing 32-grid presets. Test
   old export/import/re-export, new exact replay, rejected altered geometry or
   version, mixed-version comparisons and interrupted-job recovery.
3. Admit only the six tested larger site/seed combinations per size at the tested
   24% request initially. Keep the two blocked cases per size as visible negative
   controls. Wider seed/request selection requires a new preregistered study;
   passing two seeds does not validate every request. Keep 0.8 m voxels and use
   scene-sized decoding, bounds checks and physical-unit labels throughout.
4. Reuse the existing separate worker process, job tree cancellation, parent
   watchdog and durable queue. Audit and test deadline enforcement separately
   from cooperative generator limits. Freeze allowed launch/generation/deadline
   behavior before acceptance tests. Exercise real cancel, server death, restart,
   queued work and retained failed/interrupted records on the integrated path.
5. Move potentially longer import replay through a bounded worker path before
   admitting large imports; the current import calls generation synchronously.
   Treat bundled source as data; replay only recognized local versions. Validate
   payload size, coordinate bounds, physical context and hashes before generation.
6. Make viewport, scale, request and generator identity clear. Preserve comparisons,
   failures and export access. Verify real browser jobs against saved study fields
   and measure end-to-end latency including queue, import/runtime startup,
   evaluation, persistence and rendering. Numerical generation speedup alone
   does not establish interactive quality.

Freeze a finite acceptance matrix and capture source/inputs before implementation
outcomes. Keep the old Studio route available until compatibility and UI checks
pass. Record every result and interruption, then produce a local commit and
verified archive. Nine families and D058 volume meaning remain unchanged.

No new NCA training is required for this integration. Once a useful, measurable
volume baseline is in the Studio, return to the separate research question of
learning generation: compare a versioned NCA against this procedural comparator
using corrected objectives, held-out sites and an approved Colab budget. Do not
equate a stronger procedural baseline with progress of the historical checkpoint.
No paid compute, Drive access, push or public hosting follows automatically.
