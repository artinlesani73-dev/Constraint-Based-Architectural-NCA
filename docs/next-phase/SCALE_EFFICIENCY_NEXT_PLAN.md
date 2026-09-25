# Next proposal: exact-output efficiency before larger live jobs

2026-09-25, after MG6. Read the actual SCALE_STUDY_FINDINGS first. Preserve all
MG5/MG6 successful, failed and partial fields, configs and measured timings.

MG5 currently rebuilds whole-grid cube counts after every positive addition.
Only cubes overlapping newly occupied cells can change their counts. A separate
implementation can initialize counts once and decrement new-cell/contact/third
counts at affected origins. This is a concrete hypothesis, not a measured speedup.

Keep the same full legal frontier, coverage-deficit priority, facade admission,
seeded radial tie-break, route construction, requested count and final fields.
Use a separate module/version; do not overwrite the frozen comparator. Audit
overlap updates, boundaries, zero-delta transit and integer counts against direct
voxel enumeration. Do not use stale frontier scores when third deficits change.

Before execution freeze a finite paired comparison, including the MG5 repaired
case and MG6 compact/obstructed/blocked cases, with exact fields/routes/decisions
as equivalence targets (timing excluded). Admit broader existing-case regression
only after those checks pass. Define paired trial order, warmup scope, latency and
RSS targets/caps before any claimed optimization gain. Keep setup, growth,
evaluation and saving visible; reducing growth alone may leave other bottlenecks.

MG6 offset/seed7 had a large wall/CPU discrepancy and a cooperative deadline
overrun. Record sample timestamps and CPU/wall intervals in the paired study,
and classify scheduling/host interruptions separately from intrinsic compute.
Preserve and report deadline overruns; do not claim an OS hard bound. Investigate
supervised cancellation before live promotion, without rewriting MG6 evidence.

For time-limited MG6 outputs, compare the exact recorded decision prefix at the
same number of steps. A faster implementation may continue beyond that prefix
under the same wall-time allowance. Do not require the same partial field at the
same elapsed time, or claim full-run equivalence from a matching prefix alone.
Any newly completed larger field still needs independent validity/request checks.

If exact equivalence fails, retain and diagnose that result before broader runs.
If runtime improvement is insufficient, report that instead of relabeling the
same quality score as a speedup. No automatic tolerance relaxation or reroll.

Then prepare a separately versioned live workflow for admitted site/size presets,
with honest progress/cancellation, durable results, bounded queue resources and
old-version replay/import. Faster valid alternatives and usable latency matter;
larger grids alone do not constitute a10x product improvement.

Finer resolution and fresh-site generalization remain separate. A learned NCA
pilot still needs an explicit measurable benefit, admitted objective, compute
allowance and recovery/off-device backup. No Drive, paid training, push, hosting,
new constraint families or automatic live promotion are part of this proposal.
