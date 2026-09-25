# MG7 frozen incremental-growth comparison

2026-09-25. D071. Frozen before real-scene MG7 generation. Preserve all MG5/MG6
sources, results and partial fields. The new module duplicates the unchanged MG5
routing/growth choice rule but caches initial cube counts and decrements only
origins overlapping each newly occupied voxel. Route, RNG draws, physical units,
coverage/facade priority, tie breaks and requested-volume stop are unchanged.
No new constraint, parameter tuning, resolution change or NCA training.

Six focused synthetic tests pass before outcomes: direct enumeration of caches
after disjoint additions, holes/contact/boundary cases, width1/2/3/6, complete
growth-report equivalence, zero transit, stalls/timeouts, request stop, forged
full/prefix rejection and timestamped RSS/CPU sampling. Full regression required.

Stage1 diagnostic: the six exact baseline members in MG7-incremental.json.
Require4 full matches plus2 exact recorded prefixes, all6 optimized outputs meet
the expected quality/request outcome (blocked remains failed). Completed fields,
routes and full generation reports must match, excluding version and wall time.
For two MG6 time-limited cases, require the same initial route, stable metadata,
selected-origin prefix, every saved decision and reconstructed partial field.
Continuing past that prefix is permitted; a completed field must independently
meet MT1 and the original requested volume. Prefix agreement is not full-reference
equivalence. Preserve failures; no rerolls or silent timeout expansion.

Stage2 only after Stage1 passes: all237 saved MG5/MG6 members (225+6+6), including
235 completed-reference comparisons and2 recorded prefixes. Require exact choices
and correct final validity/request outcome for every member:188 nonpartition
passes,49 blocked failures. Identical old complete fields remain unchanged. The
two newly completed64 fields, if produced, need independent full-frontier audits
beyond the saved prefixes as well as all nine binary checks. No fresh-site claim.

Stage3 only after Stage2 passes: paired timing on the three completed-reference
cases listed in timing_cases (32 repaired,48 obstructed,64 compact). One retained
warmup per method on the32 case, then four measured trials per case/method.
Trial order is reference/optimized, optimized/reference, reference/optimized,
optimized/reference; case order follows the frozen list in each trial.26 total
executions including2 warmups. No timed-out reference is compared as if it were a
completed workload. Save every output and every trial; no favorable selection.

Timing admission requires all equivalence/quality checks, and per case optimized
median generation WALL and CPU time <=75% of reference medians, optimized maximum
sampled process RSS <=125% of reference maximum, and all measured sample gaps
<=1s. No exclusions for slow or interrupted-looking trials; a gap fails the timing
gate and remains evidence, without attributing its cause. Ratios of medians use
four trials; this is a bounded local comparison, not a general speed guarantee.

Local CPU,2 Torch threads. Per-candidate settings match each saved reference:
15s at32,45s at48,120s at64. Study caps600/1200/1200s;2GiB observed RSS threshold.
Checks are cooperative, at generator steps/study boundaries, not OS hard limits.
Retain actual elapsed overruns. Do not tune caps based on outcomes. New executions
and interruptions use immutable run IDs with admission/parent links.

Scoped one-call growth wrappers record wall and process CPU for both methods;
independent verification uses unwrapped optimized code. Record generation,
growth, independent evaluation and saving separately. Native RSS sampler retains
every sample's monotonic wall time, process CPU and resident bytes, requested10ms
interval, maximum sample gap and lifetime peak. Sampling spans generation and
evaluation and can miss short peaks. Context/input loading is outside per-case
timing but within study time. Same process with balanced order; retained allocator
effects remain a limitation. This is not a live UI latency or10x product claim.

Save exact inputs, old/new arrays, traces, scores, config/source snapshots and
resource timelines with hashes. Independently rescore all fields, check full/
prefix comparisons, audit new64 frontier choices and recompute performance gates.
Record source parity and all failures. Update findings/decisions/resume; local
commit and verified raw-source/Git-bundle archive. Private reports remain ignored;
no Drive, paid compute, public hosting, push or automatic live promotion.
