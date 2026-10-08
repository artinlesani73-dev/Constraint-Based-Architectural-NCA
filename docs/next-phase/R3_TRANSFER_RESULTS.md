# Prospective R3 transfer comparison — 2026-10-08

## Result

Completed all24 frozen method/request slots in387.906 seconds:20 generated
outputs and4 explicit no-certified-output records. No timeouts, allocation failures,
parameter changes or retries. This closes the planned bounded transfer comparison.

| Method | Accepted on3 certified sites,2 seeds each | Sealed site,2 seeds | Mean seed Jaccard distance on certified sites |
| --- | --- | --- | --- |
| Raw G10 | 2/6 | 2 rejected outputs | .29777 |
| R3 learned + planner | 6/6 | 2 no certified outputs | .25442 |
| Neutral scores + planner | 6/6 | 2 no certified outputs | .03677 |

G10 fails access at both horizons for both seeds on narrow-offset32 and
raised-obstacle40. It passes both flanking-block32 requests. Its sealed-site
outputs fail acceptance; those failures are retained. R3 and neutral correctly
find no legal thick route through the sealed site. Refusals are not passing
outputs and are reported separately from the six feasible requests.

## Fixed protocol and evidence limits

Four new authored layouts were schema validated and frozen before any rollout:
narrow-offset32, flanking-blocks32, raised-obstacle40 and sealed-gap40.
Exact geometry differs from the preceding contribution fixtures. This is a
prospective transfer check, not an exhaustive training-history exclusion audit,
a random site sample or proof of broad architectural generalization. All four
layouts are now exposed. Two firing seeds are correlated repeats per site.

Frozen G10 checkpoint0427 SHA256:
8e4ac1f13ff57b2c644cd2bd6fd27c40b25eca7062ec729f002969171a122bb2.
Seeds2201/2202,24% request,.8m voxels,original route,128steps,CPU float32,2threads.
Acceptance: all nine families at64/128,absolute volume error<=.04,late growth<=.05.
No new constraint family, learning, objective or planner change.

Neutral retains the R3 planner and guards but substitutes constant positive
occupancy logit and zero hidden updates. It removes both learned gating and
ranking; it is not MG7 or a ranking-only ablation. Fresh-process workers reuse
the previously parity-verified capture-only optimization. Scientific source files
and model hashes are unchanged; input copies, effective worker/capture source,
per-slot environment, certificates, traces, full saved trajectories and output
manifests are preserved. Analysis verifies input/worker hashes, finite arrays,
non-overlapping births,64/128 reconstruction,monotonic occupancy and acceptance
logic. Every slot has explicit provenance. Both comparison plates were inspected.

## Form and contribution

R3 again changes form and gives greater seed variation than this neutral baseline.
The neutral method meets every geometric gate without learned scores, so learning
is still not demonstrated necessary for gate satisfaction on this sample. More
voxel variation does not establish better architecture; raw G10 has even higher
variation while failing more requests.

The plots show more horizontally spread neutral forms. R3 Z spans are12/12,11/9
and15/18cells across the three sites; neutral spans9/9,6/6 and13/15. The taller
obstacle can itself make neutral forms vertically extensive. Neither bounding
height nor visual irregularity is a volumetric-quality score. Whole-volume
semantics remain; these are not construction, room-layout or structural results.

## Measured local cost

| R3 grid | Rollout seconds,2 seeds per site | Peak worker working set |
| --- | --- | --- |
| 32: narrow offset | 5.84 /6.34 | 257.7 /258.2MiB |
| 32: flanking blocks | 5.89 /5.78 | 260.7 /266.2MiB |
| 40: raised obstacle | 12.02 /11.23 | 307.4 /307.3MiB |

Windows working-set peaks cover the whole fresh worker through artifact writing,
including setup and capture, not inference alone. Peak process commit is also
recorded per slot (R3 roughly604–691MiB); working set is not total committed memory.
Run wall time excludes setup/planning/scoring; complete worker wall time is separately
recorded in controller.json. Values are single observations, not latency guarantees
or GPU forecasts. Neutral rollout range1.93–5.43s reflects avoiding neural inference.
Cross-method pooled timing summaries have different site coverage because refusals
have no rollout; use paired per-case observations for comparisons.

## Decision and next work

Keep R3 as the experimental learned hybrid, neutral as a research baseline and
MG7 as production. The bounded contribution/transfer validation round is complete;
no additional generalization test chain or paid retraining is proposed now.
The limitation statements above remain part of the release evidence.

Next complete the studio workflow and export experience in a separately versioned
app: inspect current save/reopen/export behavior, fill demonstrated gaps, and make
exported geometry carry its exact scene, units, seed, model identity and measured
status. Preserve Classic, Graphite and Porcelain. Use one focused end-to-end check
of generation -> save -> reopen -> export, rather than another scientific sweep.
Then prepare reproducible local release packaging and final handover. Publishing,
production replacement and Drive operations remain separate decisions.

## Files and restart

- Runner: scripts/r3_transfer_study.py; freeze/run refuse overwriting the attempt.
- Analysis: scripts/analyze_r3_transfer.py; rendering: scripts/render_r3_transfer.py.
- Packaging: scripts/archive_r3_transfer.py; refuses existing archive directory.
- Evidence: .local-artifacts/r3-transfer-v1-2026-10-08; controller completed.
- Protocol/results/archive receipts: experiments/reports/R3-transfer-*.json.
- Review and ZIP: C:/Users/artin/Documents/Codex/outputs/R3-Transfer-2026-10-08.

No running study,paid compute,Drive access,push or studio change from this milestone.
All historical attempts remain. Local archive is same-disk,not off-device backup.
