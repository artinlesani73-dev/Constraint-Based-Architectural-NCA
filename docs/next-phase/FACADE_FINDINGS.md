# Facade repair and constructive baseline findings

Completed2026-09-23. The experimental facade allowance resolves both measured
radius-six facade/budget contradictions. A deterministic procedural baseline now
produces connected, numerically zero-loss material for all17 feasible scenes.
The sealed reference remains incompatible. This is a consistent research baseline,
not evidence of better NCA designs or validated architecture.

## Changes and evidence

`facade_endpoint_v1` excludes only permitted, face-adjacent cells inside named
facade entrance blocks from excessive-facade accounting. Eighteen frozen sidecars
record their exact cells and hashes. No guide, scaffold or model output determines
these patches. Ground entrances receive no exemption, patches are not dilated,
and diagonal neighbors are still charged. The15% cap and all-material denominator
remain, as do both material budgets and all eight other loss terms.

A1 run `20260923T084341Z_e37699e31f26`, source `2d543de`, completed in34.26s CPU:
864 paired target-arm records,144 control-arm records,144 bound-arm records and72
gradient-arm records. No optimizer updates. All432 target pairs reproduce T1
baselines exactly. Independent report checks reproduce allowance masks, facade
values, other-term equality, bounds and analytical quotient derivatives.

| Necessary-compatible feasible scenes | Original | Endpoint allowance |
|---|---:|---:|
| Radius3 / site budget | 3/17 | 3/17 |
| Radius3 / envelope budget | 11/17 | 15/17 |
| Radius6 / site budget | 12/17 | 12/17 |
| Radius6 / envelope budget | 15/17 | 17/17 |

For legacy007 the mandatory charged facade cells fall14 to6; minimum material
falls93.333 to40 voxel equivalents against the unchanged92.28 cap. For legacy008,
charged cells fall13 to5; minimum falls86.667 to33.333 against64.08. These are
explicit changes to what counts as excessive contact, not budget relaxation.

Every facade-blanket control remains penalized; new losses range0.84136-0.85
versus0.85 originally. Attachment-only receives zero facade penalty by definition
but can fail other objectives. Eight other terms, fields, masses and binary metrics
are unchanged. The new term still allows dilution by adding unrelated material;
it does not require attachment or certify engineered connections.

[Full A1 report](../../experiments/reports/A1-facade-comparison.md).

## Constructive witnesses: sufficient examples, not only necessary bounds

A1's existing static candidates provide zero-loss examples in11/17 feasible scenes.
W1 is a separately preregistered follow-up, not a retroactive A1 addition.
`budgeted_witness_v1` starts from the full procedural guide and adds only legal,
uncharged, face-adjacent cells within the existing radius-six envelope. It stops
at the existing mass/contact lower bound, never changes the12% cap, and refuses
new eroded cores. Ordered z,y,x tie-breaking is deterministic and shape-biased.

W1 run `20260923T084933Z_eb2603cd79f7`, source `2965502`, completed all18 scenes
in4.90s CPU.17 witnesses have all nine terms<=1e-7, independent binary connectivity
and zero illegal material. The sealed reference is explicitly incompatible.
Resulting feasible material volumes range11.776-41.984m3 at0.8m per voxel. Each
witness preserves the guide and adds0-24 cells. Existing zero-loss strands remain.

All ordered additions were independently replayed for adjacency/legal region and
field equality; all nine losses and independent binary metrics were recomputed.
This supplies a reproducible procedural comparator for training. It also exposes
that some added volume only satisfies a floor or dilutes a ratio; visual quality,
load capacity, minimum member width and walkability do not follow from zero loss.

[Full W1 report](../../experiments/reports/W1-witnesses.md).

## Verification and scope

Latest verification `20260923T084753Z_489a9c4be020`:132 tests pass,zero failures,
errors or skips; original checkpoint smoke exit0. The earlier facade-only suite
`20260923T084059Z_6280ed836d9f` passed129 tests. No experiment attempt failed.
Hashed annotation JSON now keeps LF bytes on Windows via .gitattributes; the
milestone backup restore check verifies both original scene sets and all18
annotation payload hashes. Private next-phase reports remain Git-ignored.

User's 'ok go on' accepts architectural material generation as the near-term
working direction. Keep usable pavilion/bridge as the longer-term goal and never
present material connectivity as walkability. Production defaults, original
notebook/checkpoint and historical formulas remain untouched; new paths are opt-in.

## Next implementation step

Use radius6/envelope with facade_endpoint_v1 as the explicit experimental baseline
for calibration and small E2 preparation. Retain the old facade arm as an ablation.
Now measure actual NCA parameter gradients across all17 feasible scenes, including
under/in/over-budget probes and pre-clamp coverage saturation. Audit the original
notebook's TV/density/cantilever regularizers before porting them. Fix coefficients
with a declared small sensitivity comparison; do not infer weights from loss values.

Then run matched no-update checkpoint, W1 procedural, direct-optimization and NCA
controls under a frozen evaluation contract. W1 proves objective consistency; it
also raises the standard for learned value. NCA must demonstrate useful adaptation,
recovery or held-out geometry behavior beyond copying a fixed route. Existing18
scenes are development data; freeze fresh holdouts before inspecting outputs.
Paid Colab still needs a concrete configuration and compute cap. No setup or cloud
access is needed for the next local calibration work. Preserve every attempt.
