# Phase 2: growing architectural volume

**Artin Lesani Abdi · October 2026**

A research companion to the Phase 2 Substack article. The experiment studies gross building mass grown by repeated local learned updates, combined with explicit geometric admission and planning. It does not generate finished rooms or establish structural safety.

- [Read the article and model tables](substack-phase2.md)
- [Download the technical report (PDF)](NCA-Technical-Report.pdf)
- [Browse the saved-geometry showcase](showcase/index.html) — download this directory and open index.html locally; GitHub's file view does not execute the viewer.
- [Training figures, SVG exports and CSV data](figures-v4/)
- [Equation source](equations.tex)
- [Publication file checksums](release-manifest.json)

## What the comparison shows

| Method | Original ID | Contribution study, accepted feasible requests | Transfer study, accepted feasible requests |
| --- | --- | --- | --- |
| Vega-10 | G10 | 4/6 | 2/6 |
| Lyra-03 | G11-R3 / R3 | 6/6 | 6/6 |
| Neutral procedural control | neutral | 6/6 | 6/6 |

These are separate small studies, not pooled results. Each study also includes two blocked requests per method: Vega-10 produces rejected geometry; Lyra-03 and Neutral refuse. Refusals are not accepted geometry. The neutral control also passes the feasible gates, so learned necessity remains unproven. Seed variation measures geometric difference, not architectural quality.

## Names and evidence

Vega-01–10 are publication aliases for G1–G10. Lyra-01–03 refer specifically to G11-R1–R3, not unrelated older recovery runs. Lyra-03 reuses frozen Vega-10 weights; it has no independent training-loss curve. The report and frozen showcase retain original technical IDs for traceability. The original Phase 1 predates Vega-01. See the [complete alias map](publication-naming-map.json).

The article separates original training history from the later matched-start Vega-08/09/10 comparison. Different total objectives must not be ranked as interchangeable losses. All curves are extracted from retained logs; no simulated curves are used.

Evidence: [contribution summary](../experiments/reports/R3-contribution-complete-2026-10-08.json), [transfer summary](../experiments/reports/R3-transfer-results-2026-10-08.json), and [figure provenance](figures-v4/figure-provenance.json). The showcase is saved experimental geometry, not a live generation service. Neighborhood-scale generation remains future work.

## Publication status

Prepared locally for publication. No GitHub push, Pages deployment or Substack publication is performed by this preparation. The Substack draft URL is private and is deliberately omitted here. Add the final public article link after publication. The private next-phase planning report, checkpoints and runtime histories are excluded from this companion package.
