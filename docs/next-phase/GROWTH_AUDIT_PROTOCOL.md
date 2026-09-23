# H1 frozen growth and gradient diagnostic

Frozen2026-09-23 before execution. H1-growth.json fixes the matrix and limits.
User approved continuing local work. No optimizer, architecture change, extra
constraint family, larger grid, paid compute, Drive access or production switch.

## Question and controls

Does any predeclared sampled duration retain connectivity while meeting the
unchanged3%-12% continuous material/envelope budget across firing seeds? Does
access pressure conflict with the material penalty at the actual model parameters?

Use original Model C on ground-pair/minimal-smoke (two model/scene pairs), all four
F1 final models and all four F2 final models:10 sources. F2 uses the completed
linked child143002Z_55aeaac95580; preserve its interrupted parent's timing caveat.
Loading frozen weights for diagnostics is not optimizer checkpoint recovery.
Validate source/config/input hashes, checkpoint metadata and update counters.

Full growth matrix:10 sources x3 firing seeds(0,1,2) x6 horizons(16,24,32,40,50,64)
=180 fields. Each horizon starts the same weak0.15 scaffold and explicit seeded
generator under unchanged hard_preclamp; horizon pairs share their firing prefix.
Retain every field, old/new access/BFS, other8 families,3 regularizers, both recipe
totals, mass, saturation and change from the previous sampled horizon. Sampling
does not certify intermediate steps, equilibrium, indefinite stability or new sites.
Seed2 at16/50 must match all20 historical raw/material fields exactly. Any mismatch
fails the run. Do not select a favorable stopping horizon after looking at data.

Compute8 new actual parameter-gradient cases:4 F2 final models x16/50, firing2.
Save complete vectors and last-raw derivatives for access_v1, access_v2, coverage,
sparsity, total_v1 and total_v2; retain norms, cosines, scalar values and layouts.
Require exact forward match to H1 saved fields and unchanged model weights.
Reuse12 A2 original/F1 gradient cases only with verified artifacts, unchanged
formula code hashes and exact source checkpoint/forward matches. Recompute vector
statistics, not backpropagation, on reused cases. Distinguish raw gradients from
parameter gradients and gradient directions from Adam updates. No gradient probe
is an optimizer step. D1/W1 remain already-established feasibility controls, not
new learned models or architectural certificates.

## Timing-only admission

Pilot: F2 mass_3 on both scenes, firing2, all6 horizons (12 growth fields), plus
ground-pair16/50 gradient cases (2 new gradients). No pilot-quality adjustment.
Cap180s/growth worker,120s/gradient worker,600s pilot total. Before full study,
require estimate1.5*(30*max(single-seed growth worker elapsed)+8*max(gradient worker
elapsed)) <=1500s. Startup/scoring/evidence included; repeating startup30 rather
than10 times is conservative. Full study caps180s/growth source (all3 seeds),
120s/gradient case,1500s total. Check elapsed limits after process return even
when OS wait reports success. No silently increased limits after a failed gate.

Full/source/config must match the pilot exactly. A failed/interrupted attempt is
preserved, gets a new linked retry, and must not be mistaken for completed work.
There is no automatic partial-matrix resume coordinator. Inspect result/logs/
processes before retry; reuse completed evidence only through explicitly verified
imports, never overwrites. Restore exact source snapshots for historical recovery.

## Verification and decision

Rescore every growth field with shared formulas and independent component BFS;
verify all150 sampled transitions,20 anchors,180-case matrix,8 new/12 reused
gradient cases,120 parameter and120 raw vectors,720 cosines. Verify source ZIP,
input/checkpoint hashes, all worker caps and frozen-weight checks. No fresh full
rollout rerun beyond the historical anchors and gradient forward duplicates.

Report every source/horizon/seed, joint outcomes and any lost connections. Above
the upper budget, negative access/sparsity gradient cosine indicates a local
tradeoff; it does not establish causal learning behavior. Choose ONE next learning
change only from this evidence (see GROWTH_STABILITY_PLAN.md). Keep alternative
explanations and all failures; no promotion from two development scenes.
