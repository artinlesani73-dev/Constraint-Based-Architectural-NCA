# G9: access-priority objective proposal

## Decision

Prepare a **ranking-only loss supplement** to G8. Retain the original teacher
membership, volume and band losses, architecture, all45 TRAIN examples, start
schedule, stochastic firing, budget, quota, and detached cube admission.
This is an implemented and locally checked loss prototype, not an integrated
trainer, GPU-ready package or trained candidate. G8 remains the evaluated
experimental reference; MG7 remains live.

## Evidence and interpretation

G8 passed32/33 old and8/12 new cases at both64/128 steps. All45 passed volume and
stability gates. The five access failures exhausted their mass allowance by128.
G6's earlier objective audit established that static teacher membership rewards
both advancing and other eventual teacher cubes. G5's extra destination inputs
did not solve the earlier system's failures. Together these support testing
relative growth priority, without claiming it is the sole cause or guaranteed fix.

The new graph/label calculations used only the existing45 TRAIN examples.
No new held-out labels, checkpoint selection, model inference or optimizer
updates occurred. Previously evaluated G8 scenes are now regression evidence;
they must never be called fresh in a later report.

## Rejected design: postpone all other growth

An initial temporal-BCE prototype suppresses non-advancing cubes and suspends
volume losses until both interfaces are touched. Its deterministic all-fire
teacher oracle connects all45 TRAIN cases in14–22 steps, with at least338 cells
of global allowance remaining. However, with the unchanged per-step quota,
37/45 of these oracle trajectories cannot reach requested B by64 even if
every remaining step uses its full quota (maximum shortfall206 cells).
This is an upper-bound calculation for those particular oracle trajectories,
not proof that every possible gated policy fails or that every shortfall breaks
the frozen tolerance. Firing randomness can add delays. Reject hard postponement
for the next controlled experiment; preserve its code and diagnostic fields.

## Exact proposed training objective

For each TRAIN target, form its full3x3x3 cube-origin graph with six-neighbour
origin moves. Multi-source BFS starts at teacher cubes touching the east
interface. This teacher-derived distance is supervision only: it never enters
model inputs, admission scores, inference or postprocessing.

From a seed, advancing positives are eligible teacher origins with the minimum
finite distance. From a later state, they are eligible teacher origins with
distance strictly below the best finite distance among existing full origins.
Stop this auxiliary supervision once the connected field touches both west
and east interfaces. The current generator guarantees a connected field; this
contact test alone would not establish access for a disconnected generator.
If no teacher-route progress is available, record the fallback and use G8 loss.
This does not solve off-teacher recovery; any future run must disclose fallback
counts, including late capacity-saturated states, separately.

Intersect both groups with the actual firing mask. Let A be advancing teacher
origins and O the other eligible fired teacher origins. Add:

    Lrank = log(1 + mean over a in A,b in O of exp(1 + z[b] - z[a]))
    Lnew = LG8 + 1.0 * Lrank

Use log-sum-exp and softplus to compute this stably without constructing the
pair matrix. Margin1 and weight1 are fixed design choices, not tuned results.
With either group empty, after connection, or on no-route fallback, Lrank is
exactly zero. Every original positive remains a positive in the G8 BCE.
No volume term is disabled. This changes relative score gradients without
forbidding simultaneous mass growth. Ranking affects the gradient, not the
hard sort/admission algorithm. It does not guarantee a connection or timely fill.

## Local checks and what they establish

All427 saved G8 training starts reconstructed with matching original hashes:
214 seed-access,153 advance-access,60 already connected; zero no-route starts.
The hard-gating diagnostic had678 gradient-direction checks and45 exact
post-connection baseline-loss checks. The selected ranking prototype checked
769 stored oracle states; 706 have both comparison groups.
Explicit pairwise calculations match the efficient formula; its gradients
raise advancing and lower other teacher logits. No-group/no-firing behavior,
extreme-logit finiteness, and45 exact post-connection baseline delegations pass.
These are mathematical and supervision-feasibility checks, not learned quality.
The oracle ignores firing randomness and is not a deployment fallback.

## Next implementation, then one bounded Colab comparison

1. Integrate this versioned loss into a separate G9 training session. Keep
   inference byte-equivalent for fixed weights, seed and context. Cache TRAIN
   graphs deterministically; log auxiliary loss, phase counts, empty-group and
   no-route events without changing random-number consumption.
2. One consolidated local check: unchanged initial weights/start/firing sequence,
   inference parity, finite backward and exact checkpoint recovery at a seed
   and partial-start update. No new local model-quality sweep.
3. Freeze one fresh seed1201 paired427-update,64-step T4 proposal, at most600
   controlled seconds on the admitted runtime, with no automatic retry. This
   is a proposed allowance; no paid run is authorized or launched by this file.
   Preserve the full evidence and final427 checkpoint; no best-checkpoint search.
4. Freeze unchanged gates and all45 existing cases as regression before training.
   Freeze a genuinely new reserved split before its first inference. Compare G8
   and G9 on that same fresh split once, with identical firing and64/128 horizons,
   to avoid comparing success rates from different cohorts. Exclude held-out
   targets/route labels from both packages. A fresh split is still synthetic,
   and one seed cannot establish broad reliability.

Acceptance is the same nine-family conjunction plus volume/stability gates.
Inspect raw results, geometry and regressions even if aggregate scores improve.
Do not weaken gates or add a hidden procedural bridge. If access/volume still
trade off, diagnose this single comparison before considering removals or other
architecture changes. Greater grids and production deployment remain later work.

## Preservation and resume

All changes are local under this folder; checkout synchronization remains pending.
The first audit failed because its diagnostic passed a Boolean target to the
baseline convolution loss. That attempt is retained separately; the corrected
audit is v2 and uses float targets. This was an audit implementation error,
not a model/training failure. No historical evidence was overwritten.
Read RESUME.json here next. The verified archive is same-disk preservation,
not off-device backup. No Drive operation, paid compute, publication or live
replacement occurred. The original next-phase report remains untouched.
