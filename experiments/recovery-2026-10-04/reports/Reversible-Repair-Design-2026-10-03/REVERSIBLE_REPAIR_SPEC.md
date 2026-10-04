# Reversible repair architecture proposal

This proposal evaluates whether allowing a model to revise its own additions can
improve repair while preserving the supplied volume. It is a research hypothesis,
not a demonstrated remedy. A reference binary transition has passed local probes;
there is no trainable implementation, recovery verification or GPU package yet.

## What the experiments establish

| Development metric | CGR1 | CGR2 | CGR3 |
|---|---:|---:|---:|
| Cases passing all nine checks |24/27|24/27|24/27|
| Damaged median overlap |.97504|.97096|.97214|
| Correctly recovered cells |1959|1828|1949|
| Excess cells on damaged inputs |245|165|268|
| Excess cells on intact inputs |117|83|139|

CGR2 reduces both unwanted growth and repair. CGR3 fixes one access failure and
one thickness failure without fixing all checks in any of the three failing
cases. All three miss the frozen acceptance criteria. CGR3 also uses a different
CUDA runtime; these comparisons cannot perfectly isolate the curriculum effect.

TRAIN trajectory evidence shows repeated rejection of reachable missing cells.
CGR2 rejected1151of1389remaining missing cells on at least one actual firing
opportunity. Reversibility alone does not force these cells to grow. Wrong additions
cannot geometrically obstruct growth in this monotonic rule, though they change
network inputs. The evidence does not identify irreversibility as the main cause.

A reversible rule addresses a narrower limitation: CGR1 cannot retract any mistaken
addition. It may allow exploration followed by correction, but it may also delete
correct repairs or oscillate. Keep CGR1 as the reference and MG7 live.

## Exact proposed transition

Let O be the immutable original occupancy, M the current accepted occupancy, L
the legal domain, F the stochastic firing mask, and q the predicted occupancy
probability. N6(M) is the six-face neighborhood of the PREVIOUS occupancy.

A = L AND NOT O AND F AND (M OR N6(M))

A includes fired generated cells and fired empty frontier cells. Outside A,
retain previous occupancy. Within A, accept q>0.5, otherwise clear the cell:

C = O OR (M AND NOT A) OR (A AND (q>0.5))

Then retain only cells of C reachable from ANY original occupied cell through
six-face connections. This flood-fill result is the new accepted occupancy.
The original must be nonempty and legal. A detached original component remains
an anchor; this does not guarantee that multiple original components merge.

Both direct deletion and connectivity cleanup may remove correctly reconstructed
cells. Cleanup can remove cells that did not fire. It is an explicit global step,
not a local NCA operation, and must be included in inference time and documented
as part of the model system. It is not a tenth constraint family; it enforces a
raw-connectivity property relevant to existing access/support checks. It does not
ensure thick bulk connections, architectural circulation or structural safety.

Keep the same local60-to64-to8network and seven hidden channels for this candidate.
Keep hidden updates and legal masking as CGR1; do not reset hidden states on
removed voxels in this first proposal. The immutable mask is an external update
rule, not an extra network input. q now represents occupancy at mutable cells,
not just willingness to birth. Use fresh weights and a new semantic identity;
never relabel a CGR1 checkpoint as a trained reversible model.

## Training objective and limitations

Start only from original TRAIN inputs; no CGR3 curriculum and no CGR2 bulk term.
At each step supervise q on A with the target occupancy label. Retain CGR1's
positive0.5 and negative1 weights (negative1.5 for intact examples), class means,
and empty-class differentiable zero. This necessarily expands supervision to
already-added cells, so the proposal changes update semantics AND the supervised
set. It is not a pure single-factor deletion ablation.

Define soft pre-cleanup occupancy S = M*(1-A) + A*q. O is retained because A
excludes O. Retain the0.25mean absolute3-cube local-volume discrepancy, selecting
windows that touch A. Average per-step losses over32steps. Hard decisions and
connectivity cleanup remain detached; no straight-through estimator or gradient
through flood fill is claimed. The soft proxy is not the projected field. This
remaining mismatch and lack of gradient through future discrete decisions must
be reported rather than described as solved credit assignment.

Target data is used only for supervision. Inference receives original occupancy
and the existing scene context. No target, damage mask or target-derived proposal
may enter inference. Preserve the nine constraint families and volume semantics;
this experiment concerns repair, not generation of new architectural volumes.

## Local evidence and required implementation gate

reference_transition.py is a NumPy binary-rule prototype only. Local probes verify
original preservation, deletion of a generated cell, pruning detached descendants,
legal birth restriction, use of the previous six-face frontier, no-fire identity
for an already anchored field, and preservation of disconnected originals.
The first bridge probe accidentally allowed births around the deleted bridge,
reconnecting the tip; its corrected fixture suppresses those alternate births.
The transition implementation did not change. Both outcomes are documented.

Before training, implement the learned rollout and separately verify loss-gradient
direction for removal/retention/birth, empty active masks, absence of target effects
on inference, and exact checkpoint recovery through a delete-and-rebirth sequence.
Save original,current,candidate,projected occupancy, direct removals,cleanup removals,
q and RNG state. Check mathematical parity to the reference transition. Measure
32-cubed TRAIN inference time including global cleanup; do not assume GPU efficiency.
A CPU flood fill copied every step may be too expensive. Any optimized implementation
must match the reference exactly; a bounded incomplete flood fill is not equivalent.
Do not package paid training if this gate exposes an unresolved correctness or
runtime problem. These checks cannot establish model quality.

## Fair comparison and decision

Historical CGR1 remains contextual evidence. For an eventual comparison, prepare
one paired job on the SAME verified runtime: fresh CGR1 control and fresh reversible
candidate, each seed1201,256updates,32steps,Adam.001,identical sorted TRAIN81 rows,
row order and per-step firing draws. This is512updates total, not a single256-update
run. Propose a shared600-second controlled cap only after local timings support it;
setup/export/idle extra. Each arm gets its own checkpoints and results. A partial
paired job is incomplete evidence, not permission to retry. Obtain explicit paid
compute approval after the implementation and package are concrete.

Evaluate final256 of both arms on the original27development inputs,CPUfloat32,
32steps,firing2101. No threshold,horizon or checkpoint search, no TEST. Score the
accepted projected field; also report pre-cleanup geometry so cleanup cannot hide
bad proposals. Measure wrong additions removed,correct additions removed,cells
pruned by connectivity,birth/death repetitions,runtime and memory. Do not introduce
new numerical acceptance thresholds after seeing outputs.

Retain the original frozen gates: all9intact IoU>=.99 and valid; damaged validity
>=17/18,medianIoU>=.9705768039313023,excess<=325,recovered>=1945,median absolute
requested-volume error<=19; zero original-input removals. Report all regressions
against the contemporaneous control. Passing gates is not automatic deployment
approval; one seed and repeatedly used development data remain limited evidence.
No claim that rule reversibility caused a benefit without a more targeted ablation.

## Scope and records

No paid job,live replacement,Drive operation or repository mutation occurred.
The agreed export remains the full ZIP; no small-review-ZIP workflow is proposed.
All prior findings are preserved. Repository documentation is still pending write
access; this local specification,prototype,probe results and archive support resume.
Next implement and validate the learned candidate locally, then decide whether a
paired GPU comparison is justified. Do not request compute approval yet.

Evidence sources are the local CGR1,CGR2,CGR3 final reviews and the TRAIN diagnosis
under C:/Users/artin/Documents/Codex/outputs. No new external research claims are
made in this proposal.
