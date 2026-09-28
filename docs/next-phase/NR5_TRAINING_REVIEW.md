# Training stability review and NR5 proposal

The code establishes a mismatch, not a causal diagnosis. NR4 supervises a fresh
state after16 local updates; D080 evaluates32. Gradients therefore do not train
behavior at steps17-32. Both intact and damaged examples start from occupancy
logits +/-2 with zero hidden state. There is no state pool, persisted trajectory,
explicit stopping gate, contraction condition or update-magnitude penalty.
The zero-initialized final layer initially gives zero updates; training can change
that. Half-rate stochastic firing remains active on every subsequent update.
Gradient clipping limits parameter-update gradients, not inference drift.
Legality projection limits where output is occupied, not excess inside the domain.

Intact examples already constitute27/81 TRAIN rows. NR4 weights them more heavily,
but does not guarantee their preservation at16 or32 steps. Its reduced excess and
remaining736 intact false additions do not identify which mechanism is responsible.
No new horizon sweep or heldout model evaluation was performed in this review.

NR5 changes only the fixed training rollout from16 to32 steps; the diagnostic
TRAIN boundary also uses32 and remains separate from quality scoring. Same NR4
loss, fresh seed1201,256 Adam updates,32grid,rows,sampling order,9families and model.
It directly supervises the reviewed horizon with fewer simultaneous assumptions
than a new gate, loss penalty, pooled-state scheme or architecture. Those alternatives
are deferred, not disproven. More firing draws and backpropagation depth change
stochastic trajectories and compute; this is not a matched-FLOP comparison.

A successful result cannot prove stability beyond32, attractor convergence,
independent generalization or mechanical safety. Conditions are frozen in
NR5-horizon.json before any GPU result. Do not use TEST or select a best checkpoint.
If it fails, make one explicit model-path decision rather than automatically
continuing a sequence of small penalty or horizon trials.
