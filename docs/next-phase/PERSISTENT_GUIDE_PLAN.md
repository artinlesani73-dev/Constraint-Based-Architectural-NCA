# Proposed next experiment: persistent route conditioning

D051 proposal following verified F4: 62/72 connections versus F2's 59/72,
zero joint connectivity/material-budget successes in either arm. This is an
unexecuted architecture proposal, not evidence that missing context caused the
observed failures.

The original next-phase report recommends exposing agreed spatial context during
every update. The current research rollout adds 0.15 times corridor_legal_v1 to
the initial material state. Its update network subsequently sees only the eight
state channels; the scaffold has no separate persistent input. A3 also measured
dead access gradients after earlier clamps and non-firing steps. These motivate
a bounded conditioning test, without claiming they prove the architecture is
incapable or that conditioning will fix the material tradeoff.

## One explicit architecture change

Keep eight total state channels, four evolving channels, all clamps, firing behavior,
update scale, original Model C weights and the existing nine families/three
regularizers. Keep the initial scaffold application. Expose exactly the same
`item['scaffold']` (corridor_legal_v1), unchanged, at each update. Do not replace it
with legal_centerline or add a new route/envelope definition in this comparison.

Proposed implementation: cache identity and three existing Sobel perceptions of
the static one-channel scaffold. A zero-initialized, bias-free 1x1x1 projection
maps its four features into the first 96-unit preactivation. Add this to the
original first-layer output before its ReLU. The original 32-feature state
projection and all subsequent layers remain intact. This adds 384 trainable
weights, with no additional recurrent channels or constraint family. It avoids
reshaping the original state/input weights just to supply persistent context.

The zero projection must initially preserve the original output. Verify exact
CPU forward fields, original parameter derivatives and firing/global RNG where
the implementation permits exact preservation; investigate any mismatch before
training, rather than assuming mathematical equivalence guarantees bitwise
equivalence. Construct the zero branch without consuming additional RNG. Give
the architecture, checkpoint migration and conditioning field an explicit version.
Historical checkpoints remain immutable and cannot silently load into the new
architecture as optimizer resumes.

The guide perception cache must be tied to its exact scene/field/config/device
and dtype. Recompute its trainable projection during each rollout so gradients
remain valid. Do not detach the projection, share mutable context across jobs,
or pair a state with another scene's cached guide.

## Gates before any learning

Use F4 raw_component_objective_v3 with constant 16-step training as the single
unpromoted research control for all members. Retain F2 as an additional reference.
This choice carries forward the audited access signal; it does not promote F4
as a usable design model. Initialize both arms from the original Model C, with
only the new branch at zero. Do not continue from the trained F4 checkpoints or
give one arm extra pretraining. Freeze the F4 loss and training horizon. A
conditioning test must not also change access, weights, horizon schedule,
state-pool use, latent bounds, grid resolution or scene distribution.

Test null-branch/zero-initialization parity, deterministic context immutability,
correct scene binding, actual guide-projection gradients and malformed-input
rejection. Compare common backbone derivatives separately from the new weights;
the expanded parameter vector changes gradient clipping and must be recorded.
Use a control path that excludes the disabled branch from optimization.
Then verify actual-loop checkpoint recovery, profile a fixed CPU pilot and
freeze timing allowance before a bounded matched learning comparison.

## Judge geometry and contribution

Keep the same two development scenes, recipes, 64 updates, training seed and
evaluation grid for the first comparison, unless a separately justified protocol
is frozen. Evaluate joint connectivity/material budget, every other family and
longer-growth behavior; save all fields/checkpoints/failed outcomes. Any promising
case requires new training seeds and genuinely fresh scenes before promotion.

Retain W1 constructive witnesses and D1 direct optimization as controls. They
already establish geometric feasibility under these proxy definitions. Persistent
guidance is planner information, not evidence of scaffold-free emergence. An NCA
must eventually demonstrate useful stable refinement, recovery or variation
beyond those controls; matching their geometry alone does not establish its value.

If this bounded test fails, review representation and the NCA's intended role
before scaling compute or trying an indefinite sequence of small loss changes.
Production Studio work remains an independent planned deliverable; a research
comparison does not certify architectural, walkability or mechanical validity.

No training, paid Colab, Drive access, deployment or remote push is authorized by
this proposal alone. Document all decisions and archive the completed milestone.
