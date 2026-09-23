# L2_v1 and conditional R1_v1 local protocol

Frozen before outcomes on 2026-09-23. Local CPU only; no paid training or deployment.
Original notebook/checkpoint/rollout stay untouched. Interventions are explicit
research alternatives, not implicit production defaults or new constraint families.

## L2: two separate questions

**Budget matrix: 108 cases.** All 18 frozen scenes, three unchanged envelopes
(C1 thick scaffold, graph radius3 and radius6 from the C1 centerline), two mass
contracts. The site contract keeps the original non-building denominator. The
experimental envelope contract uses the legal envelope's volume as denominator,
with the same numeric 3%-12% fractions. Actual mass includes all non-building
material, including spill, for both contracts. This is an explicit reduction of
absolute material budgets, not preservation of their original physical meaning.
Record old/new voxel and cubic-meter bounds, guide size, capacity, route and
necessary compatibility. A changed denominator is a design hypothesis; do not
claim all nine objectives satisfied because these necessary checks pass.

**Gradient matrix: 54 cases.** Scenes legacy-easy-seed-000, ref-01-ground-pair,
ref-06-minimal-smoke; seeds 0/1/2; horizons 4/16; three arms:

- hard_projected: historical forward and projected guide coverage.
- hard_preclamp: exactly the same hard forward; replace only coverage's readout
  with mean relu(1-raw material candidate) on the legal guide at the final update.
  This supplies a true auxiliary coverage derivative before clipping. It does
  not claim the projected access loss's derivative itself has been repaired.
- smooth_projected: replace only material clipping with
  [softplus(20*x)-softplus(20*(x-1))]/20; hidden channels retain their hard clamp.
  Coverage is measured on the projected smooth output. This changes dynamics
  and introduces positive background mass near raw=0; record that effect.

Keep the historical .15 scaffold seed, checkpoint firing/update config, explicit
RNG, two CPU threads, and hard legality. Record continuous fields, raw candidates,
coverage/access parameter and raw-candidate gradients, finite/zero norms, binary
connectivity, illegal count, mass, and derivatives at the two ground cells exposed
by L1 (z,y,x 0,15,9 and 1,15,9). Match hard-arm forward fields exactly at each
scene/seed/horizon. A finite/nonzero derivative is not trained-model quality.

**Zero-scaffold controls: nine cases.** Same three scenes and arms, seed0, four
steps, scaffold scale effectively zero. Record fields and gradients as above.
These inspect initially absent routes and smooth background mass. No optimizer
updates in L2. Independent numerical derivative and forward-parity tests apply.

## Gate for R1: recovery mechanics only

After L2, permit a tiny local optimizer/recovery test only if: all executions and
artifact checks pass; hard arms reproduce each other's full state; all outputs
retain zero forbidden material; ground-pair seed0/four-step pre-clamp coverage
has nonzero derivatives at both previously blocked raw cells and nonzero model
parameter gradient; radius6/envelope-budget contexts pass necessary checks for
all 17 feasible scenes and retain the sealed scene's invalid status.

This gate selects a **recovery-test configuration only**: hard_preclamp,
radius6/envelope denominator, three feasible diagnostic scenes. It does not
approve a trained architectural baseline or a Colab pilot. Nine continuous
families participate with explicit unit coefficients to exercise backward and
optimizer state; these coefficients are not calibrated research weights.
Regularizer calibration/integration and architectural semantics remain pending.

R1 uses the original checkpoint, Adam lr1e-4, StepLR every2 updates gamma.9,
gradient clipping max norm1, four optimizer updates total, batch1 and 2-4 rollout
steps sampled by Python RNG. NumPy samples a scene; an explicit torch generator
controls firing; a recorded global-torch draw verifies that stream too. Seed123.
Loss proxies use 64 hops and thickness radius2. Refuse any invalid context,
nonfinite objective/gradient or frozen-context/legality violation. Record every
sample, update, loss, model field and checkpoint.

Run one uninterrupted four-update process, then a fresh process that stops after
two completed updates and saves all state, then another fresh process that restores
and finishes updates3/4. Compare model weights/buffers, Adam slots/counters, scheduler,
Python/NumPy/global-torch/firing RNG states, sample sequence, losses and final fields
exactly. Add one retry continuation from the same update2 checkpoint to verify
repeatability. Each branch has separate retained files; never overwrite checkpoints.
Save at completed-update boundaries, not mid-backward. Reject changed metadata,
parameter ordering, optimizer/scheduler classes and corrupted/truncated files.

The CPU recovery implementation does not certify CUDA, mixed precision, Colab
filesystems or abrupt mid-write recovery. Preserve hashes/source/config and local
backups. A paid pilot still needs a compute cap and artifact procedure approval.
