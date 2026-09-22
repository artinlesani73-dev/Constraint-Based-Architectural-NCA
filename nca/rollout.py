"""Shared rollout with named historical profiles, version rollout_v1.

One implementation, every behavioural axis declared in a profile rather than
implied by which file called it. The three historical profiles are reconstructed
from primary sources and each records where it came from:

* ``historical-training`` from ``ArchitecturalIntentTrainerV31.train_epoch`` in
  ``notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb``.
* ``historical-evaluation`` from ``ArchitecturalIntentTrainerV31.evaluate`` in
  the same notebook, which is what produced ``v31_evaluation.json``.
* ``historical-serving`` from the ``/generate`` handler in ``deploy/server.py``.

The legacy code paths are untouched and remain the reference for exact
comparison. Running a profile reproduces forward dynamics only; nothing here
trains, and no profile is a repaired or recommended configuration.

What the three actually disagree about, all confirmed in source:

| Axis | training | evaluation | serving |
|---|---|---|---|
| Module mode | `train()` | `eval()` | `eval()` |
| Firing | mask on the update delta, rate 0.65 | none | blend of whole states, only if rate < 1; request default 1.0 |
| Per-step noise | none | none | `randn_like` on grown channels, request default 0.02 |
| Steps | `random.randint(30, 50)` per epoch | fixed 50 | request default 50 |
| Corridor seed scale | 0.15 | none | request default 0.005 |
| Corridor mask | once on the seed, gated on epoch | none | every step, gated on step index |

``z_taper_strength`` and ``z_taper_floor`` appear in both configurations and in
no code path in either the notebook or the deployment. They are dead keys, not a
training behaviour that serving dropped.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import random

import torch

from nca.contract import MATERIAL_THRESHOLD_DEFAULT

ROLLOUT_VERSION = "rollout_v1"

FIRING_MODES = ("none", "delta_mask", "state_blend")
CORRIDOR_MASK_MODES = ("none", "seed_once", "every_step")
RNG_SOURCES = ("global", "explicit")


@dataclass(frozen=True)
class RolloutProfile:
    """A complete, explicit description of one rollout's dynamics."""

    name: str
    provenance: str
    module_mode: str                     # "train" or "eval"
    firing: str                          # see FIRING_MODES
    fire_rate: float
    noise_std: float
    steps: int = 50
    steps_sampler: str = "fixed"         # "fixed" or "uniform_int"
    steps_min: int = 50
    steps_max: int = 50
    corridor_seed_scale: float = 0.0
    corridor_mask: str = "none"
    corridor_mask_epochs: int = 0
    corridor_mask_anneal: int = 0
    corridor_width: int = 1
    vertical_envelope: int = 1
    update_scale: float = 0.1
    threshold: float = MATERIAL_THRESHOLD_DEFAULT
    rng_source: str = "global"
    scene_distribution: str = "unspecified"
    version: str = ROLLOUT_VERSION
    notes: tuple = ()

    def __post_init__(self):
        if not self.name.strip() or not self.provenance.strip():
            raise ValueError("A profile needs a name and a provenance string")
        if self.module_mode not in ("train", "eval"):
            raise ValueError("module_mode must be 'train' or 'eval'")
        if self.firing not in FIRING_MODES:
            raise ValueError(f"firing must be one of {FIRING_MODES}")
        if self.corridor_mask not in CORRIDOR_MASK_MODES:
            raise ValueError(f"corridor_mask must be one of {CORRIDOR_MASK_MODES}")
        if self.rng_source not in RNG_SOURCES:
            raise ValueError(f"rng_source must be one of {RNG_SOURCES}")
        if self.steps_sampler not in ("fixed", "uniform_int"):
            raise ValueError("steps_sampler must be 'fixed' or 'uniform_int'")
        if not 0 <= self.fire_rate <= 1:
            raise ValueError("fire_rate must lie in [0, 1]")
        if self.noise_std < 0:
            raise ValueError("noise_std cannot be negative")
        if self.firing == "none" and self.fire_rate != 1.0:
            raise ValueError("firing 'none' must declare fire_rate 1.0 to stay unambiguous")
        if self.steps_sampler == "uniform_int" and not self.steps_min <= self.steps_max:
            raise ValueError("steps_min must not exceed steps_max")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")

    def as_dict(self):
        record = asdict(self)
        record["notes"] = list(self.notes)
        return record

    def replace(self, **changes):
        """A new profile with an amended name, so a variant is never mistaken for a historical one."""
        if "name" not in changes:
            changes["name"] = f"{self.name}+modified"
        if "provenance" not in changes:
            changes["provenance"] = f"derived from {self.name}: {sorted(changes)}"
        merged = {**self.as_dict(), **changes}
        merged.pop("version", None)
        merged["notes"] = tuple(merged.get("notes", ()))
        return RolloutProfile(**merged)


def historical_training(config):
    """Forward dynamics of ``train_epoch``. Reproduces the rollout, not the training."""
    return RolloutProfile(
        name="historical-training",
        provenance=("NB02_AllConstraints_v3_1_C.ipynb, ArchitecturalIntentTrainerV31."
                    "train_epoch; embedded checkpoint config"),
        module_mode="train",
        firing="delta_mask",
        fire_rate=float(config["fire_rate"]),
        noise_std=0.0,
        steps_sampler="uniform_int",
        steps_min=int(config["steps_min"]),
        steps_max=int(config["steps_max"]),
        steps=int(config["steps_max"]),
        corridor_seed_scale=float(config["corridor_seed_scale"]),
        corridor_mask="seed_once",
        corridor_mask_epochs=int(config.get("corridor_mask_epochs", 0)),
        corridor_mask_anneal=int(config.get("corridor_mask_anneal", 0)),
        corridor_width=int(config.get("corridor_width", 1)),
        vertical_envelope=int(config.get("vertical_envelope", 1)),
        update_scale=float(config["update_scale"]),
        rng_source="global",
        scene_distribution=("procedural 'easy' difficulty; every access point was typed "
                            "'facade', so the ground-anchor branch never executed"),
        notes=(
            "No per-step noise: the training loop adds none and _step adds none.",
            "Step count is drawn per epoch from Python's `random`, not torch.",
            "The corridor mask is applied once to the seed before the rollout, gated on "
            "the epoch index, and annealed over epochs 20-60.",
            "Running this profile does not train, compute a loss or update weights.",
        ),
    )


def historical_evaluation(config):
    """Dynamics of ``evaluate``, the source of ``v31_evaluation.json``."""
    return RolloutProfile(
        name="historical-evaluation",
        provenance=("NB02_AllConstraints_v3_1_C.ipynb, ArchitecturalIntentTrainerV31."
                    "evaluate, which calls model.grow(scene, steps=50)"),
        module_mode="eval",
        firing="none",
        fire_rate=1.0,
        noise_std=0.0,
        steps=50,
        steps_sampler="fixed",
        steps_min=50,
        steps_max=50,
        corridor_seed_scale=0.0,
        corridor_mask="none",
        corridor_width=int(config.get("corridor_width", 1)),
        vertical_envelope=int(config.get("vertical_envelope", 1)),
        update_scale=float(config["update_scale"]),
        rng_source="global",
        scene_distribution="procedural 'easy' difficulty, as in training",
        notes=(
            "The corridor target was computed and used for scoring only. The rollout "
            "received no corridor seeding and no corridor mask, unlike training.",
            "`grow` forces eval(), so no firing occurs at all.",
            "The recorded aggregate in v31_evaluation.json came from this profile over "
            "50 procedurally sampled scenes, not from any frozen scene set.",
        ),
    )


def historical_serving(config, request=None):
    """Dynamics of the ``/generate`` handler, at its own request defaults."""
    request = dict(request or {})
    fire_rate = float(request.get("fire_rate", 1.0))
    return RolloutProfile(
        name="historical-serving",
        provenance="deploy/server.py, /generate handler; request defaults where unset",
        module_mode="eval",
        firing="state_blend" if fire_rate < 1.0 else "none",
        fire_rate=fire_rate,
        noise_std=float(request.get("noise_std", 0.02)),
        steps=int(request.get("steps", 50)),
        steps_sampler="fixed",
        steps_min=int(request.get("steps", 50)),
        steps_max=int(request.get("steps", 50)),
        corridor_seed_scale=float(request.get("corridor_seed_scale", 0.005)),
        corridor_mask="every_step",
        corridor_mask_epochs=int(config.get("corridor_mask_epochs", 0)),
        corridor_mask_anneal=int(config.get("corridor_mask_anneal", 0)),
        corridor_width=int(request.get("corridor_width", 1)),
        vertical_envelope=int(request.get("vertical_envelope", 1)),
        update_scale=float(request.get("update_scale", 0.1)),
        threshold=float(request.get("threshold", MATERIAL_THRESHOLD_DEFAULT)),
        rng_source="global",
        scene_distribution="user-supplied building and access parameters",
        notes=(
            "Per-step noise on the grown channels is a deployment addition; neither the "
            "training loop nor the notebook model adds noise.",
            "Firing blends whole states after the step, whereas training masked the "
            "update delta before it was applied. These are different operations.",
            "The corridor mask reuses corridor_mask_epochs and corridor_mask_anneal, "
            "which are epoch counts, as rollout step indices.",
            "The handler writes update_scale into the shared config dictionary; this "
            "module scopes the override instead, which is a narrower behaviour.",
        ),
    )


HISTORICAL_PROFILES = {
    "historical-training": historical_training,
    "historical-evaluation": historical_evaluation,
    "historical-serving": historical_serving,
}


@contextmanager
def _scoped_config(model, overrides):
    """Apply config overrides for one rollout without touching the shared dictionary.

    The legacy handler mutated the shared config and restored it afterwards,
    which leaks on an exception and is visible to a concurrent request. Swapping
    a copy is narrower but still not per-job isolation: two rollouts sharing one
    model instance would still contend. Real isolation is M4 work.
    """
    original = model.config
    model.config = {**original, **overrides}
    try:
        yield model.config
    finally:
        model.config = original


def resolve_steps(profile, python_random=None):
    """Step count plus how it was obtained, for the run record."""
    if profile.steps_sampler == "fixed":
        return profile.steps, "fixed"
    source = python_random if python_random is not None else random
    drawn = source.randint(profile.steps_min, profile.steps_max)
    return drawn, f"uniform_int[{profile.steps_min},{profile.steps_max}] via python random"


def _mask_strength(profile, position):
    epochs = profile.corridor_mask_epochs
    anneal = profile.corridor_mask_anneal
    if position < epochs:
        return 1.0
    if anneal > 0 and position < epochs + anneal:
        return 1.0 - (position - epochs) / float(anneal)
    return 0.0


def run_rollout(model, seed_state, profile, corridor_target=None, steps=None,
                schedule_position=0, generator=None, python_random=None):
    """Run one rollout under ``profile`` and return the state plus a full record.

    ``seed_state`` is never mutated. ``corridor_target`` is required by any
    profile that seeds or masks with it, and is rejected as unused otherwise, so
    a caller cannot silently pass a corridor that does nothing.

    ``schedule_position`` is the epoch index for ``seed_once`` masking; per-step
    masking uses the step index and ignores it. ``generator`` is required when
    ``rng_source`` is ``"explicit"`` and refused when it is ``"global"``, because
    reproducing a historical stream means drawing from the same source the
    original did.

    The returned record states what was applied, not how good the result is.
    """
    if not isinstance(profile, RolloutProfile):
        raise ValueError("profile must be a RolloutProfile")
    config = model.config
    needs_corridor = profile.corridor_seed_scale > 0 or profile.corridor_mask != "none"
    if needs_corridor and corridor_target is None:
        raise ValueError(f"{profile.name} seeds or masks with a corridor target; supply one")
    if corridor_target is not None and not needs_corridor:
        raise ValueError(f"{profile.name} uses no corridor target; passing one would be misleading")
    if profile.rng_source == "explicit" and generator is None:
        raise ValueError("rng_source 'explicit' requires a torch.Generator")
    if profile.rng_source == "global" and generator is not None:
        raise ValueError("rng_source 'global' draws from the default generator; refusing a generator")

    if steps is None:
        steps, steps_origin = resolve_steps(profile, python_random)
    else:
        steps_origin = "supplied by caller"
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    grown_start = config["n_frozen"]
    structure_index = config["ch_structure"]
    state = seed_state.clone()
    applied = {"noise_steps": 0, "fired_steps": 0, "masked_steps": 0,
               "seed_scaled": False, "seed_masked": False, "seed_mask_strength": 0.0}

    was_training = model.training
    model.train(profile.module_mode == "train")
    try:
        with _scoped_config(model, {"update_scale": profile.update_scale}):
            if profile.corridor_seed_scale > 0:
                state[:, structure_index] = torch.clamp(
                    state[:, structure_index]
                    + profile.corridor_seed_scale * corridor_target, 0.0, 1.0)
                applied["seed_scaled"] = True

            if profile.corridor_mask == "seed_once":
                strength = _mask_strength(profile, schedule_position)
                applied["seed_mask_strength"] = strength
                if strength > 0:
                    state[:, structure_index] = (
                        state[:, structure_index] * (1.0 - strength)
                        + state[:, structure_index] * corridor_target * strength)
                    applied["seed_masked"] = True

            grid = config["grid_size"]
            context = torch.no_grad() if profile.module_mode == "eval" else torch.enable_grad()
            with context:
                for step in range(steps):
                    if profile.noise_std > 0:
                        shape = state[:, grown_start:].shape
                        if generator is None:
                            noise = torch.randn(shape, device=state.device,
                                                dtype=state.dtype) * profile.noise_std
                        else:
                            noise = torch.randn(shape, device=state.device, dtype=state.dtype,
                                                generator=generator) * profile.noise_std
                        state = torch.cat([
                            state[:, :grown_start],
                            torch.clamp(state[:, grown_start:] + noise, 0.0, 1.0)], dim=1)
                        applied["noise_steps"] += 1

                    if profile.firing == "state_blend" and profile.fire_rate < 1.0:
                        batch = state.shape[0]
                        rand_shape = (batch, 1, grid, grid, grid)
                        if generator is None:
                            draw = torch.rand(rand_shape, device=state.device)
                        else:
                            draw = torch.rand(rand_shape, device=state.device, generator=generator)
                        fire_mask = (draw < profile.fire_rate).to(state.dtype)
                        previous = state[:, grown_start:].clone()
                        stepped = model._step(state)
                        blended = previous * (1 - fire_mask) + stepped[:, grown_start:] * fire_mask
                        state = torch.cat([stepped[:, :grown_start], blended], dim=1)
                        applied["fired_steps"] += 1
                    else:
                        # "delta_mask" is applied inside _step while module_mode is
                        # "train"; "none" leaves the step deterministic.
                        state = model._step(state)
                        if profile.firing == "delta_mask":
                            applied["fired_steps"] += 1

                    if profile.corridor_mask == "every_step":
                        strength = _mask_strength(profile, step)
                        if strength > 0:
                            masked = (state[:, structure_index] * (1.0 - strength)
                                      + state[:, structure_index] * corridor_target * strength)
                            state = torch.cat([
                                state[:, :structure_index], masked.unsqueeze(1),
                                state[:, structure_index + 1:]], dim=1)
                            applied["masked_steps"] += 1
    finally:
        model.train(was_training)

    return {
        "rollout_version": ROLLOUT_VERSION,
        "profile": profile.as_dict(),
        "steps_run": steps,
        "steps_origin": steps_origin,
        "schedule_position": schedule_position,
        "applied": applied,
        "state": state,
        "interpretation": ("Reproduced forward dynamics under one declared profile. "
                           "No quality, latency or training claim."),
    }
