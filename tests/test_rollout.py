"""Rollout profile regressions, including exact agreement with the legacy paths.

The point of these is attribution: the shared rollout must reproduce what the
historical code did, so that a later behaviour change can be attributed to the
change rather than to the rewrite. Nothing here measures architectural quality.
"""
import random
import unittest

import numpy as np
import torch

from nca.contract import load_reference_set, to_generator_params
from nca.rollout import (HISTORICAL_PROFILES, ROLLOUT_VERSION, RolloutProfile,
                         historical_evaluation, historical_serving,
                         historical_training, resolve_steps, run_rollout)
from deploy.checkpoints import load_model_c
from deploy.model_utils import (UrbanPavilionNCA, UrbanSceneGenerator,
                                compute_corridor_target_v31)


class ProfileDeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _, _ = load_model_c()

    def test_the_three_profiles_disagree_where_the_sources_disagree(self):
        training = historical_training(self.config)
        evaluation = historical_evaluation(self.config)
        serving = historical_serving(self.config)

        self.assertEqual(training.module_mode, "train")
        self.assertEqual((evaluation.module_mode, serving.module_mode), ("eval", "eval"))
        self.assertEqual(training.firing, "delta_mask")
        self.assertEqual(evaluation.firing, "none")
        self.assertEqual(serving.firing, "none")            # request default fire_rate 1.0
        self.assertEqual(historical_serving(self.config, {"fire_rate": 0.65}).firing,
                         "state_blend")

        # Noise exists only in serving.
        self.assertEqual((training.noise_std, evaluation.noise_std), (0.0, 0.0))
        self.assertGreater(serving.noise_std, 0.0)

        # Corridor seeding differs threefold and then some.
        self.assertAlmostEqual(training.corridor_seed_scale, 0.15)
        self.assertEqual(evaluation.corridor_seed_scale, 0.0)
        self.assertAlmostEqual(serving.corridor_seed_scale, 0.005)

        # Masking differs in when and how often.
        self.assertEqual(training.corridor_mask, "seed_once")
        self.assertEqual(evaluation.corridor_mask, "none")
        self.assertEqual(serving.corridor_mask, "every_step")

        self.assertEqual(training.steps_sampler, "uniform_int")
        self.assertEqual((training.steps_min, training.steps_max), (30, 50))
        self.assertEqual(evaluation.steps, 50)

    def test_every_profile_carries_provenance_and_a_version(self):
        for name, build in HISTORICAL_PROFILES.items():
            with self.subTest(profile=name):
                profile = build(self.config)
                self.assertEqual(profile.name, name)
                self.assertEqual(profile.version, ROLLOUT_VERSION)
                self.assertTrue(profile.provenance)
                self.assertTrue(profile.notes)
                self.assertIn("profile", {"profile"})
                self.assertEqual(profile.as_dict()["name"], name)

    def test_a_variant_cannot_keep_a_historical_name(self):
        variant = historical_evaluation(self.config).replace(noise_std=0.02)
        self.assertNotEqual(variant.name, "historical-evaluation")
        self.assertIn("modified", variant.name)
        self.assertIn("historical-evaluation", variant.provenance)

    def test_contradictory_profiles_are_refused(self):
        base = dict(name="x", provenance="test", module_mode="eval", firing="none",
                    fire_rate=1.0, noise_std=0.0)
        RolloutProfile(**base)
        for override in ({"firing": "sometimes"}, {"module_mode": "inference"},
                         {"firing": "none", "fire_rate": 0.5}, {"noise_std": -1.0},
                         {"rng_source": "seeded"}, {"threshold": 1.0},
                         {"steps_sampler": "uniform_int", "steps_min": 9, "steps_max": 3}):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    RolloutProfile(**{**base, **override})

    def test_step_sampling_is_reproducible_and_reported(self):
        profile = historical_training(self.config)
        first = resolve_steps(profile, random.Random(7))
        second = resolve_steps(profile, random.Random(7))
        self.assertEqual(first, second)
        self.assertIn("python random", first[1])
        self.assertTrue(30 <= first[0] <= 50)
        self.assertEqual(resolve_steps(historical_evaluation(self.config)), (50, "fixed"))


class RolloutBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config, weights, _ = load_model_c()
        cls.model = UrbanPavilionNCA(cls.config)
        cls.model.load_state_dict(weights, strict=True)
        cls.model.eval()
        cls.scene = load_reference_set()["ref-06-minimal-smoke"]
        cls.seed_state, _ = UrbanSceneGenerator(cls.config).generate(
            to_generator_params(cls.scene), device="cpu")
        cls.corridor = compute_corridor_target_v31(cls.seed_state, cls.config)

    def test_rollout_does_not_mutate_the_seed_or_the_shared_config(self):
        before_state = self.seed_state.clone()
        before_scale = self.config["update_scale"]
        profile = historical_serving(self.config, {"steps": 2, "update_scale": 0.9})
        run_rollout(self.model, self.seed_state, profile, corridor_target=self.corridor)
        self.assertTrue(torch.equal(self.seed_state, before_state))
        self.assertEqual(self.config["update_scale"], before_scale)
        self.assertEqual(self.model.config["update_scale"], before_scale)

    def test_module_mode_is_restored_even_when_a_step_raises(self):
        profile = historical_training(self.config)
        broken = torch.zeros(1, 2, 4, 4, 4)
        self.assertFalse(self.model.training)
        with self.assertRaises(Exception):
            run_rollout(self.model, broken, profile, corridor_target=self.corridor, steps=1)
        self.assertFalse(self.model.training)

    def test_corridor_target_is_required_when_used_and_refused_when_not(self):
        with self.assertRaises(ValueError):
            run_rollout(self.model, self.seed_state,
                        historical_serving(self.config, {"steps": 1}))
        with self.assertRaises(ValueError):
            run_rollout(self.model, self.seed_state, historical_evaluation(self.config),
                        corridor_target=self.corridor, steps=1)

    def test_generator_discipline_matches_the_declared_source(self):
        profile = historical_serving(self.config, {"steps": 1})
        with self.assertRaises(ValueError):
            run_rollout(self.model, self.seed_state, profile, corridor_target=self.corridor,
                        generator=torch.Generator())
        explicit = profile.replace(rng_source="explicit")
        with self.assertRaises(ValueError):
            run_rollout(self.model, self.seed_state, explicit, corridor_target=self.corridor)

    def test_an_explicit_generator_is_reproducible_without_touching_global_state(self):
        profile = historical_serving(self.config, {"steps": 3}).replace(rng_source="explicit")
        outputs = []
        for _ in range(2):
            torch.manual_seed(1234)          # deliberately identical global state
            generator = torch.Generator().manual_seed(99)
            result = run_rollout(self.model, self.seed_state, profile,
                                 corridor_target=self.corridor, generator=generator)
            outputs.append(result["state"])
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

        torch.manual_seed(1234)
        other = run_rollout(self.model, self.seed_state, profile,
                            corridor_target=self.corridor,
                            generator=torch.Generator().manual_seed(100))
        self.assertFalse(torch.equal(outputs[0], other["state"]))

    def test_evaluation_profile_equals_the_legacy_grow_call(self):
        profile = historical_evaluation(self.config)
        legacy = self.model.grow(self.seed_state, steps=50)
        result = run_rollout(self.model, self.seed_state, profile)
        self.assertTrue(torch.equal(result["state"], legacy))
        self.assertEqual(result["steps_run"], 50)
        self.assertEqual(result["applied"]["noise_steps"], 0)
        self.assertEqual(result["applied"]["fired_steps"], 0)
        self.assertEqual(result["applied"]["masked_steps"], 0)
        self.assertFalse(result["applied"]["seed_scaled"])

    def test_training_profile_fires_on_the_delta_and_stays_stochastic(self):
        profile = historical_training(self.config)
        torch.manual_seed(5)
        first = run_rollout(self.model, self.seed_state, profile,
                            corridor_target=self.corridor, steps=4)
        torch.manual_seed(5)
        second = run_rollout(self.model, self.seed_state, profile,
                             corridor_target=self.corridor, steps=4)
        torch.manual_seed(6)
        third = run_rollout(self.model, self.seed_state, profile,
                            corridor_target=self.corridor, steps=4)
        self.assertTrue(torch.equal(first["state"], second["state"]))
        self.assertFalse(torch.equal(first["state"], third["state"]))
        self.assertEqual(first["applied"]["fired_steps"], 4)
        self.assertTrue(first["applied"]["seed_scaled"])
        # Epoch 0 falls inside corridor_mask_epochs, so the seed mask is at full strength.
        self.assertEqual(first["applied"]["seed_mask_strength"], 1.0)
        self.assertTrue(first["applied"]["seed_masked"])

    def test_seed_once_masking_anneals_over_the_epoch_schedule(self):
        profile = historical_training(self.config)
        strengths = []
        for epoch in (0, 19, 20, 40, 60, 61):
            torch.manual_seed(0)
            result = run_rollout(self.model, self.seed_state, profile,
                                 corridor_target=self.corridor, steps=0,
                                 schedule_position=epoch)
            strengths.append(round(result["applied"]["seed_mask_strength"], 4))
        self.assertEqual(strengths, [1.0, 1.0, 1.0, 0.5, 0.0, 0.0])


class ServingProfileMatchesTheLiveHandlerTests(unittest.TestCase):
    """The serving profile must reproduce /generate exactly, or attribution fails."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config, weights, _ = load_model_c()
        cls.model = UrbanPavilionNCA(cls.config)
        cls.model.load_state_dict(weights, strict=True)
        cls.model.eval()
        cls.scene = load_reference_set()["ref-06-minimal-smoke"]

    def legacy_generate(self, request):
        """The /generate loop as written, kept here verbatim as the comparison target."""
        config = self.config
        params = to_generator_params(self.scene)
        generator = UrbanSceneGenerator(config)
        seed_state, _ = generator.generate(params, device="cpu")
        corridor_target = compute_corridor_target_v31(
            seed_state, config,
            corridor_width=request["corridor_width"],
            vertical_envelope=request["vertical_envelope"])
        seed_scale = request["corridor_seed_scale"]
        if seed_scale > 0:
            struct_idx = config["ch_structure"]
            seed_state[:, struct_idx] = torch.clamp(
                seed_state[:, struct_idx] + seed_scale * corridor_target, 0.0, 1.0)
        mask_epochs = config.get("corridor_mask_epochs", 0)
        mask_anneal = config.get("corridor_mask_anneal", 0)
        original_update_scale = config.get("update_scale", 0.1)
        config["update_scale"] = request["update_scale"]
        try:
            with torch.no_grad():
                state = seed_state
                grid = config["grid_size"]
                for step in range(request["steps"]):
                    if request["noise_std"] > 0:
                        noise = torch.randn_like(state[:, config["n_frozen"]:]) * request["noise_std"]
                        state[:, config["n_frozen"]:] = torch.clamp(
                            state[:, config["n_frozen"]:] + noise, 0.0, 1.0)
                    if request["fire_rate"] < 1.0:
                        fire_mask = (torch.rand(1, 1, grid, grid, grid) < request["fire_rate"]).float()
                        old_grown = state[:, config["n_frozen"]:].clone()
                        state = self.model._step(state)
                        new_grown = state[:, config["n_frozen"]:]
                        state[:, config["n_frozen"]:] = (old_grown * (1 - fire_mask)
                                                         + new_grown * fire_mask)
                    else:
                        state = self.model._step(state)
                    mask_strength = 0.0
                    if step < mask_epochs:
                        mask_strength = 1.0
                    elif mask_anneal > 0 and step < mask_epochs + mask_anneal:
                        mask_strength = 1.0 - (step - mask_epochs) / float(mask_anneal)
                    if mask_strength > 0:
                        struct_idx = config["ch_structure"]
                        state[:, struct_idx] = (state[:, struct_idx] * (1.0 - mask_strength)
                                                + state[:, struct_idx] * corridor_target
                                                * mask_strength)
                return state
        finally:
            config["update_scale"] = original_update_scale

    def profile_generate(self, request):
        params = to_generator_params(self.scene)
        seed_state, _ = UrbanSceneGenerator(self.config).generate(params, device="cpu")
        corridor_target = compute_corridor_target_v31(
            seed_state, self.config,
            corridor_width=request["corridor_width"],
            vertical_envelope=request["vertical_envelope"])
        profile = historical_serving(self.config, request)
        return run_rollout(self.model, seed_state, profile,
                           corridor_target=corridor_target)["state"]

    def requests(self):
        default = {"steps": 12, "noise_std": 0.02, "corridor_seed_scale": 0.005,
                   "fire_rate": 1.0, "corridor_width": 1, "vertical_envelope": 1,
                   "threshold": 0.5, "update_scale": 0.1}
        return [
            ("request defaults", default),
            ("stochastic firing", {**default, "fire_rate": 0.65}),
            ("no noise", {**default, "noise_std": 0.0}),
            ("training seed scale", {**default, "corridor_seed_scale": 0.15}),
            ("past the mask schedule", {**default, "steps": 25}),
            ("altered update scale", {**default, "update_scale": 0.2}),
        ]

    def test_profile_rollout_is_bitwise_identical_to_the_legacy_loop(self):
        for label, request in self.requests():
            with self.subTest(case=label):
                torch.manual_seed(4242)
                legacy = self.legacy_generate(request)
                torch.manual_seed(4242)
                through_profile = self.profile_generate(request)
                self.assertTrue(torch.equal(legacy, through_profile),
                                f"{label}: rollout diverged from the legacy handler")

    def test_the_three_profiles_produce_measurably_different_geometry(self):
        params = to_generator_params(self.scene)
        generator = UrbanSceneGenerator(self.config)
        counts = {}
        for name, build in HISTORICAL_PROFILES.items():
            seed_state, _ = generator.generate(params, device="cpu")
            corridor = compute_corridor_target_v31(seed_state, self.config)
            profile = build(self.config)
            needs = profile.corridor_seed_scale > 0 or profile.corridor_mask != "none"
            torch.manual_seed(11)
            random.seed(11)
            result = run_rollout(self.model, seed_state, profile,
                                 corridor_target=corridor if needs else None, steps=30)
            material = result["state"][0, self.config["ch_structure"]] > profile.threshold
            counts[name] = int(material.sum().item())
        # Recorded as evidence that the profiles are not interchangeable. The
        # specific counts are a property of this scene and this checkpoint only.
        self.assertEqual(len(set(counts.values())), len(counts), counts)
        self.assertTrue(all(isinstance(v, int) for v in counts.values()))


if __name__ == "__main__":
    unittest.main()
