"""Behavioral ablation checks and a read-only notebook training-forward oracle."""
import ast
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from deploy import model_utils
from deploy.checkpoints import load_model_c
from nca.contract import load_reference_set
from nca.legacy_scenes import legacy_seed_state
from nca.rollout import historical_training, run_rollout

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb'


def notebook_training_oracle(config, weights, state):
    """Execute original class definitions and train_epoch only through `final`.

    The notebook is never executed wholesale. AST slicing stops before any
    losses, backward calls or optimizer updates; original forward statements
    are compiled without rewriting them. Seed generation is supplied separately,
    already verified against the notebook in test_legacy_scenes.py.
    """
    definitions = {}
    for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        try:
            tree = ast.parse(source)
        except SyntaxError:  # Colab installation/magic cells are not Python definitions.
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name in definitions:
                    raise AssertionError(f'Duplicate notebook definition: {node.name}')
                definitions[node.name] = node
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'random': random,
                 'Tuple': tuple, 'List': list, 'Dict': dict}
    # Compile the notebook's own perception, legality, corridor and model code.
    selected = {'Perceive3D', 'LocalLegalityLoss', '_extract_access_centroids',
                'compute_corridor_target_v31', 'UrbanPavilionNCA'}
    while True:
        dependencies = {n.id for name in selected for n in ast.walk(definitions[name])
                        if isinstance(n, ast.Name) and n.id in definitions}
        if dependencies <= selected:
            break
        selected |= dependencies
    for name in sorted(selected):
        exec(compile(ast.Module(body=[definitions[name]], type_ignores=[]),
                     str(NOTEBOOK), 'exec'), namespace)
    trainer = definitions['ArchitecturalIntentTrainerV31']
    method = copy.deepcopy(next(n for n in trainer.body if isinstance(n, ast.FunctionDef)
                                and n.name == 'train_epoch'))
    final_at = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'final' for t in n.targets))
    method.body = method.body[:final_at + 1] + [ast.Return(value=ast.Tuple(
        elts=[ast.Name(id=n, ctx=ast.Load()) for n in ('final', 'steps', 'corridor_target')],
        ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(NOTEBOOK), 'exec'), namespace)
    model = namespace['UrbanPavilionNCA'](dict(config))
    model.load_state_dict(weights, strict=True)
    holder = SimpleNamespace(model=model, config=model.config, device='cpu',
                             legality_loss=namespace['LocalLegalityLoss'](model.config),
                             scene_gen=SimpleNamespace(batch=lambda *args: state.clone()))
    return lambda epoch: namespace['train_epoch'](holder, epoch)


class RolloutControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config, cls.weights, _ = load_model_c()
        cls.model = model_utils.UrbanPavilionNCA(dict(cls.config))
        cls.model.load_state_dict(cls.weights, strict=True)
        cls.model.eval()
        scene = load_reference_set(ROOT / 'experiments/scenes/legacy_easy_v1')['legacy-easy-seed-004']
        cls.state = legacy_seed_state(scene, cls.config)
        cls.corridor = model_utils.compute_corridor_target_v31(cls.state, cls.config)

    def test_delta_mask_uses_the_declared_rate_and_restores_config(self):
        original = self.model.config
        baseline = historical_training(self.config)
        for rate in (0.0, 0.35, 1.0):
            with self.subTest(rate=rate):
                profile = baseline.replace(fire_rate=rate, corridor_seed_scale=0,
                                           corridor_mask='none')
                torch.manual_seed(71)
                actual = run_rollout(self.model, self.state, profile, steps=3)['state']
                oracle = model_utils.UrbanPavilionNCA({**self.config, 'fire_rate': rate})
                oracle.load_state_dict(self.weights)
                oracle.train()
                torch.manual_seed(71)
                expected = oracle(self.state.clone(), steps=3)
                self.assertTrue(torch.equal(actual, expected))
                self.assertIs(self.model.config, original)
                self.assertEqual(original['fire_rate'], self.config['fire_rate'])
                if rate == 0:
                    self.assertTrue(torch.equal(actual, self.state))

    def test_explicit_delta_rng_is_independent_and_leaves_global_rng_untouched(self):
        profile = historical_training(self.config).replace(rng_source='explicit')
        outputs = []
        for global_seed, explicit_seed in ((12, 5), (999, 5), (12, 6)):
            torch.manual_seed(global_seed)
            before = torch.random.get_rng_state().clone()
            generator = torch.Generator().manual_seed(explicit_seed)
            initial_generator = generator.get_state().clone()
            outputs.append(run_rollout(self.model, self.state, profile,
                           corridor_target=self.corridor, steps=3, generator=generator)['state'])
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertFalse(torch.equal(initial_generator, generator.get_state()))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertFalse(torch.equal(outputs[0], outputs[2]))

    def test_contradictory_module_mode_and_firing_are_rejected(self):
        profile = historical_training(self.config)
        for change in ({'module_mode': 'eval'}, {'firing': 'none', 'fire_rate': 1.0},
                       {'firing': 'state_blend'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                profile.replace(**change)

    def test_training_forward_matches_original_notebook_at_each_mask_phase(self):
        # Short step sampling keeps the oracle cheap; epoch phases exercise the
        # original mask schedule. This is forward parity, not a training test.
        config = {**self.config, 'steps_min': 2, 'steps_max': 4}
        for batch, epoch in ((1, 0), (1, 40), (2, 60)):
            with self.subTest(batch=batch, epoch=epoch):
                state = self.state.repeat(batch, 1, 1, 1, 1)
                oracle = notebook_training_oracle(config, self.weights, state)
                random.seed(44)
                torch.manual_seed(55)
                expected, steps, corridor = oracle(epoch)
                random.seed(44)
                torch.manual_seed(55)
                actual = run_rollout(self.model, state, historical_training(config),
                                     corridor_target=corridor, schedule_position=epoch)
                self.assertEqual(actual['steps_run'], steps)
                self.assertTrue(torch.equal(actual['state'], expected))


if __name__ == '__main__':
    unittest.main()
