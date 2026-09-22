"""Reporting regressions: empty, failed and unscorable cases stay visible."""
import copy
import unittest
import torch
from deploy.checkpoints import load_model_c
from nca.contract import load_reference_set
from nca.e0 import evaluate, aggregate
from nca.legacy_scenes import legacy_seed_state
from pathlib import Path


class E0ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _, _ = load_model_c()
        root = Path(__file__).resolve().parents[1]
        cls.scene = load_reference_set(root / 'experiments/scenes/legacy_easy_v1')['legacy-easy-seed-000']
        cls.state = legacy_seed_state(cls.scene, cls.config)

    def test_empty_output_is_not_a_successful_architecture(self):
        result = evaluate(self.state, self.config, self.scene)
        self.assertFalse(result['legality']['nonempty'])
        self.assertIsNone(result['legality']['illegal_fraction'])
        self.assertFalse(result['connectivity']['all_connected'])
        self.assertIsNone(result['support']['supported_fraction'])
        self.assertIsNone(result['thickness_proxy']['core_fraction'])
        self.assertEqual(result['ground']['open_fraction'], 1.0)

    def test_failed_and_unscorable_cases_keep_separate_denominators(self):
        metrics = evaluate(self.state, self.config, self.scene)
        base = {'scene_set': 'test', 'profile_name': 'example', 'seed': 0,
                'status': 'completed', 'rollout_seconds': 1.0, 'metrics': metrics}
        unscorable = copy.deepcopy(base)
        unscorable['metrics']['connectivity'] = {'status': 'unscorable', 'all_connected': None}
        failed = {'scene_set': 'test', 'profile_name': 'example', 'seed': 0, 'status': 'failed'}
        group = aggregate([base, unscorable, failed])[0]
        self.assertEqual((group['cases'], group['completed'], group['failed']), (3, 2, 1))
        self.assertEqual(group['connectivity_scored_cases'], 1)
        self.assertEqual(group['connectivity_unscorable_cases'], 1)
        self.assertEqual(group['connected_cases'], 0)
        self.assertEqual(group['empty_cases'], 2)


if __name__ == '__main__':
    unittest.main()
