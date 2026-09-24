"""Admission checks for the frozen MG4 stress design, not generator success tests."""
from pathlib import Path
import importlib.util
import json
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mg4_runner',ROOT/'scripts/run_site_generalization.py')
mg4=importlib.util.module_from_spec(spec);spec.loader.exec_module(mg4)


class SiteGeneralizationTests(unittest.TestCase):
    def test_frozen_design_is_unique_and_declared_controls_partition(self):
        recipe=json.loads((ROOT/'experiments/configs/MG4-sites.json').read_bytes());sites=mg4.load_design(recipe)
        self.assertEqual(len(sites),20);self.assertEqual(recipe['seeds'],[3,4,5])
        for s in sites:
            scene=s['scene'];self.assertEqual(len(scene['entrances']),2)
            if s['partition_control']:
                wall=next(b for b in scene['buildings'] if b['id']=='B_partition')
                self.assertEqual(wall['y'],[0,32]);self.assertEqual(wall['z'],[0,32])
                self.assertLess(scene['entrances'][0]['x'],wall['x'][0]);self.assertGreater(scene['entrances'][1]['x'],wall['x'][1])

    def test_geometry_identity_ignores_descriptive_metadata(self):
        scene=mg4.generation_scenes()[0][1];changed={**scene,'scene_id':'other','description':'other','notes':[]}
        self.assertEqual(mg4.geometry_key(scene),mg4.geometry_key(changed))

    def test_sampler_records_native_residency_and_stops_after_error(self):
        value=mg4.memory_bytes();self.assertGreater(value['rss_bytes'],0)
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with mg4.MemorySampler(.001) as sampler:
                time.sleep(.005);raise RuntimeError('sentinel')
        self.assertFalse(sampler.thread.is_alive());report=sampler.report()
        self.assertGreaterEqual(report['samples'],2);self.assertGreaterEqual(report['sampled_peak_rss_bytes'],report['initial_rss_bytes'])

    def test_valid_only_diversity_does_not_invent_pairs(self):
        self.assertIsNone(mg4.diversity([])['mean_jaccard_distance'])
        self.assertIsNone(mg4.diversity([{(1,2,3)}])['mean_jaccard_distance'])
        result=mg4.diversity([{(1,2,3)},{(1,2,3),(2,3,4)}])
        self.assertEqual(result['mean_jaccard_distance'],.5);self.assertEqual(result['pairs'],1)


if __name__=='__main__':unittest.main()
