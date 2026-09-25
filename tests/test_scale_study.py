import unittest
from copy import deepcopy
import numpy as np
from nca.scale_study import decode_grid,embed_field,embedded_scene,scale_sites,scale_context
from nca.massing_cases import target_scenes,target_controls,target_context
from nca.massing_targets import evaluate_targets
from deploy.checkpoints import load_model_c


class ScaleStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.config,_,_=load_model_c(device='cpu')

    def test_decoder_uses_scene_size_and_rejects_bad_coordinates(self):
        self.assertTrue(decode_grid([[63,62,61]],64)[63,62,61])
        for coords in ([[64,0,0]],[[-1,0,0]],[[1.,2.,3.]],[[True,False,True]],[[1,2]]):
            with self.assertRaises(ValueError):decode_grid(coords,64)
        self.assertEqual(decode_grid([],48).shape,(48,)*3)

    def test_context_is_exact_at_historical_size(self):
        scene=dict(target_scenes())['aligned']
        old,domain,_=target_context(scene,self.config)
        fields,new,report=scale_context(scene,self.config)
        self.assertTrue(np.array_equal(domain,new))
        for k,v in fields.items():self.assertTrue(np.array_equal(v,old[k]))
        self.assertEqual(report['state_scene_problems'],[])
        self.assertEqual(self.config['grid_size'],32)

    def test_embedded_domain_and_scores_preserve_physical_site(self):
        scene=dict(target_scenes())['aligned'];fields,domain,_=scale_context(scene,self.config)
        field=target_controls(scene,domain)['compact_mass'];expected,_=evaluate_targets(field,scene,fields,domain)
        for size in (48,64):
            larger=embedded_scene(scene,size);f,d,report=scale_context(larger,self.config)
            self.assertTrue(np.array_equal(d,embed_field(domain,size)))
            actual,_=evaluate_targets(embed_field(field,size),larger,f,d)
            self.assertEqual(expected,actual)
            self.assertAlmostEqual(report['ground_band_m'],4.8)
            self.assertAlmostEqual(report['bulk_width_m'],2.4)
            self.assertTrue(all(e['extent_m']==1.6 for e in report['interfaces'].values()))

    def test_larger_sites_actually_increase_domain(self):
        previous={}
        for size in (48,64):
            for entry in scale_sites(size):
                fields,domain,report=scale_context(entry['scene'],self.config)
                self.assertEqual(domain.shape,(size,)*3)
                self.assertFalse((domain&fields['existing']).any())
                if entry['kind'] in previous:self.assertGreater(int(domain.sum()),previous[entry['kind']])
                previous[entry['kind']]=int(domain.sum())
                self.assertEqual(report['effective_context_config']['grid_size'],size)

    def test_finer_interface_is_refused_not_silently_shrunk(self):
        scene=deepcopy(dict(target_scenes())['aligned']);scene['entrances'][0]['extent']=3
        with self.assertRaisesRegex(ValueError,'fixed'):scale_context(scene,self.config)


if __name__=='__main__':unittest.main()
