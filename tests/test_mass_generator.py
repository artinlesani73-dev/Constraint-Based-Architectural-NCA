from dataclasses import replace
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from deploy.checkpoints import load_model_c
from nca.mass_generator import generate_mass, MassGeneratorSpec, pairwise_diversity
from nca.mass_generation_cases import generation_scenes, challenge_fields
from nca.massing_cases import target_context
from nca.massing_targets import cube_supported
from nca.evaluation import flood_fill


class MassGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_model_c(device='cpu')[0]
        cls.scenes = dict(generation_scenes())
        cls.scene = cls.scenes['aligned']
        cls.fields, cls.domain, _ = target_context(cls.scene, cls.config)
        cls.spec = MassGeneratorSpec(target_fraction=.16)

    def run_case(self, seed=41, spec=None):
        return generate_mass(self.scene, self.fields, self.domain, seed, spec or self.spec)

    def test_deterministic_seed_replay_and_input_preservation(self):
        before = self.domain.copy(); saved = deepcopy(self.fields)
        a, route, r = self.run_case(); b, route2, s = self.run_case()
        np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(route,route2)
        self.assertGreater(a.sum(), 0)
        r.pop('wall_seconds'); s.pop('wall_seconds'); self.assertEqual(r,s)
        np.testing.assert_array_equal(self.domain,before)
        def same(left,right):
            if isinstance(left,dict):
                self.assertEqual(set(left),set(right))
                for key in left:same(left[key],right[key])
            elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
            else:self.assertEqual(left,right)
        same(saved,self.fields)

    def test_union_reconstructs_exactly_and_is_legal_bulk_connected(self):
        a, route, r = self.run_case()
        rebuild=np.zeros_like(a); w=r['cube_width_cells']
        for z,y,x in r['selected_origins_zyx']: rebuild[z:z+w,y:y+w,x:x+w]=True
        np.testing.assert_array_equal(a,rebuild)
        np.testing.assert_array_equal(cube_supported(a,w),a)
        self.assertFalse((a & ~self.domain).any()); self.assertFalse((route & ~a).any())
        seed=np.zeros_like(a); seed[tuple(np.argwhere(a)[0])]=True
        np.testing.assert_array_equal(flood_fill(a,seed),a)
        path=np.array(r['route_origins_zyx'])
        self.assertTrue((np.abs(np.diff(path,axis=0)).sum(axis=1)==1).all())

    def test_volume_request_rounding_and_last_cube_overshoot(self):
        a,_,r=self.run_case()
        self.assertEqual(r['requested_voxels'],int(np.ceil(self.domain.sum()*.16)))
        self.assertEqual(r['status'],'target_reached')
        self.assertGreaterEqual(a.sum(),r['requested_voxels'])
        self.assertLess(r['target_error_voxels'],r['cube_width_cells']**3)

    def test_small_request_returns_route_without_discarding_it(self):
        a,route,r=self.run_case(spec=replace(self.spec,target_fraction=.001))
        self.assertEqual(r['status'],'route_at_or_above_request')
        np.testing.assert_array_equal(a,route); self.assertGreater(r['target_error_voxels'],0)

    def test_blocked_graph_does_not_union_disconnected_origins(self):
        scene=self.scenes['blocked_gap']; fields,domain,_=target_context(scene,self.config)
        a,_,r=generate_mass(scene,fields,domain,41,self.spec)
        self.assertEqual(r['status'],'no_cube_route'); self.assertFalse(a.any())

    def test_growth_exhaustion_preserves_partial_field(self):
        a,_,r=self.run_case(spec=replace(self.spec,target_fraction=1))
        self.assertEqual(r['status'],'component_exhausted')
        self.assertGreater(a.sum(),0); self.assertFalse(r['target_reached'])

    def test_explicit_clock_cap(self):
        with patch('nca.mass_generator.time.perf_counter',side_effect=[0,100,101]):
            a,_,r=self.run_case()
        self.assertEqual(r['status'],'time_limit'); self.assertFalse(a.any())

    def test_no_supported_endpoint_and_no_cube_are_explicit(self):
        fields={**self.fields,'support_boundary':np.zeros_like(self.domain)}
        _,_,r=generate_mass(self.scene,fields,self.domain,41,self.spec)
        self.assertEqual(r['status'],'no_supported_interface_cube')
        _,_,r=self.run_case(spec=replace(self.spec,cube_m=100))
        self.assertEqual(r['status'],'no_legal_cube')

    def test_more_than_two_interfaces_is_not_silently_ignored(self):
        scene=deepcopy(self.scene)
        scene['entrances'].append({'id':'third','kind':'facade','x':8,'y':21,'z':10,'extent':2})
        a,_,r=generate_mass(scene,self.fields,self.domain,41,self.spec)
        self.assertEqual(r['status'],'unsupported_interface_count');self.assertFalse(a.any())

    def test_invalid_inputs(self):
        for kwargs in ({'cube_m':False},{'target_fraction':2},{'max_seconds':float('nan')}):
            with self.assertRaises(ValueError):MassGeneratorSpec(**kwargs)
        for seed in (True,-1,1.5):
            with self.assertRaises(ValueError):self.run_case(seed=seed)
        with self.assertRaises(ValueError):generate_mass(self.scene,self.fields,np.ones_like(self.domain),1)

    def test_diversity_duplicates_and_no_samples_are_explicit(self):
        a=np.zeros((3,3,3),bool);a[0]=True;b=~a
        r=pairwise_diversity([a,a,b])
        self.assertEqual(r['unique_valid_fields'],2);self.assertEqual(r['pair_count'],3)
        self.assertEqual(r['jaccard_distances'],[0,1,1])
        self.assertIsNone(pairwise_diversity([])['mean_jaccard_distance'])
        self.assertIsNone(pairwise_diversity([a])['mean_jaccard_distance'])

    def test_challenges_are_separate_unclipped_fields(self):
        cases=challenge_fields(self.scene,self.domain)
        self.assertEqual(len(cases),4)
        self.assertFalse(cases['quarter_turn_field'][:,:,8:10].any())
        self.assertTrue((cases['bulky_lattice']).any())


if __name__=='__main__':unittest.main()
