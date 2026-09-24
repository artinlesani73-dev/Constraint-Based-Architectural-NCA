"""Synthetic coverage accounting and finite priority tests, not tuned case outcomes."""
import unittest
import numpy as np
from nca.coverage_mass_generator import box_counts,coverage_parts,grow_coverage
from nca.massing_targets import cube_supported,evaluate_targets
from nca.mass_generation_cases import generation_scenes
from nca.massing_cases import target_context
from deploy.checkpoints import load_model_c


class CoverageGrowthTests(unittest.TestCase):
    def test_summed_volume_matches_brute_cubes(self):
        a=np.random.default_rng(98).random((5,6,7))>.4
        for width in (1,2,3):
            expected=np.lib.stride_tricks.sliding_window_view(a,(width,)*3).sum(axis=(-3,-2,-1))
            np.testing.assert_array_equal(box_counts(a,width),expected)

    def test_thirds_match_independent_evaluator_and_exact_tolerance(self):
        scene=dict(generation_scenes())['partial_obstruction']
        fields,domain,_=target_context(scene,load_model_c(device='cpu')[0])
        parts,minima=coverage_parts(domain,.08)
        result,_=evaluate_targets(domain,scene,fields,domain)
        self.assertEqual([int(p.sum()) for p in parts],[t['domain_cells'] for t in result['thirds']])
        np.testing.assert_array_equal(sum(p.astype(int) for p in parts),domain.astype(int))
        for p,m in zip(parts,minima):
            n=int(p.sum());self.assertGreaterEqual(m/n+1e-10,.08)
            self.assertLess((m-1)/n+1e-10,.08)

    def test_priority_moves_after_each_deficit_is_met(self):
        domain=np.ones((1,2,6),bool);field=np.zeros_like(domain);field[:,0,:]=True
        parts,_=coverage_parts(domain,.08);cost=np.full(domain.shape,10.);cost[0,1,2:4]=0
        status,report=grow_coverage(field,np.zeros_like(field),1,8,domain,
            [(0,0,x) for x in range(6)],cost,.15,lambda:False,parts,np.array([3,3,2]))
        self.assertEqual(status,'target_reached')
        self.assertEqual(report['trace'][0]['origin_zyx'],[0,1,2])
        self.assertIn(report['trace'][1]['origin_zyx'],[[0,1,0],[0,1,1]])
        self.assertTrue(report['coverage_minima_met']);self.assertEqual(report['final_third_cells'],[3,3,2])

    def test_cube_unions_and_overlap_counts_are_qualified_bulk(self):
        domain=np.ones((3,3,6),bool);field=np.zeros_like(domain);field[:2,:2,:2]=True
        initial=field.copy();parts,minima=coverage_parts(domain,.08)
        valid=np.ones((2,2,5),bool)
        status,report=grow_coverage(field,np.zeros_like(field),2,24,valid,[(0,0,0)],
            np.arange(valid.size).reshape(valid.shape).astype(float),.15,lambda:False,parts,minima)
        self.assertEqual(status,'target_reached');np.testing.assert_array_equal(cube_supported(field,2),field)
        for step in report['trace']:
            z,y,x=step['origin_zyx'];candidate=initial.copy();candidate[z:z+2,y:y+2,x:x+2]=True;new=candidate&~initial
            self.assertEqual(step['added_cells'],int(new.sum()))
            self.assertEqual(step['added_third_cells'],[int((new&p).sum()) for p in parts]);initial=candidate
        np.testing.assert_array_equal(initial,field)

    def test_contact_candidate_remains_available_after_other_growth(self):
        field=np.array([[[True,False],[False,False]]]);domain=np.ones_like(field);contact=np.zeros_like(field);contact[0,0,1]=True
        parts,_=coverage_parts(domain,.08)
        status,report=grow_coverage(field,contact,1,4,domain,[(0,0,0)],np.array([[[0,0],[1,2]]]),.25,lambda:False,parts,np.zeros(3,int))
        self.assertEqual(status,'target_reached');self.assertGreater(report['budget_rejections'],0)
        self.assertEqual(report['trace'][-1]['origin_zyx'],[0,0,1])

    def test_zero_delta_transit_expands_once(self):
        field=np.array([[[True,True,False]]]);domain=np.ones_like(field);parts,_=coverage_parts(domain,.08)
        status,report=grow_coverage(field,np.zeros_like(field),1,3,domain,[(0,0,0)],np.zeros(field.shape),.15,lambda:False,parts,np.zeros(3,int))
        self.assertEqual(status,'target_reached');self.assertEqual([t['added_cells'] for t in report['trace']],[0,1])
        self.assertEqual(len({tuple(p) for p in report['accepted_origins_zyx']}),2)

    def test_stall_and_timeout_preserve_partial_fields(self):
        for timeout in (False,True):
            field=np.array([[[True,False]]]);initial=field.copy();domain=np.ones_like(field);parts,_=coverage_parts(domain,.08)
            status,report=grow_coverage(field,np.array([[[False,True]]]),1,2,domain,[(0,0,0)],np.zeros(field.shape),.15,lambda:timeout,parts,np.zeros(3,int))
            self.assertEqual(status,'time_limit' if timeout else 'contact_budget_stalled');np.testing.assert_array_equal(field,initial)
            if not timeout:self.assertIsNone(report['trace'][0]['origin_zyx'])

    def test_mass_stop_does_not_silently_extend_for_unmet_coverage(self):
        field=np.array([[[True,False,False]]]);domain=np.ones_like(field);parts,_=coverage_parts(domain,.08)
        status,report=grow_coverage(field,np.zeros_like(field),1,2,domain,[(0,0,0)],np.zeros(field.shape),.15,lambda:False,parts,np.ones(3,int))
        self.assertEqual(status,'target_reached');self.assertEqual(int(field.sum()),2);self.assertFalse(report['coverage_minima_met'])


if __name__=='__main__':unittest.main()
