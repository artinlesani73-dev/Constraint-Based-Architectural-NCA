import unittest
import numpy as np
from nca.budget_mass_generator import cube_delta, grow_budgeted, generate_budget_mass
from nca.contact_mass_generator import generate_contact_mass
from nca.mass_generation_cases import generation_scenes
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets
from deploy.checkpoints import load_model_c


class BudgetGrowthTests(unittest.TestCase):
    def test_overlap_counts_unique_new_contact(self):
        field=np.zeros((2,2,3),bool);field[:,:,:2]=True
        contact=np.zeros_like(field);contact[:,0,:]=True
        self.assertEqual(cube_delta(field,contact,(0,0,1),2),(4,2))
        self.assertEqual(cube_delta(field,contact,(0,0,0),2),(0,0))

    def test_deferred_candidate_becomes_feasible_after_growth(self):
        field=np.zeros((1,2,2),bool);field[0,0,0]=True
        contact=np.zeros_like(field);contact[0,0,1]=True
        valid=np.ones_like(field);cost=np.array([[[0,0],[1,2]]],float)
        status,report=grow_budgeted(field,contact,1,4,valid,[(0,0,0)],cost,.25,lambda:False)
        self.assertEqual(status,'target_reached');self.assertEqual(int(field.sum()),4)
        decisions=[a['accepted'] for a in report['trace'] if a['origin_zyx']==[0,0,1]]
        self.assertIn(False,decisions);self.assertTrue(decisions[-1])
        for a in report['trace']:
            if a['accepted']:
                self.assertLessEqual((a['contact_before']+a['added_contact_cells'])/(a['mass_before']+a['added_cells']),.25)

    def test_stalled_growth_returns_once_without_discarding_evidence(self):
        field=np.array([[[True,False]]]);contact=np.array([[[False,True]]])
        status,report=grow_budgeted(field,contact,1,2,np.ones_like(field),[(0,0,0)],np.zeros(field.shape),.15,lambda:False)
        self.assertEqual(status,'contact_budget_stalled');self.assertEqual(len(report['trace']),1)
        self.assertEqual(report['deferred_origins_zyx'],[[0,0,1]])
        np.testing.assert_array_equal(field,[[[True,False]]])

    def test_zero_delta_origin_expands_once_without_infinite_retry(self):
        field=np.array([[[True,True,False]]]);contact=np.zeros_like(field)
        status,report=grow_budgeted(field,contact,1,3,np.ones_like(field),[(0,0,0)],np.zeros(field.shape),.15,lambda:False)
        self.assertEqual(status,'target_reached');self.assertEqual(len(report['trace']),2)
        self.assertEqual(report['trace'][0]['added_cells'],0)
        self.assertEqual(report['final_voxels'],3)

    def test_invalid_completed_route_is_retained(self):
        field=np.array([[[True,False]]]);contact=np.array([[[True,False]]]);initial=field.copy()
        status,report=grow_budgeted(field,contact,1,2,np.ones_like(field),[(0,0,0)],np.zeros(field.shape),.15,lambda:False)
        self.assertEqual(status,'route_contact_budget_exceeded');self.assertEqual(report['trace'],[])
        np.testing.assert_array_equal(field,initial)

    def test_time_limit_retains_partial_state(self):
        field=np.array([[[True,False]]]);contact=np.zeros_like(field)
        status,report=grow_budgeted(field,contact,1,2,np.ones_like(field),[(0,0,0)],np.zeros(field.shape),.15,lambda:True)
        self.assertEqual(status,'time_limit');self.assertEqual(report['final_voxels'],1)

    def test_real_route_unchanged_and_growth_trace_reconstructs_field(self):
        scene=dict(generation_scenes())['partial_obstruction']
        fields,domain,_=target_context(scene,load_model_c(device='cpu')[0])
        original_domain=domain.copy()
        _,before_route,_=generate_contact_mass(scene,fields,domain,2)
        field,route,report=generate_budget_mass(scene,fields,domain,2)
        np.testing.assert_array_equal(route,before_route);np.testing.assert_array_equal(domain,original_domain)
        replay=route.copy()
        for step in report['growth']['trace']:
            if step['accepted']:
                z,y,x=step['origin_zyx'];replay[z:z+3,y:y+3,x:x+3]=True
                self.assertEqual(int(replay.sum()),step['mass_before']+step['added_cells'])
                self.assertLessEqual((step['contact_before']+step['added_contact_cells'])/int(replay.sum()),.15+1e-10)
        np.testing.assert_array_equal(replay,field)
        binary,_=evaluate_targets(field,scene,fields,domain)
        self.assertTrue(binary['family_pass']['facade'])
        self.assertFalse((field&~domain).any())


if __name__=='__main__':unittest.main()
