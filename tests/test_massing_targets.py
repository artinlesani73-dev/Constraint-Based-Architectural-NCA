from dataclasses import replace
import unittest
import numpy as np
from nca.massing_targets import MassingTargetSpec,cube_supported,evaluate_targets,FAMILIES,context_interfaces_connectable
from nca.massing_cases import target_scenes,target_context,target_controls
from deploy.checkpoints import load_model_c


class MassingTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_model_c(device='cpu')[0]
        cls.scene=dict(target_scenes())['aligned']
        cls.fields,cls.domain,_=target_context(cls.scene,cls.config)
        cls.controls=target_controls(cls.scene,cls.domain)

    def evaluate(self,name,spec=MassingTargetSpec()):
        return evaluate_targets(self.controls[name],self.scene,self.fields,self.domain,spec)[0]

    def test_cube_opening_preserves_box_boundaries(self):
        a=np.zeros((8,9,10),bool);a[1:6,2:7,3:8]=True
        np.testing.assert_array_equal(cube_supported(a,3),a)
        np.testing.assert_array_equal(cube_supported(a,4),a)
        self.assertFalse(cube_supported(a,6).any())

    def test_thin_sheet_and_outside_padding_cannot_supply_bulk(self):
        a=np.zeros((7,7,7),bool);a[0:2,:,:]=True
        self.assertFalse(cube_supported(a,3).any())
        a=np.ones((3,3,3),bool)
        np.testing.assert_array_equal(cube_supported(a,3),a)
        self.assertFalse(cube_supported(a,4).any())

    def test_physical_aligned_refinement_even_width(self):
        a=np.zeros((10,10,10),bool);a[2:7,2:7,2:7]=True;a[7,4,4]=True
        coarse=cube_supported(a,3)
        fine=cube_supported(a.repeat(2,0).repeat(2,1).repeat(2,2),6)
        np.testing.assert_array_equal(fine,coarse.repeat(2,0).repeat(2,1).repeat(2,2))

    def test_operator_axis_permutation(self):
        a=self.controls['articulated_mass']
        np.testing.assert_array_equal(cube_supported(a,3).transpose(2,0,1),cube_supported(a.transpose(2,0,1),3))

    def test_context_can_use_later_source_component_but_not_union_of_disjoint_paths(self):
        a=np.zeros((9,9,9),bool);a[1:4,1:4,1:4]=True;a[4:7,4:7,1:8]=True
        source=np.zeros_like(a);source[3:5,3:5,1:3]=True
        target=np.zeros_like(a);target[5,5,7]=True
        self.assertTrue(context_interfaces_connectable(a,{'source':source,'target':target}))
        other=np.zeros_like(a);other[1,1,1]=True
        self.assertFalse(context_interfaces_connectable(a,{'source':source,'target':target,'third':other}))

    def test_compact_and_articulated_positive_examples(self):
        for name in ('compact_mass','articulated_mass'):
            r=self.evaluate(name)
            self.assertTrue(r['contract_pass'],(name,r['family_pass']))
            self.assertEqual(set(r['family_pass']),set(FAMILIES))

    def test_thin_neck_cannot_use_raw_connectivity(self):
        r=self.evaluate('thin_neck')
        self.assertTrue(all(r['raw_interface_hits'].values()))
        self.assertFalse(r['family_pass']['access'])
        self.assertGreater(r['unreached_bulk_voxels'],0)

    def test_satellite_is_not_hidden_by_connected_interface_component(self):
        r=self.evaluate('detached_satellite')
        self.assertTrue(all(r['raw_interface_hits'].values()))
        self.assertGreater(r['unreached_occupied_voxels'],0)
        self.assertFalse(r['contract_pass'])

    def test_expected_negative_reasons(self):
        for case,family in [('empty','coverage'),('thin_sheet','thickness'),('fragmented','access'),
                            ('unsupported','support'),('context_collision','legality'),('ground_intrusion','ground'),
                            ('outside_domain','spill'),('excessive_fill','sparsity')]:
            self.assertFalse(self.evaluate(case)['family_pass'][family],(case,family))

    def test_budget_does_not_drop_illegal_or_outside_cells(self):
        r=self.evaluate('outside_domain')
        self.assertEqual(r['occupied_voxels'],int(self.controls['outside_domain'].sum()))
        self.assertEqual(r['volume_fraction'],r['occupied_voxels']/int(self.domain.sum()))
        self.assertGreater(r['outside_domain_voxels'],0)

    def test_blocked_context_is_reported_not_silently_repaired(self):
        scene=dict(target_scenes())['blocked_gap'];fields,domain,_=target_context(scene,self.config)
        a=target_controls(scene,domain)['compact_mass']
        r,_=evaluate_targets(a,scene,fields,domain)
        self.assertFalse(r['context_necessary_checks_pass']);self.assertGreater(r['illegal_voxels'],0)
        self.assertFalse(r['contract_pass'])

    def test_coverage_regions_are_candidate_independent(self):
        a=self.evaluate('empty');b=self.evaluate('compact_mass')
        self.assertEqual([p['domain_cells'] for p in a['thirds']],[p['domain_cells'] for p in b['thirds']])
        self.assertEqual(sum(p['domain_cells'] for p in a['thirds']),int(self.domain.sum()))

    def test_minimum_cube_resolved_upwards_in_metres(self):
        r=self.evaluate('compact_mass',replace(MassingTargetSpec(),min_cube_m=2.41))
        self.assertEqual(r['resolved_cube_width_cells'],4);self.assertEqual(r['resolved_cube_width_m'],3.2)

    def test_empty_never_passes_despite_no_collisions(self):
        r=self.evaluate('empty');self.assertTrue(r['family_pass']['legality']);self.assertFalse(r['contract_pass'])

    def test_invalid_specs_and_masks_fail_loudly(self):
        for args in ({'min_cube_m':True},{'min_bulk_fraction':0},{'max_volume_fraction':float('nan')},{'min_volume_fraction':.6}):
            with self.assertRaises(ValueError):MassingTargetSpec(**args)
        with self.assertRaises(ValueError):cube_supported(np.ones((3,3,3),bool),False)
        with self.assertRaises(ValueError):evaluate_targets(self.controls['empty'],self.scene,self.fields,np.ones_like(self.domain))


if __name__=='__main__':unittest.main()
