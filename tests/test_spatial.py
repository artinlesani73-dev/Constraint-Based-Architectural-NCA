"""Independent geometric examples for the versioned single-plane surface gate."""
import unittest
from copy import deepcopy
from dataclasses import replace
import numpy as np
import torch
from nca.spatial import PlatformSpec,construct_platform,evaluate_platform,full_footprint,reachable,legacy_diagnostics
from scripts.run_spatial_prototype import cases
from deploy.studio import plan_scene
from deploy.checkpoints import load_model_c


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.scenes=dict(cases());cls.config,_,_=load_model_c(device='cpu')

    def test_hand_built_deck_dimensions_and_clearance(self):
        m=np.zeros((32,32,32),bool);m[7,14:17,8:24]=True;m[7,13:18,13:18]=True
        report,masks=evaluate_platform(self.scenes['aligned'],m)
        self.assertTrue(report['spatial_gate']);self.assertEqual(report['material_voxels'],58)
        self.assertAlmostEqual(report['floor_area_m2'],37.12)
        self.assertAlmostEqual(report['landing_area_m2'],16)
        self.assertEqual(int(masks['clearance'].sum()),58*3)
        self.assertFalse((masks['clearance']&m).any())

    def test_narrow_strips_fail_width_even_when_connected(self):
        m=np.zeros((32,32,32),bool);m[7,15,8:24]=True;m[7,13:18,13:18]=True
        r,_=evaluate_platform(self.scenes['aligned'],m)
        self.assertTrue(r['landing_clear']);self.assertFalse(r['approach_connected']);self.assertFalse(r['spatial_gate'])

    def test_missing_floor_breaks_access(self):
        m,_=construct_platform(self.scenes['aligned']);m[:,:,11]=False
        r,_=evaluate_platform(self.scenes['aligned'],m)
        self.assertFalse(r['approach_connected'])

    def test_headroom_includes_context_and_proposed_material(self):
        m,_=construct_platform(self.scenes['aligned'])
        blocked=m.copy();blocked[10,:,14:17]=True
        for scene,field in [(self.scenes['low-headroom'],m),(self.scenes['aligned'],blocked)]:
            r,_=evaluate_platform(scene,field)
            self.assertGreater(r['floor_cells_with_insufficient_clearance'],0)
            self.assertFalse(r['spatial_gate'])

    def test_exact_headroom_boundary(self):
        m,_=construct_platform(self.scenes['aligned']);m[11,14:17,8:24]=True
        r,_=evaluate_platform(self.scenes['aligned'],m)
        self.assertTrue(r['approach_connected']) # z=11 is above required [8,11)

    def test_collision_retained_not_clipped(self):
        m,status=construct_platform(self.scenes['blocked-span'])
        self.assertEqual(status['status'],'candidate');self.assertEqual(int(m.sum()),58)
        r,_=evaluate_platform(self.scenes['blocked-span'],m)
        self.assertGreater(r['material_context_collision_voxels'],0);self.assertFalse(r['spatial_gate'])

    def test_split_levels_are_unsupported_not_globally_infeasible(self):
        m,s=construct_platform(self.scenes['split-levels'])
        self.assertEqual(s['status'],'unsupported_layout');self.assertFalse(m.any())
        r,_=evaluate_platform(self.scenes['split-levels'],m)
        self.assertFalse(r['layout_supported']);self.assertFalse(r['spatial_gate'])

    def test_empty_and_floating_floor_fail(self):
        for field in (np.zeros((32,32,32),bool),construct_platform(self.scenes['aligned'])[0]):
            field[:,:,:9]=False;field[:,:,23:]=False
            r,_=evaluate_platform(self.scenes['aligned'],field)
            self.assertFalse(r['spatial_gate'])

    def test_footprint_never_wraps_and_diagonal_is_disconnected(self):
        self.assertEqual(int(full_footprint(np.ones((5,5),bool),3).sum()),9)
        mask=np.eye(5,dtype=bool);seeds=np.zeros_like(mask);seeds[0,0]=True
        self.assertEqual(int(reachable(mask,seeds).sum()),1)

    def test_brief_bounds_and_field_validation(self):
        for spec in (replace(PlatformSpec(),width_cells=2),replace(PlatformSpec(),headroom_cells=25),replace(PlatformSpec(),width_cells=True)):
            with self.assertRaises(ValueError):construct_platform(self.scenes['aligned'],spec)
        with self.assertRaises(ValueError):evaluate_platform(self.scenes['aligned'],np.zeros((32,32,32)))

    def test_material_access_and_floor_access_are_distinct(self):
        scene=self.scenes['aligned'];m,_=construct_platform(scene)
        new,_=evaluate_platform(scene,m);old=legacy_diagnostics(scene,m,self.config)
        self.assertTrue(new['spatial_gate']);self.assertFalse(old['connectivity']['all_connected'])

    def test_legacy_audit_matches_existing_studio(self):
        scene=self.scenes['aligned'];w1=plan_scene(scene,self.config)
        m=np.zeros((32,32,32),bool)
        for z,y,x in w1['material_zyx']:m[z,y,x]=True
        audit=legacy_diagnostics(scene,m,self.config)
        for key in ('families','material_ratio','in_budget','envelope_voxels','context_valid','connectivity','joint_budget_connectivity','legality'):
            self.assertEqual(audit[key],w1['diagnostics'][key],key)


if __name__=='__main__':unittest.main()
