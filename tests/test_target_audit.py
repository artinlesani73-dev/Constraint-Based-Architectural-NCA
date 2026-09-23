import unittest
from dataclasses import replace
import torch
from nca.losses import LossSpec
from nca.target_audit import joint_bounds,target_candidates
from test_losses import fixture

class TargetAuditTests(unittest.TestCase):
    def test_facade_can_conflict_when_coverage_budget_passes(self):
        ctx=fixture();ctx=replace(ctx,facade=ctx.coverage.clone())
        b=joint_bounds(ctx,'site',LossSpec(max_mass_ratio=.05,max_facade_ratio=.15))
        self.assertTrue(b['budget_compatible'][0])
        self.assertFalse(b['joint_necessary_compatible'][0])
        self.assertAlmostEqual(b['facade_required_mass'][0].item(),5/.15)

    def test_no_facade_contact_adds_no_mass(self):
        ctx=fixture();b=joint_bounds(ctx,'site')
        self.assertEqual(b['mandatory_facade_voxels'][0],0)
        self.assertTrue(b['joint_necessary_compatible'][0])

    def test_no_dilution_capacity_is_detected(self):
        ctx=fixture();ctx=replace(ctx,facade=ctx.permitted.clone())
        b=joint_bounds(ctx,'envelope',LossSpec(max_mass_ratio=1.))
        self.assertEqual(b['minimum_possible_facade_fraction'][0],1.)
        self.assertFalse(b['joint_necessary_compatible'][0])

    def test_zero_facade_cap_handles_absent_and_present_contact(self):
        ctx=fixture();spec=LossSpec(max_facade_ratio=0.)
        self.assertTrue(joint_bounds(ctx,'site',spec)['joint_necessary_compatible'][0])
        ctx=replace(ctx,facade=ctx.coverage.clone())
        self.assertFalse(joint_bounds(ctx,'site',spec)['joint_necessary_compatible'][0])

    def test_candidate_geometry_is_nested_and_preserves_inputs(self):
        ctx=fixture();guide=ctx.coverage.clone()
        targets=target_candidates(guide,guide.float(),ctx.permitted)
        self.assertEqual(len(targets),6)
        self.assertTrue(torch.equal(guide,ctx.coverage))
        for small,large in [('guide','radius1'),('radius1','radius3'),('radius3','radius6')]:
            self.assertFalse((targets[small] & ~targets[large]).any())
        self.assertFalse((targets['radius6'] & ~ctx.permitted).any())
        with self.assertRaises(ValueError):target_candidates(~ctx.permitted,guide.float(),ctx.permitted)
