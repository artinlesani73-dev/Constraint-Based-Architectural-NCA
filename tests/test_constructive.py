from dataclasses import replace
import unittest
import torch
from nca.constructive import build_witness
from nca.losses import LossSpec
from test_losses import fixture

class ConstructiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_deterministic_budgeted_growth_is_legal_connected_and_nonmutating(self):
        ctx=fixture();before=ctx.coverage.clone();allow=torch.zeros_like(before)
        a=build_witness(ctx,allow);b=build_witness(ctx,allow)
        self.assertEqual(a['status'],'constructed');self.assertTrue(torch.equal(a['material'],b['material']))
        self.assertTrue(torch.equal(ctx.coverage,before));self.assertFalse((a['material']&~ctx.permitted).any())
        current=before.clone()
        for z,y,x in a['added_cells']:
            neighbors=[(z+dz,y+dy,x+dx) for dz,dy,dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
            self.assertTrue(any(all(0<=v<7 for v in xyz) and bool(current[(0,)+xyz]) for xyz in neighbors));current[0,z,y,x]=True
        self.assertEqual(int(current.sum()),a['target_mass'])

    def test_incompatible_route_not_reclassified(self):
        ctx=fixture();ctx=replace(ctx,route_feasible=torch.tensor([False]))
        result=build_witness(ctx,torch.zeros_like(ctx.coverage));self.assertEqual(result['status'],'incompatible')
        self.assertEqual(result['added_cells'],[])

    def test_fractional_budget_can_lack_binary_witness(self):
        ctx=fixture();spec=LossSpec(min_mass_ratio=.0301,max_mass_ratio=.0302)
        result=build_witness(ctx,torch.zeros_like(ctx.coverage),spec)
        self.assertEqual(result['status'],'no_binary_zero_core_target')
