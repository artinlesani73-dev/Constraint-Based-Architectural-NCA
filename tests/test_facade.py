import unittest
from dataclasses import replace
import torch
from nca.losses import LossSpec,loss_terms
from nca.facade import endpoint_allowance,facade_term,facade_bounds
from nca.contract import load_reference_set
from test_losses import fixture

class FacadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_empty_allowance_equals_original(self):
        ctx=fixture();p=torch.rand(ctx.permitted.shape,generator=torch.Generator().manual_seed(6))
        old=loss_terms(p,ctx,LossSpec(reach_hops=8))['terms']['facade']
        self.assertTrue(torch.equal(old,facade_term(p,ctx,torch.zeros_like(ctx.permitted))))

    def test_allowed_patch_and_unwanted_blanket_are_distinguished(self):
        ctx=fixture();facade=torch.zeros_like(ctx.facade);facade[:,3,:,:]=True
        ctx=replace(ctx,facade=facade);allowed=torch.zeros_like(facade);allowed[:,3,3,1]=True
        self.assertEqual(facade_term(allowed.float(),ctx,allowed)[0],0)
        self.assertGreater(facade_term(facade.float(),ctx,allowed)[0],.8)
        with self.assertRaises(ValueError):facade_term(allowed.float(),ctx,~facade)

    def test_numerical_derivative_away_from_hinge(self):
        ctx=fixture();ctx=replace(ctx,facade=ctx.permitted.clone())
        allowed=torch.zeros_like(ctx.facade);allowed[:,3,3,1]=True
        p=torch.full(ctx.facade.shape,.4,dtype=torch.float64,requires_grad=True)
        direction=torch.randn(p.shape,generator=torch.Generator().manual_seed(3),dtype=p.dtype);direction/=direction.norm()
        grad,=torch.autograd.grad(facade_term(p,ctx,allowed).sum(),p)
        h=1e-6;fd=(facade_term(p.detach()+h*direction,ctx,allowed)-facade_term(p.detach()-h*direction,ctx,allowed))/(2*h)
        self.assertAlmostEqual(float((grad*direction).sum()),float(fd[0]),delta=1e-8)

    def test_batch_preserves_individual_terms(self):
        ctx=fixture(2);allow=torch.zeros_like(ctx.facade);allow[:,1,2,2]=True
        p=torch.rand(ctx.facade.shape,generator=torch.Generator().manual_seed(2))
        both=facade_term(p,ctx,allow)
        for i in range(2):self.assertTrue(torch.equal(both[i:i+1],facade_term(p[i:i+1],fixture(),allow[i:i+1])))

    def test_annotation_is_scene_derived_and_has_no_ground_exemption(self):
        scenes=load_reference_set();scene=scenes['ref-01-ground-pair']
        allowed,annotation=endpoint_allowance(scene,torch.ones((1,32,32,32),dtype=torch.bool))
        self.assertFalse(allowed.any());self.assertEqual(annotation['patches'],[])
        scene=scenes['ref-02-facade-pair-and-ground'];permitted=torch.ones_like(allowed)
        a,first=endpoint_allowance(scene,permitted);b,second=endpoint_allowance(scene,permitted)
        self.assertTrue(torch.equal(a,b));self.assertEqual(first,second);self.assertTrue(a.any())
        self.assertTrue(all(p['count']<=8 for p in first['patches']))
        blocked,_=endpoint_allowance(scene,torch.zeros_like(permitted));self.assertFalse(blocked.any())

    def test_bound_changes_only_charged_facade_count(self):
        ctx=fixture();ctx=replace(ctx,facade=ctx.coverage.clone());spec=LossSpec(max_mass_ratio=.05)
        old=facade_bounds(ctx,'site',torch.zeros_like(ctx.facade),spec)
        new=facade_bounds(ctx,'site',ctx.coverage,spec)
        self.assertFalse(old['joint_necessary_compatible'][0]);self.assertTrue(new['joint_necessary_compatible'][0])
        for key in ['denominator_voxels','minimum_mass','maximum_mass','envelope_capacity','coverage_voxels']:
            self.assertTrue(torch.equal(old[key],new[key]))
