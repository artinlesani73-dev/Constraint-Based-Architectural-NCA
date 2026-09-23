import unittest
import torch
from nca.direct import DirectField
from nca.interventions import guidance_loss

class DirectFieldTests(unittest.TestCase):
    def test_hard_projection_and_true_clamp_derivative(self):
        raw=torch.tensor([[[[-.3,.2,1.4,.6]]]])
        allowed=torch.tensor([[[[True,True,True,False]]]])
        field=DirectField(raw,allowed);p=field()
        self.assertTrue(torch.equal(p,torch.tensor([[[[0.,.2,1.,0.]]]])))
        p.sum().backward()
        self.assertTrue(torch.equal(field.raw.grad,torch.tensor([[[[0.,1.,0.,0.]]]])))

    def test_preclamp_coverage_crosses_negative_raw_without_softening_material(self):
        field=DirectField(torch.full((1,1,1,1),-.2),torch.ones((1,1,1,1),dtype=torch.bool))
        loss=guidance_loss(field()[:,None],field.raw,field.permitted,{'ch_structure':0},'hard_preclamp').sum()
        loss.backward()
        self.assertEqual(float(field().detach()),0.)
        self.assertEqual(float(field.raw.grad),-1.)

    def test_initialization_is_independent_and_masks_are_validated(self):
        raw=torch.full((1,2,2,2),.15);allowed=torch.ones_like(raw,dtype=torch.bool)
        field=DirectField(raw,allowed);raw.zero_();allowed.zero_()
        self.assertTrue(torch.all(field()==.15))
        with self.assertRaises(ValueError):DirectField(raw,allowed.float())
        with self.assertRaises(ValueError):DirectField(torch.full_like(raw,float('nan')),allowed)

if __name__=='__main__':unittest.main()
