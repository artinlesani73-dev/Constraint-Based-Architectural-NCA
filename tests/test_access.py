import unittest
import numpy as np
import torch
from nca.access import component_strength,component_access,component_connectivity
from nca.losses import soft_reach
from nca.evaluation import endpoint_connectivity


class ComponentAccessTests(unittest.TestCase):
    def masks(self,shape,groups):
        out={}
        for name,cells in groups.items():
            a=np.zeros(shape,dtype=bool)
            for c in cells:a[c]=True
            out[name]=a
        return out

    def test_empty_fixed_corner_connected_remainder(self):
        p=torch.zeros(1,2,5);p[0,1,:]=1
        regions=self.masks(p.shape,{'A':[(0,0,0),(0,1,0)],'B':[(0,1,4)]})
        fixed=torch.zeros_like(p,dtype=torch.bool);fixed[0,0,0]=True
        self.assertEqual(float(soft_reach(p[None],fixed[None],8).max()),0.)
        self.assertEqual(float(component_strength(p,torch.ones_like(fixed),regions)[0]),1.)
        self.assertTrue(component_connectivity(p.numpy(),np.ones(p.shape,dtype=bool),regions)['all_connected'])

    def test_disconnected_origins_cannot_collect_destinations(self):
        p=torch.zeros(1,3,5);p[0,0,:]=1;p[0,2,:]=1
        regions=self.masks(p.shape,{'A':[(0,0,0),(0,2,0)],'B':[(0,0,4)],'C':[(0,2,4)]})
        legal=torch.ones_like(p,dtype=torch.bool)
        self.assertEqual(float(component_strength(p,legal,regions)[0]),0.)
        self.assertFalse(component_connectivity(p.numpy(),legal.numpy(),regions)['all_connected'])
        with self.assertRaisesRegex(ValueError,'disconnected'):
            endpoint_connectivity(p.numpy()>.5,regions,'A')

    def test_isolated_source_fragment_does_not_make_two_origins(self):
        p=torch.zeros(1,3,5);p[0,0,:]=1;p[0,2,0]=1
        regions=self.masks(p.shape,{'A':[(0,0,0),(0,2,0)],'B':[(0,0,4)]})
        self.assertEqual(float(component_strength(p,torch.ones_like(p,dtype=torch.bool),regions)[0]),1.)
        self.assertEqual(component_connectivity(p.numpy(),np.ones(p.shape,dtype=bool),regions)['components_touching_all_entrances'],1)

    def test_no_hop_cutoff_and_empty_endpoint(self):
        p=torch.ones(1,1,70);regions=self.masks(p.shape,{'A':[(0,0,0)],'B':[(0,0,69)]})
        legal=torch.ones_like(p,dtype=torch.bool);source=torch.tensor(regions['A'])[None]
        self.assertEqual(float(soft_reach(p[None],source,64)[0,0,0,-1]),0.)
        self.assertEqual(float(component_strength(p,legal,regions)[0]),1.)
        p[0,0,-1]=0
        self.assertEqual(float(component_strength(p,legal,regions)[0]),0.)

    def test_critical_voxel_gradient_matches_finite_difference(self):
        p=torch.tensor([[[.9,.8,.23,.7,.95]]],dtype=torch.float64,requires_grad=True)
        legal=torch.ones_like(p,dtype=torch.bool);regions=self.masks(p.shape,{'A':[(0,0,0)],'B':[(0,0,4)]})
        score,_=component_strength(p,legal,regions);score.backward()
        self.assertTrue(torch.equal(p.grad,torch.tensor([[[0.,0.,1.,0.,0.]]],dtype=torch.float64)))
        for i in range(5):
            a=p.detach().clone();b=a.clone();a[0,0,i]+=1e-6;b[0,0,i]-=1e-6
            numeric=(component_strength(a,legal,regions)[0]-component_strength(b,legal,regions)[0])/2e-6
            self.assertAlmostEqual(float(numeric),float(p.grad[0,0,i]),places=8)

    def test_random_threshold_equivalence_against_independent_bfs(self):
        rng=np.random.default_rng(17);shape=(2,3,4)
        regions=self.masks(shape,{'ground':[(0,0,0),(0,0,1)],'facade':[(1,2,3)],'other':[(0,2,0)]})
        for _ in range(20):
            p=rng.choice([0.,.2,.6,1.],size=shape);legal=rng.random(shape)>.15
            score,_=component_strength(torch.tensor(p),torch.tensor(legal),regions)
            for threshold in (0.,.1,.2,.5,.6,.9):
                self.assertEqual(float(score)>threshold,component_connectivity(p,legal,regions,threshold)['all_connected'])

    def test_invalid_masks_batch_and_infeasible_route(self):
        p=torch.ones(1,1,3,requires_grad=True);legal=torch.tensor([[[True,False,True]]])
        regions=self.masks(p.shape,{'A':[(0,0,0)],'B':[(0,0,2)]})
        score,info=component_strength(p,legal,regions)
        self.assertFalse(info['legal_route_exists']);score.backward();self.assertEqual(float(p.grad.sum()),0.)
        with self.assertRaises(ValueError):component_strength(p,legal,{'A':regions['A'],'B':regions['A']})
        loss,_=component_access(torch.stack([p.detach(),p.detach()]),torch.stack([legal,~legal]),[regions,regions])
        self.assertEqual(loss.tolist(),[1.,1.])


if __name__=='__main__':unittest.main()
