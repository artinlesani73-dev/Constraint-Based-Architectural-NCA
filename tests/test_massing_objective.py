import unittest
import numpy as np
import torch
from nca.massing_objective import soft_bulk, component_excess, support_strength, make_context, massing_residuals, batched_residuals
from nca.massing_targets import cube_supported, evaluate_targets, FAMILIES
from nca.massing_cases import target_scenes,target_context,target_controls
from nca.evaluation import geometric_support
from deploy.checkpoints import load_model_c


class MassingObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.scene=dict(target_scenes())['aligned']
        cls.fields,cls.domain,_=target_context(cls.scene,load_model_c(device='cpu')[0])
        cls.context=make_context(cls.scene,cls.fields,cls.domain)
        cls.controls=target_controls(cls.scene,cls.domain)

    def test_soft_opening_matches_binary_even_and_odd(self):
        a=np.zeros((8,9,10),bool);a[1:7,2:8,3:9]=True;a[0,0,0]=True
        for w in (1,2,3,4,9):
            actual=soft_bulk(torch.tensor(a,dtype=torch.double),w).numpy()
            np.testing.assert_array_equal(actual,cube_supported(a,w))

    def test_soft_opening_gradcheck_untied(self):
        torch.manual_seed(3)
        p=(.1+.8*torch.rand((4,4,4),dtype=torch.double)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda q:soft_bulk(q,2),p,fast_mode=True))

    def test_component_excess_counts_detached_binary_parts(self):
        a=np.zeros((5,5,5),bool);a[0,0,0]=True;a[2,2,2]=True;a[4,4,4]=True
        p=torch.tensor(a,dtype=torch.double)
        self.assertEqual(component_excess(p,np.ones_like(a)).item(),2)
        self.assertEqual(component_excess(p,a).item(),2)
        self.assertEqual(component_excess(p*0,np.ones_like(a)).item(),0)
        self.assertEqual(component_excess(torch.ones_like(p),np.ones_like(a)).item(),0)

    def test_component_excess_gradcheck_untied(self):
        torch.manual_seed(4)
        p=(.1+.8*torch.rand((4,4,4),dtype=torch.double)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda q:component_excess(q,np.ones(q.shape,bool)),p,fast_mode=True))

    def test_support_matches_independent_binary_flood(self):
        a=np.zeros((6,6,6),bool);a[1:5,1,1]=True;a[4,4,4]=True
        fixed=np.zeros_like(a);fixed[0,1,1]=True
        p=torch.tensor(a,dtype=torch.double);s=support_strength(p,fixed)
        self.assertEqual(int(torch.relu(p-s).sum()),geometric_support(a,fixed)['unsupported_voxels'])
        np.testing.assert_array_equal(support_strength(p,np.zeros_like(a)),np.zeros_like(a))

    def test_support_gradcheck_untied(self):
        torch.manual_seed(5)
        p=(.1+.8*torch.rand((4,4,4),dtype=torch.double)).requires_grad_();fixed=np.zeros(p.shape,bool);fixed[0,0,0]=True
        self.assertTrue(torch.autograd.gradcheck(lambda q:support_strength(q,fixed),p,fast_mode=True))

    def test_binary_family_agreement_on_positive_and_negative_controls(self):
        for case in ('compact_mass','thin_neck','detached_satellite','empty','ground_intrusion','outside_domain'):
            field=self.controls[case];terms,_=massing_residuals(torch.tensor(field,dtype=torch.double),self.context)
            binary,_=evaluate_targets(field,self.scene,self.fields,self.domain)
            self.assertEqual(set(terms),set(FAMILIES))
            for key,value in terms.items():self.assertEqual(value.item()<=1e-10,binary['family_pass'][key],(case,key,value.item()))

    def test_batch_values_and_gradients_are_scene_independent(self):
        p=torch.stack([torch.tensor(self.controls[k],dtype=torch.double) for k in ('compact_mass','thin_neck')]).requires_grad_()
        batch=batched_residuals(p,[self.context,self.context])
        single=massing_residuals(p[0],self.context)[0]
        for key in batch:torch.testing.assert_close(batch[key][0],single[key])
        gradient=torch.autograd.grad(sum(v[0] for v in batch.values()),p)[0]
        self.assertEqual(gradient[1].abs().sum().item(),0)

    def test_context_masks_do_not_alias_inputs(self):
        c=make_context(self.scene,self.fields,self.domain)
        self.assertFalse(np.shares_memory(c.domain,self.domain))
        for key in c.masks:self.assertFalse(np.shares_memory(c.masks[key],self.fields[key]))

    def test_validation_and_probability_bounds(self):
        for value in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):soft_bulk(torch.full((3,3,3),value),2)
        with self.assertRaises(ValueError):soft_bulk(torch.ones((3,3,3)),True)
        with self.assertRaises(ValueError):component_excess(torch.ones((3,3,3)),np.ones((3,3,3),float))
        with self.assertRaises(ValueError):batched_residuals(torch.zeros((1,3,3,3)),[])


if __name__=='__main__':unittest.main()
