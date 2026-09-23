import unittest
import numpy as np
import torch
from nca.access import component_strength, component_connectivity
from nca.raw_access import raw_component_strength, raw_component_access


class RawAccessTests(unittest.TestCase):
    def fixture(self, values):
        x=torch.tensor(values,dtype=torch.float64).reshape(1,1,-1).requires_grad_()
        a=np.zeros(x.shape,dtype=bool);b=a.copy();a[0,0,0]=True;b[0,0,-1]=True
        return x,torch.ones_like(x,dtype=torch.bool),{'a':a,'b':b}

    def test_negative_gradient_and_finite_difference(self):
        x,m,e=self.fixture([.8,-.3,.9]);loss,_=raw_component_access(x[None],m[None],[e])
        g,=torch.autograd.grad(loss.sum(),x)
        np.testing.assert_array_equal(g.numpy().ravel(),[0,-1,0])
        eps=1e-6;plus=x.detach().clone();minus=plus.clone()
        plus[0,0,1]+=eps;minus[0,0,1]-=eps
        a=raw_component_access(plus[None],m[None],[e])[0]
        b=raw_component_access(minus[None],m[None],[e])[0]
        self.assertAlmostEqual(float((a-b)/(2*eps)),-1,places=8)
        old,_=component_strength(x.clamp(0,1),m,e)
        self.assertEqual(float(old.detach()),0.)
        self.assertEqual(float(torch.autograd.grad(old,x)[0].norm()),0.)

    def test_saturation_and_zero(self):
        for values,expected,norm in [([2,3,4],0,0),([.8,0,.9],1,1)]:
            x,m,e=self.fixture(values);l,_=raw_component_access(x[None],m[None],[e])
            self.assertEqual(float(l.detach()),expected)
            self.assertEqual(float(torch.autograd.grad(l.sum(),x)[0].norm()),norm)

    def test_impossible_graph_and_mask(self):
        x,m,e=self.fixture([.8,-.3,.9]);m[0,0,1]=False
        l,d=raw_component_access(x[None],m[None],[e]);self.assertFalse(d[0]['legal_route_exists'])
        self.assertIsNone(d[0]['critical_zyx']);self.assertEqual(float(l.detach()),1)
        self.assertEqual(float(torch.autograd.grad(l.sum(),x)[0].norm()),0)

    def test_impossible_graph_extreme_finite_values_do_not_overflow_zero(self):
        x,m,e=self.fixture([torch.finfo(torch.float64).max]*3);m[0,0,1]=False
        l,d=raw_component_access(x[None],m[None],[e])
        self.assertEqual(float(l.detach()),1);self.assertFalse(d[0]['legal_route_exists'])
        self.assertTrue(torch.equal(torch.autograd.grad(l.sum(),x)[0],torch.zeros_like(x)))

    def test_multi_entrance_and_stable_ties(self):
        x,m,e=self.fixture([0,0,0]);c=np.zeros(x.shape,dtype=bool);c[0,0,1]=True;e['c']=c
        b,d=raw_component_strength(x,m,e);self.assertEqual(float(b.detach()),0)
        self.assertEqual(d['critical_zyx'],[0,0,2])
        self.assertEqual(d,raw_component_strength(x,m,e)[1])

    def test_random_clamp_identity_and_binary_bfs(self):
        rng=np.random.default_rng(123)
        for _ in range(30):
            x=torch.tensor(rng.uniform(-1,2,(2,3,4)),dtype=torch.float64)
            m=torch.tensor(rng.random(x.shape)>.15);a=np.zeros(x.shape,dtype=bool);b=a.copy()
            a[0,0,0]=True;b[-1,-1,-1]=True;e={'a':a,'b':b}
            raw,rd=raw_component_strength(x,m,e);p=x.clamp(0,1)*m
            old,od=component_strength(p,m,e);self.assertEqual(rd['legal_route_exists'],od['legal_route_exists'])
            if rd['legal_route_exists']:self.assertEqual(float(raw.clamp(0,1)),float(old))
            connected=component_connectivity(p.numpy(),m.numpy(),e)['all_connected']
            self.assertEqual(connected,rd['legal_route_exists'] and float(raw)>.5)

    def test_invalid_input(self):
        x,m,e=self.fixture([0,0,0])
        for bad in (x.int(),x*float('nan'),x*float('inf')):
            with self.assertRaises(ValueError):raw_component_access(bad[None],m[None],[e])
        e['b']=e['a'].copy()
        with self.assertRaises(ValueError):raw_component_access(x[None],m[None],[e])

    def test_untied_gradcheck(self):
        x,m,e=self.fixture([.9,-.2,.7])
        self.assertTrue(torch.autograd.gradcheck(lambda t:raw_component_access(t[None],m[None],[e])[0],(x,)))

    def test_instrumented_real_forward_gradients_and_rng(self):
        from deploy.checkpoints import load_model_c
        from deploy.model_utils import UrbanPavilionNCA,UrbanSceneGenerator
        from nca.contract import load_reference_set,to_generator_params
        from nca.legal_corridor import compute_legal_corridor_v1
        from nca.interventions import experimental_rollout
        from nca.access_trace import traced_rollout
        torch.set_num_threads(2)
        cfg,weights,_=load_model_c();model=UrbanPavilionNCA(cfg);model.load_state_dict(weights)
        scene=load_reference_set()['ref-01-ground-pair']
        seed,_=UrbanSceneGenerator(cfg).generate(to_generator_params(scene),device='cpu')
        scaffold=compute_legal_corridor_v1(seed,cfg,[scene])['target'];out=[];grad=[];rng=[]
        for fn in (experimental_rollout,traced_rollout):
            gen=torch.Generator().manual_seed(2)
            q=fn(model,seed,scaffold,'hard_preclamp',3,gen);out.append(q);rng.append(gen.get_state())
            grad.append(torch.autograd.grad(q['state'].sum()+q['raw_material'].sum(),tuple(model.parameters())))
        for k in ('state','raw_material'):self.assertTrue(torch.equal(out[0][k],out[1][k]))
        self.assertTrue(torch.equal(*rng))
        for a,b in zip(*grad):self.assertTrue(torch.equal(a,b))
        self.assertEqual(len(out[1]['trajectory']),3)
