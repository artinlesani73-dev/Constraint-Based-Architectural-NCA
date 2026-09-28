import unittest,tempfile
from pathlib import Path
import torch,numpy as np
from nca.bulk_repair import ConnectedRepair,ConnectedSession,neighbors6,step_loss,cube_mean
from nca.repair_horizon import HorizonSession
from nca.recovery import tree_equal
from nca.experiments import digest

class ConnectedTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)
 def test_growth_invariants_and_target_free_path(self):
  model=ConnectedRepair();m=torch.zeros(1,1,7,7,7);m[0,0,3,3,3]=1;a=torch.ones_like(m,dtype=torch.bool);a[:,:,:,:,0]=False;f=torch.zeros(1,28,7,7,7)
  with torch.no_grad():model.last.bias[0]=10
  r=model.rollout(m,f,a,torch.Generator().manual_seed(1),4,capture=True);prior=m.bool()
  for born in r['births']:
   self.assertFalse((born&~neighbors6(prior)).any());self.assertFalse((born&~a).any());self.assertFalse((born&prior).any());prior=prior|born
  self.assertTrue(torch.equal(prior,r['field']));self.assertTrue((prior&m.bool()).any())
  self.assertFalse(neighbors6(m.bool())[0,0,4,4,3]);self.assertTrue(neighbors6(m.bool())[0,0,4,3,3])
  with self.assertRaises(ValueError):model.rollout(m*0,f,a,torch.Generator(),1)
  with self.assertRaises(ValueError):model.rollout(m,f,~a,torch.Generator(),1)
  with torch.no_grad():model.last.bias[0]=-10
  disconnected=m.clone();disconnected[0,0,1,1,1]=1
  r=model.rollout(disconnected,f,a,torch.Generator(),2);self.assertTrue(torch.equal(r['field'],disconnected.bool()))
 def test_loss_gradients_and_no_teacher_influence_on_births(self):
  x=torch.zeros(1,1,5,5,5,requires_grad=True);m=torch.zeros_like(x,dtype=torch.bool);e=m.clone();e[0,0,2,2,2]=True;t=torch.ones_like(x)
  loss,_,volume=step_loss(x,m,e,t,False);vg=torch.autograd.grad(volume,x,retain_graph=True)[0];self.assertGreater(float(vg.abs().sum()),0);loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
  model=ConnectedRepair();o=torch.zeros_like(x);o[0,0,2,2,2]=1;f=torch.zeros(1,28,5,5,5);a=torch.ones_like(m)
  r=model.rollout(o,f,a,torch.Generator().manual_seed(4),3,target=t);r['loss'].backward();self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0)
  q=model.rollout(o,f,a,torch.Generator().manual_seed(4),3);self.assertTrue(torch.equal(r['field'],q['field']))
  z,_,_=step_loss(x,m,m,t,False);self.assertEqual(float(z.detach()),0)
 def test_bulk_direction_and_intact_suppression(self):
  from nca.bulk_repair import bulk_completion
  from nca.connected_repair import step_loss as previous
  t=torch.zeros(1,1,7,7,7);t[:,:,1:6,1:6,1:6]=1
  m=t.bool();m[:,:,3,3,3]=False;e=torch.zeros_like(m);e[:,:,3,3,3]=True;e[:,:,0,0,0]=True
  x=torch.zeros_like(t,requires_grad=True)
  soft=m.float()+e.float()*x.sigmoid();v=bulk_completion(soft,e,t)
  grad=torch.autograd.grad(v,x)[0]
  self.assertLess(float(grad[0,0,3,3,3]),0)
  self.assertEqual(float(grad[0,0,0,0,0]),0)
  e=torch.zeros_like(m);e[:,:,0,0,0]=True
  a=step_loss(x,t.bool(),e,t,True)[0];b=previous(x,t.bool(),e,t,True)[0]
  ga=torch.autograd.grad(a,x,retain_graph=True)[0];gb=torch.autograd.grad(b,x)[0]
  self.assertGreater(float(ga.sum()),float(gb.sum()))
 def test_fixed_cube_mean_value_and_gradient_parity(self):
  g=torch.Generator().manual_seed(91)
  x=torch.rand(1,1,6,7,8,generator=g,requires_grad=True);y=x.detach().clone().requires_grad_(True)
  expected=torch.nn.functional.avg_pool3d(x,3,1);actual=cube_mean(y)
  torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
  weights=torch.rand(actual.shape,generator=g)
  (expected*weights).sum().backward();(actual*weights).sum().backward()
  torch.testing.assert_close(y.grad,x.grad,rtol=1e-6,atol=1e-7)
 def test_exact_recovery_and_semantic_rejection(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);t=np.zeros((6,)*3,np.float32);t[1:5,1:5,1:5]=1;d=t.copy();d[2:4,2:4,2:4]=0;c=np.zeros((7,6,6,6),np.float32);c[:2]=1;c[6]=.24
   p=root/'example.npz';np.savez_compressed(p,target=t,damaged=d,condition=c);rows=[dict(arrays=p.name,arrays_sha256=digest(p),split='train')]
   a=ConnectedSession(root,rows,{'test':'CGR2'});a.step();p=root/'saved.pt';a.save(p);trace,state=a.step();expected=a.payload()
   b=ConnectedSession(root,rows,{'test':'CGR2'});b.restore(p);actual,raw=b.step();self.assertEqual(trace,actual);np.testing.assert_array_equal(state,raw);self.assertTrue(tree_equal(expected,b.payload()))
   with self.assertRaises(ValueError):HorizonSession(root,rows,{'test':'CGR2'}).restore(p)
   old=HorizonSession(root,rows,{'test':'CGR2'});old.save(root/'old.pt')
   with self.assertRaises(ValueError):b.restore(root/'old.pt')
   from nca.connected_repair import ConnectedSession as PriorSession
   with self.assertRaises(ValueError):PriorSession(root,rows,{'test':'CGR2'}).restore(p)
if __name__=='__main__':unittest.main()
