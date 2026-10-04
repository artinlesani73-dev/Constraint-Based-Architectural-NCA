from pathlib import Path
import sys,unittest,tempfile,importlib.util,time,json
ROOT=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
import torch,numpy as np
from reversible_repair import ConnectedRepair,ConnectedSession,transition,step_loss
from nca.connected_repair import ConnectedSession as Prior
from nca.experiments import digest
from nca.recovery import tree_equal
spec=importlib.util.spec_from_file_location('reference','C:/Users/artin/Documents/Codex/outputs/Reversible-Repair-Design-2026-10-03/reference_transition.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
class Checks(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)
 def test_reference_parity_and_delete_rebirth(self):
  rng=np.random.default_rng(77);o=np.zeros((7,)*3,bool);o[3,3,3]=1;m=o.copy();a=np.ones_like(o);a[0]=False
  for _ in range(12):
   f=rng.random(o.shape)<.5;q=rng.random(o.shape).astype(np.float32)
   expected=ref.transition(o,m,a,f,q);result=transition(*[torch.from_numpy(x)[None,None] for x in [o,m,a,f,q]])
   for key,value in zip(['field','active','candidate','direct_removed','projection_removed','births'],result):np.testing.assert_array_equal(value.numpy()[0,0],expected[key])
   m=expected['field'];self.assertFalse((o&~m).any())
  o=torch.zeros(1,1,5,5,5,dtype=torch.bool);o[:,:,2,2,2]=1;a=torch.ones_like(o)
  born=transition(o,o,a,a,torch.ones_like(o,dtype=torch.float32))[0]
  deleted=transition(o,born,a,a,torch.zeros_like(o,dtype=torch.float32))[0];self.assertTrue(torch.equal(deleted,o))
  reborn=transition(o,deleted,a,a,torch.ones_like(o,dtype=torch.float32))[0];self.assertTrue(torch.equal(born,reborn))
 def test_loss_direction_and_target_free_rollout(self):
  x=torch.zeros(1,1,7,7,7,requires_grad=True);m=torch.zeros_like(x,dtype=torch.bool);t=torch.zeros_like(x);a=m.clone()
  t[:,:,2:5,2:5,2:5]=1;m[:,:,3,3,3]=1;m[:,:,1,3,3]=1
  for point in [(3,3,3),(3,3,4),(1,3,3)]:a[(0,0,*point)]=True
  loss,_,_=step_loss(x,m,a,t,False);loss.backward()
  self.assertLess(float(x.grad[0,0,3,3,3]),0);self.assertLess(float(x.grad[0,0,3,3,4]),0);self.assertGreater(float(x.grad[0,0,1,3,3]),0)
  z=step_loss(x,m,a&False,t,False)[0];self.assertEqual(float(z.detach()),0)
  model=ConnectedRepair();o=torch.zeros_like(t);o[:,:,3,3,3]=1;legal=torch.ones_like(a);features=torch.zeros(1,28,7,7,7)
  r=model.rollout(o,features,legal,torch.Generator().manual_seed(2),4,target=t,capture=True)
  q=model.rollout(o,features,legal,torch.Generator().manual_seed(2),4,capture=True)
  self.assertTrue(torch.equal(r['state'],q['state']));r['loss'].backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
 def test_exact_training_recovery_and_identity(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);t=np.zeros((7,)*3,np.float32);t[1:6,1:6,1:6]=1;d=t.copy();d[2:5,2:5,2:5]=0;c=np.zeros((7,7,7,7),np.float32);c[:2]=1
   p=root/'example.npz';np.savez_compressed(p,target=t,damaged=d,condition=c);rows=[dict(arrays=p.name,arrays_sha256=digest(p),split='train')]
   a=ConnectedSession(root,rows,{'test':'RGR1'});a.step();checkpoint=root/'saved.pt';a.save(checkpoint);trace,state=a.step();expected=a.payload()
   b=ConnectedSession(root,rows,{'test':'RGR1'});b.restore(checkpoint);actual,raw=b.step();self.assertEqual(trace,actual);np.testing.assert_array_equal(state,raw);self.assertTrue(tree_equal(expected,b.payload()))
   with self.assertRaises(ValueError):Prior(root,rows,{'test':'RGR1'}).restore(checkpoint)
if __name__=='__main__':
 t=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 (OUT/'checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,seconds=time.monotonic()-t,scope='Synthetic CPU correctness and optimizer-step recovery. Delete/rebirth transition tested separately; no GPU evidence.'),indent=2),encoding='utf-8')
 raise SystemExit(not result.wasSuccessful())
