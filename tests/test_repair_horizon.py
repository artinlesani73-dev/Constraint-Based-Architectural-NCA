import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
from nca.repair_preservation import preservation_loss,PreservationSession
from nca.repair_horizon import HorizonSession
from nca.repair_portable import PortableSession
from nca.recovery import tree_equal
from nca.experiments import digest

class HorizonTests(unittest.TestCase):
 def test_loss_gradients_and_exterior_mask(self):
  x=torch.zeros(1,1,1,1,3,requires_grad=True);t=torch.tensor([1.,0.,0.]).reshape_as(x);a=torch.tensor([True,True,False]).reshape_as(x)
  damaged=torch.zeros_like(t);loss,parts=preservation_loss(x,t,a,damaged);loss.backward()
  self.assertAlmostEqual(loss.item(),1.5*np.log(2),places=6)
  torch.testing.assert_close(x.grad.flatten(),torch.tensor([-.25,.5,0.]));self.assertFalse(parts['intact'])
  x.grad=None;loss,parts=preservation_loss(x,t,a,t);loss.backward()
  self.assertAlmostEqual(loss.item(),2.5*np.log(2),places=6)
  torch.testing.assert_close(x.grad.flatten(),torch.tensor([-.5,.75,0.]));self.assertTrue(parts['intact'])
 def test_recovery_and_legacy_identity_rejection(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);t=np.zeros((6,)*3,np.float32);t[1:5,1:5,1:5]=1;d=t.copy();d[2:4,2:4,2:4]=0;c=np.zeros((7,6,6,6),np.float32);c[:2]=1;c[6]=.24
   p=root/'example.npz';np.savez_compressed(p,target=t,damaged=d,condition=c);rows=[{'arrays':p.name,'arrays_sha256':digest(p),'split':'train'}]
   a=HorizonSession(root,rows,{'test':'recovery'});calls=[];original=a.model.forward
   def counted(*args,**kwargs):
    calls.append(1);return original(*args,**kwargs)
   a.model.forward=counted;a.step();self.assertEqual(len(calls),32);a.model.forward=original;checkpoint=root/'checkpoint.pt';a.save(checkpoint);trace,state=a.step();expected=a.payload()
   b=HorizonSession(root,rows,{'test':'recovery'});b.restore(checkpoint);actual,raw=b.step()
   self.assertEqual(trace,actual);np.testing.assert_array_equal(state,raw);self.assertTrue(tree_equal(expected,b.payload()))
   with self.assertRaises(ValueError):PreservationSession(root,rows,{'test':'recovery'}).restore(checkpoint)
   with self.assertRaises(ValueError):PortableSession(root,rows,{'test':'recovery'}).restore(checkpoint)
if __name__=='__main__':unittest.main()
