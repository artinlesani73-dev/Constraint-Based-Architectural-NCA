import unittest,tempfile
from pathlib import Path
import numpy as np,torch
from nca.curriculum_repair import ConnectedSession
from nca.connected_repair import ConnectedSession as Prior
from nca.bulk_repair import ConnectedSession as Bulk
from nca.experiments import digest
from nca.recovery import tree_equal

class CurriculumTests(unittest.TestCase):
 def test_original_parity_augmented_recovery_and_evaluation(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);t=np.zeros((8,)*3,np.float32);t[1:7,1:7,1:7]=1;d=t.copy();d[2:6,2:6,2:6]=0;c=np.zeros((7,8,8,8),np.float32);c[:2]=1;c[6]=.24
   p=root/'example.npz';np.savez_compressed(p,target=t,damaged=d,condition=c);rows=[dict(arrays=p.name,arrays_sha256=digest(p),split='train')]
   prior=Prior(root,rows,{'test':'CGR3'});pt,ps=prior.step()
   a=ConnectedSession(root,rows,{'test':'CGR3'});at,ast=a.step()
   np.testing.assert_array_equal(ps,ast)
   for key in pt:self.assertEqual(pt[key],at[key])
   self.assertTrue(tree_equal(prior.model.state_dict(),a.model.state_dict()))
   p=root/'before-augmented.pt';a.save(p);trace,state=a.step();start=a.last_start.copy();expected=a.payload()
   self.assertEqual(trace['start']['mode'],'intermediate');self.assertGreater(trace['start']['added'],0)
   self.assertTrue((start>=d).all());self.assertTrue((start<=t).all());self.assertFalse(np.array_equal(start,t))
   b=ConnectedSession(root,rows,{'test':'CGR3'});b.restore(p);actual,raw=b.step()
   self.assertEqual(trace,actual);np.testing.assert_array_equal(start,b.last_start);np.testing.assert_array_equal(state,raw);self.assertTrue(tree_equal(expected,b.payload()))
   before=b.payload();evaluation=b.evaluate(0);np.testing.assert_array_equal(evaluation['initial'][0,0],d);self.assertTrue(tree_equal(before,b.payload()))
   for cls in [Prior,Bulk]:
    with self.assertRaises(ValueError):cls(root,rows,{'test':'CGR3'}).restore(p)
   old=Prior(root,rows,{'test':'CGR3'});old.save(root/'old.pt')
   with self.assertRaises(ValueError):b.restore(root/'old.pt')
   with self.assertRaises(ValueError):b.validate_visits([99],b.trace,b.completed,1)
if __name__=='__main__':unittest.main()
