"""Exercise real optimizer, scheduler and four RNG streams across checkpoint restore."""
from pathlib import Path
from copy import deepcopy
import random
import tempfile
import unittest
import numpy as np
import torch
from nca.recovery import save_checkpoint, restore_checkpoint, tree_equal


class RecoveryTests(unittest.TestCase):
    def instances(self):
        model=torch.nn.Linear(3,2).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.8)
        return model,optimizer,scheduler,torch.Generator().manual_seed(73)

    def step(self, items):
        model,opt,sched,gen=items
        scale=random.random()+float(np.random.random())
        x=torch.randn(4,3,dtype=torch.float64)*scale
        y=torch.rand(4,2,dtype=torch.float64,generator=gen)
        opt.zero_grad(set_to_none=True)
        loss=(model(x)-y).square().mean();loss.backward();opt.step();sched.step()
        return loss.item()

    def test_exact_optimizer_scheduler_and_rng_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            random.seed(11);np.random.seed(11);torch.manual_seed(11)
            items=self.instances(); self.step(items)
            path=Path(d)/'saved.pt';meta={'scenes':['a','b'],'config':{'steps':4},'source':'fixture'}
            save_checkpoint(path,*items,meta,1)
            losses=[self.step(items) for _ in range(3)]
            final=Path(d)/'final.pt';save_checkpoint(final,*items,meta,4)
            random.seed(93);np.random.seed(93);torch.manual_seed(93)
            fresh=self.instances()
            self.assertEqual(restore_checkpoint(path,*fresh,meta),1)
            self.assertEqual(losses,[self.step(fresh) for _ in range(3)])
            resumed=Path(d)/'resumed.pt';save_checkpoint(resumed,*fresh,meta,4)
            self.assertTrue(tree_equal(torch.load(final,weights_only=True),torch.load(resumed,weights_only=True)))

    def test_refuses_overwrite_and_wrong_config_before_model_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            items=self.instances();path=Path(d)/'state.pt';meta={'scene':'a'}
            save_checkpoint(path,*items,meta,0);before=path.read_bytes()
            with self.assertRaises(FileExistsError):save_checkpoint(path,*items,meta,1)
            self.assertEqual(path.read_bytes(),before)
            fresh=self.instances();original=deepcopy(fresh[0].state_dict())
            with self.assertRaisesRegex(ValueError,'metadata'):restore_checkpoint(path,*fresh,{'scene':'b'})
            self.assertTrue(tree_equal(original,fresh[0].state_dict()))

    def test_truncated_checkpoint_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            items=self.instances();path=Path(d)/'broken.pt';path.write_bytes(b'not-a-checkpoint')
            with self.assertRaises(Exception):restore_checkpoint(path,*items,{'scene':'a'})


if __name__ == '__main__':unittest.main()
