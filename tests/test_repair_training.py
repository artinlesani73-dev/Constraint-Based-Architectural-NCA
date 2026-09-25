from pathlib import Path
from copy import deepcopy
import json
import uuid
import random
import unittest
import numpy as np
import torch
from nca.repair_training import (RepairSession, RepairNCA, perceive, balanced_loss,
    save_payload, read_payload, latest_verified)
from nca.recovery import tree_equal

ROOT=Path(__file__).resolve().parents[1]


def sample():
    target=np.zeros((6,)*3,np.float32);target[1:5,1:5,1:5]=1
    damaged=target.copy();damaged[2:4,2:4,2:4]=0
    context=np.zeros((7,6,6,6),np.float32);context[:2]=1;context[6]=.24
    return {'occupancy':damaged,'context':context},target


class RepairTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory=ROOT/'.local-artifacts/testing/NR1'/uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.inputs,self.target=sample()

    def session(self):
        return RepairSession(self.inputs,self.target,{'study':'unit'})

    def test_local_derivatives_and_zero_update_initialization(self):
        x=torch.arange(6,dtype=torch.float32)[None,None,None,None,:].expand(1,1,6,6,6)
        p=perceive(x)
        self.assertEqual(float(p[0,1,2,2,2]),1.)
        self.assertEqual(float(p[0,2,2,2,2]),0.)
        s=self.session();_,state,prob=s.evaluate()
        np.testing.assert_array_equal(prob>.5,self.inputs['occupancy'].astype(bool))
        self.assertFalse(state[1:].any())
        self.assertEqual(sum(p.numel() for p in s.model.parameters()),4424)

    def test_bce_gradient_reaches_wrong_saturated_logit(self):
        logits=torch.tensor([-20.,20.],requires_grad=True)
        target=torch.tensor([1.,0.]);allowed=torch.ones(2,dtype=torch.bool)
        loss=balanced_loss(logits,target,allowed);loss.backward()
        self.assertLess(float(logits.grad[0]),-.49)
        self.assertGreater(float(logits.grad[1]),.49)
        with self.assertRaises(ValueError):balanced_loss(logits,torch.ones(2),allowed)

    def test_actual_update_and_full_resume_match(self):
        s=self.session();s.step();checkpoint=s.save(self.directory/'checkpoint-0001.pt')
        a=s.step();expected=s.payload()
        random.random();np.random.rand();torch.rand(4)
        restart=self.session();self.assertEqual(restart.restore(checkpoint),1)
        b=restart.step()
        self.assertTrue(tree_equal(a[0],b[0]));np.testing.assert_array_equal(a[1],b[1])
        self.assertTrue(tree_equal(expected,restart.payload()))

    def test_changed_source_and_actual_data_rejected(self):
        s=self.session();p=s.save(self.directory/'checkpoint-0000.pt')
        with self.assertRaises(ValueError):RepairSession(self.inputs,self.target,{'study':'changed'}).restore(p)
        changed=deepcopy(self.inputs);changed['occupancy'][1,1,1]=0
        with self.assertRaises(ValueError):RepairSession(changed,self.target,{'study':'unit'}).restore(p)

    def test_partial_publication_not_resumable_and_no_overwrite(self):
        s=self.session();a=s.save(self.directory/'checkpoint-0000.pt')
        s.step()
        with self.assertRaises(OSError):save_payload(self.directory/'checkpoint-0001.pt',s.payload(),fault='before_manifest')
        with self.assertRaises(OSError):save_payload(self.directory/'checkpoint-0002.pt',s.payload(),fault='before_publish')
        self.assertTrue(list(self.directory.glob('*.partial.pt')))
        latest,_=latest_verified(self.directory,s.identity);self.assertEqual(latest,a)
        with self.assertRaises(FileExistsError):s.save(a)
        with self.assertRaises(FileNotFoundError):read_payload(self.directory/'checkpoint-0001.pt',s.identity)

    def test_corrupt_newest_falls_back_with_rejection_record(self):
        s=self.session();old=s.save(self.directory/'checkpoint-0000.pt');s.step()
        new=s.save(self.directory/'checkpoint-0001.pt')
        # Test-created corruption is deliberately retained, including manifest.
        with new.open('ab') as f:f.write(b'corruption')
        latest,rejections=latest_verified(self.directory,s.identity)
        self.assertEqual(latest,old);self.assertEqual(len(rejections),1)
        with self.assertRaises(ValueError):s.restore(new)

    def test_static_inputs_immutable_and_projection_explicit(self):
        inputs=deepcopy(self.inputs);inputs['context'][0,0]=0
        s=RepairSession(inputs,self.target,{'study':'unit'})
        with torch.no_grad():s.model.last.bias.fill_(.2)
        _,state,prob=s.evaluate()
        self.assertFalse(prob[0].any());self.assertFalse(state[1:,0].any())
        s.context[0,6,0,0,0]=.3
        with self.assertRaises(ValueError):s.step()


if __name__=='__main__':unittest.main()
