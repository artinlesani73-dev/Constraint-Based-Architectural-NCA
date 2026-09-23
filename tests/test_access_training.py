"""Isolation, metadata and RNG guards for the access-only learning intervention."""
import copy
import unittest
import torch
from deploy.checkpoints import load_model_c
from nca.access_training import (REPO,Session,make_metadata,objective_pair,cost_gate,LEGACY,CANDIDATE)
from nca.fitting import make_metadata as f1_metadata
from nca.losses import LossSpec
from scripts.diagnostic_inputs import load_inputs
from scripts.run_access_training import parity_metadata_equal


class AccessTrainingGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,_,cls.checkpoint=load_model_c();_,cls.inputs=load_inputs(REPO)

    def meta(self,objective=CANDIDATE):
        return make_metadata('mass_3','ref-01-ground-pair',self.inputs,self.cfg,self.checkpoint,objective)

    def test_exactly_one_family_replaced(self):
        s=Session(self.meta());name=s.metadata['scene'];ctx,allow=s.contexts[name]
        state=s.inputs[name]['seed'].clone()
        raw=torch.full_like(state[:,self.cfg['ch_structure']],.4,requires_grad=True)
        state[:,self.cfg['ch_structure']]=raw*ctx.permitted
        old,new,details=objective_pair(state,raw,ctx,self.cfg,allow,LossSpec())
        self.assertEqual(len(old['terms']),9);self.assertEqual(old['terms'].keys(),new['terms'].keys())
        for k in old['terms']:
            if k!='access':self.assertIs(old['terms'][k],new['terms'][k])
        self.assertIs(old['regularizers'],new['regularizers'])
        self.assertIs(old['mass_ratio'],new['mass_ratio'])
        new['terms']['access'].sum().backward()
        self.assertEqual(float(raw.grad.abs().sum()),1.)
        self.assertTrue(details[0]['legal_route_exists'])

    def test_source_runtime_or_objective_drift_rejected(self):
        for k,v in [('objective_version','wrong'),('rollout_steps',50),('proposal_sha256','bad'),('code_sha256',{})]:
            m=copy.deepcopy(self.meta());m[k]=v
            with self.assertRaises(ValueError):Session(m)

    def test_baseline_parity_only_allows_declared_identity_changes(self):
        a=self.meta(LEGACY);b=f1_metadata('mass_3','ref-01-ground-pair',self.inputs,self.cfg,self.checkpoint)
        self.assertTrue(parity_metadata_equal(a,b))
        for k,v in [('seed',1),('updates',65),('new_unexpected_key',True),('objective_version',CANDIDATE)]:
            bad=copy.deepcopy(a);bad[k]=v
            self.assertFalse(parity_metadata_equal(bad,b))

    def test_dual_evaluation_preserves_rng(self):
        s=Session(self.meta());before=s.generator.get_state().clone();global_before=torch.get_rng_state().clone()
        a,fields=s.score(1,2);b,other=s.score(1,2)
        self.assertEqual(a,b);self.assertIn('candidate_metrics',a)
        self.assertTrue(torch.equal(before,s.generator.get_state()))
        self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
        self.assertTrue((fields['material']==other['material']).all())

    def test_timing_gate_includes_evaluation_and_refuses_invalid(self):
        self.assertTrue(cost_gate([1.],[1.],[1.])['admitted'])
        self.assertFalse(cost_gate([1.],[100.],[1.])['admitted'])
        for v in ([],[-1.],[float('nan')]):
            with self.assertRaises(ValueError):cost_gate(v,[1.],[1.])


if __name__=='__main__':unittest.main()
