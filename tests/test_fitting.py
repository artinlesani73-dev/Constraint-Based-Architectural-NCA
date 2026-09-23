"""Scientific guards for repeated-scene fitting and cost admission."""
import copy
import unittest
import torch
from deploy.checkpoints import load_model_c
from nca.fitting import REPO, Session, make_metadata, cost_gate
from nca.sensitivity import Session as K2Session
from scripts.diagnostic_inputs import load_inputs


class FittingGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _, cls.checkpoint = load_model_c()
        _, cls.inputs = load_inputs(REPO)

    def meta(self, recipe='mass_3'):
        return make_metadata(recipe,'ref-01-ground-pair',self.inputs,self.cfg,self.checkpoint)

    def test_matched_recipes_repeated_schedule_unchanged_step(self):
        a,b=self.meta(),self.meta('mapped_30')
        self.assertEqual(a['scene_order'],['ref-01-ground-pair']*64)
        b['recipe']=a['recipe']; b['coefficients']['family_weights']['sparsity']=3.
        self.assertEqual(a,b)
        self.assertIs(Session.step,K2Session.step)

    def test_schedule_or_runtime_drift_rejected(self):
        for key,value in [('rollout_steps',50),('python_version','wrong'),('scene_order',['ref-06-minimal-smoke']*64)]:
            m=copy.deepcopy(self.meta());m[key]=value
            with self.assertRaisesRegex(ValueError,'differ'):
                Session(m)

    def test_evaluation_does_not_consume_training_rng(self):
        session=Session(self.meta())
        before=session.generator.get_state().clone(); global_before=torch.get_rng_state().clone()
        a,fields=session.score(1,2);b,other=session.score(1,2)
        self.assertEqual(a,b)
        self.assertTrue(torch.equal(before,session.generator.get_state()))
        self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
        self.assertTrue((fields['material']==other['material']).all())

    def test_cost_admission_rejects_slow_and_invalid_timings(self):
        self.assertTrue(cost_gate([1.,1.],[1.],[1.])['admitted'])
        self.assertFalse(cost_gate([100.],[1.],[1.])['admitted'])
        for values in ([],[float('nan')],[-1.]):
            with self.assertRaises(ValueError):cost_gate(values,[1.],[1.])


if __name__=='__main__':unittest.main()
