"""Guard the fixed comparison against source/config drift and scene leakage."""
import copy
import unittest
from deploy.checkpoints import load_model_c
from nca.sensitivity import REPO,Session,make_metadata
from scripts.diagnostic_inputs import load_inputs

class SensitivityGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,_,cls.checkpoint=load_model_c()
        _,cls.inputs=load_inputs(REPO)

    def test_only_budget_coefficient_differs_between_matched_arms(self):
        a=make_metadata('mapped_30',0,self.inputs,self.cfg,self.checkpoint)
        b=make_metadata('mass_3',0,self.inputs,self.cfg,self.checkpoint)
        self.assertEqual(len(a['scene_order']),17)
        self.assertEqual(len(set(a['scene_order'])),17)
        self.assertNotIn('ref-05-sealed-partition',a['scene_order'])
        a['recipe']=b['recipe'];a['coefficients']['family_weights']['sparsity']=3.
        self.assertEqual(a,b)

    def test_runtime_or_coefficient_drift_is_rejected_before_training(self):
        meta=make_metadata('mass_3',0,self.inputs,self.cfg,self.checkpoint)
        for key,value in [('python_version','wrong'),('rollout_steps',4),('scheduler','decay')]:
            changed=copy.deepcopy(meta);changed[key]=value
            with self.assertRaisesRegex(ValueError,'differ'):
                Session(changed)

    def test_changed_scene_eligibility_is_rejected(self):
        modified=copy.deepcopy(self.inputs)
        modified['ref-05-sealed-partition']['feasible'].fill_(True)
        with self.assertRaisesRegex(ValueError,'Scene set differs'):
            make_metadata('mass_3',0,modified,self.cfg,self.checkpoint)

if __name__=='__main__':unittest.main()
