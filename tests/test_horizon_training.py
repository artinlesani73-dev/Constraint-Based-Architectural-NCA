"""Prevent schedule drift, invalid recovery cursors and incomplete cost admission."""
import copy
import unittest
from deploy.checkpoints import load_model_c
from nca.horizon_training import REPO,Session,make_metadata,next_horizon,cost_gate,CONSTANT,MIXED
from nca.access_training import make_metadata as f2_metadata
from scripts.diagnostic_inputs import load_inputs
from scripts.run_horizon_training import parity_metadata_equal


class HorizonTrainingGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,_,cls.checkpoint=load_model_c();_,cls.inputs=load_inputs(REPO)

    def meta(self,schedule=MIXED):
        return make_metadata('mass_3','ref-01-ground-pair',self.inputs,self.cfg,self.checkpoint,schedule)

    def test_restore_cursor_switches_both_directions_and_stops(self):
        m=self.meta()
        self.assertEqual([next_horizon(m,u) for u in (0,1,2,63,64)],[16,50,16,50,None])
        for u in (-1,65,True,1.5):
            with self.assertRaises(ValueError):next_horizon(m,u)

    def test_metadata_rejects_schedule_or_source_drift(self):
        for key,value in [('horizon_schedule',[50,16]*32),('schedule_version','unknown'),('code_sha256',{}),('objective_version','research_objective_v1')]:
            m=copy.deepcopy(self.meta());m[key]=value
            with self.assertRaises(ValueError):Session(m)

    def test_constant_parity_rejects_changed_science(self):
        a=self.meta(CONSTANT);b=f2_metadata('mass_3','ref-01-ground-pair',self.inputs,self.cfg,self.checkpoint)
        self.assertTrue(parity_metadata_equal(a,b))
        for key,value in [('horizon_schedule',[16,50]*32),('seed',1),('unexpected',True)]:
            changed=copy.deepcopy(a);changed[key]=value
            self.assertFalse(parity_metadata_equal(changed,b))

    def test_cost_admission_requires_long_rollouts_and_final_grid(self):
        self.assertTrue(cost_gate([1],[1],[1],[1],[1])['admitted'])
        self.assertFalse(cost_gate([1],[100],[1],[1],[1])['admitted'])
        self.assertFalse(cost_gate([1],[1],[1],[1000],[1])['admitted'])
        for index in range(5):
            for invalid in ([],[-1],[float('nan')]):
                groups=[[1] for _ in range(5)];groups[index]=invalid
                with self.assertRaises(ValueError):cost_gate(*groups)


if __name__=='__main__':unittest.main()
