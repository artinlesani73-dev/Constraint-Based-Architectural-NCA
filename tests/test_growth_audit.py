import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch,MagicMock
import numpy as np
from scripts.growth_common import transition,vector_summary,cost_gate
import scripts.run_growth_audit as runner


class GrowthAuditTests(unittest.TestCase):
    def test_transition_distinguishes_replacement_from_equal_mass(self):
        a=np.array([0.,1.,0.5]);b=np.array([1.,0.,0.5])
        r=transition(a,b)
        self.assertEqual(r,{'absolute_material_change_sum':2.,'binary_added':1,'binary_removed':1,'binary_iou':0.})
        self.assertEqual(transition(np.zeros(3),np.zeros(3))['binary_iou'],1.)

    def test_zero_norm_and_conflicting_directions(self):
        norms,c=vector_summary({'a':np.array([3.,4.]),'b':np.array([-3.,-4.]),'zero':np.zeros(2)})
        self.assertEqual(norms['a'],5.);self.assertEqual(c['a']['b'],-1.)
        self.assertIsNone(c['a']['zero']);self.assertIsNone(c['zero']['zero'])

    def test_fixed_timing_matrix_and_invalid_pilot(self):
        self.assertTrue(cost_gate([10.,10.],[20.,20.])['admitted'])
        self.assertFalse(cost_gate([100.,100.],[20.,20.])['admitted'])
        for a,b in [([1.],[1.,1.]),([1.,1.],[0.,2.]),([float('nan'),1.],[1.,1.])]:
            with self.assertRaises(ValueError):cost_gate(a,b)

    def test_delayed_success_does_not_bypass_elapsed_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=MagicMock();process.wait.return_value=0
            with patch.object(runner.STORE,'path',return_value=Path(tmp)),patch.object(runner.STORE,'attach'), \
                 patch.object(runner,'record') as record,patch.object(runner.subprocess,'Popen',return_value=process), \
                 patch.object(runner.time,'perf_counter',side_effect=[0.,181.]):
                with self.assertRaisesRegex(TimeoutError,'elapsed cap'):
                    runner.launch('mock','worker',[],180,'growth')
                saved=record.call_args.args[2]
                self.assertTrue(saved['elapsed_cap_exceeded']);self.assertFalse(saved['timed_out'])
                self.assertEqual(saved['returncode'],0)


if __name__=='__main__':unittest.main()
