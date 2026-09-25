import sys,time,unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_incremental_study import equivalent,TimedSampler


class IncrementalProtocolTests(unittest.TestCase):
    def test_full_and_prefix_checks_reject_forged_choices(self):
        route=np.zeros((4,)*3,bool);route[0,0,0]=True;old=route.copy();old[0,0,1]=True
        gen={'version':'old','wall_seconds':1.,'status':'time_limit','cube_width_cells':1,'seed':7,
            'selected_origins_zyx':[[0,0,0],[0,0,1]],'growth':{'trace':[{'origin_zyx':[0,0,1],'added_cells':1}]}}
        new=deepcopy(gen);new.update(version='new',status='target_reached',wall_seconds=.2)
        new['selected_origins_zyx'].append([0,0,2]);new['growth']['trace'].append({'origin_zyx':[0,0,2],'added_cells':1})
        field=old.copy();field[0,0,2]=True
        self.assertTrue(equivalent(new,field,route,{'generation':gen},old,route)['equal'])
        forged=deepcopy(new);forged['growth']['trace'][0]['added_cells']=9
        self.assertFalse(equivalent(forged,field,route,{'generation':gen},old,route)['equal'])
        forged=deepcopy(new);forged['seed']=8
        self.assertFalse(equivalent(forged,field,route,{'generation':gen},old,route)['equal'])
        old_gen=deepcopy(new);old_gen['version']='old';old_gen['wall_seconds']=99
        self.assertTrue(equivalent(new,field,route,{'generation':old_gen},field,route)['equal'])
        self.assertFalse(equivalent(new,field,route,{'generation':old_gen},old,route)['equal'])

    def test_sampler_retains_wall_cpu_and_rss_timestamps(self):
        with TimedSampler(.005) as sampler:time.sleep(.025)
        samples=np.asarray(sampler.timeline);report=sampler.report()
        self.assertEqual(samples.shape[1],3);self.assertGreaterEqual(len(samples),2)
        self.assertTrue((np.diff(samples[:,0])>=0).all());self.assertTrue((np.diff(samples[:,1])>=0).all())
        self.assertEqual(report['samples'],len(samples))
        self.assertEqual(report['max_sample_gap_seconds'],float(np.diff(samples[:,0]).max()))


if __name__=='__main__':unittest.main()
