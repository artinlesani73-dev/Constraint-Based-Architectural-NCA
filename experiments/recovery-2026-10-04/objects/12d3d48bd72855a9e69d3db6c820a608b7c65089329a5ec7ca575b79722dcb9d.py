from pathlib import Path
import sys,json,hashlib,unittest,time
ROOT=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from generation_data import seed_inputs,teacher_distance,training_start,generate
from nca.connected_repair import ConnectedRepair,neighbors6
from nca.repair_training import perceive
from nca.massing_targets import evaluate_targets
from nca.massing_cases import target_context
from anchored_teacher import generate_anchored_mass,CoverageGeneratorSpec
OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)

class PreparationTests(unittest.TestCase):
    def test_dataset_integrity_and_recomputation(self):
        manifest=json.loads((OUT/'split-manifest.json').read_text());data=json.loads((OUT/'dataset.json').read_text())
        entries={e['id']:e for e in manifest['entries']};families={};hashes={}
        for e in entries.values():
            self.assertEqual(families.setdefault(e['family'],e['split']),e['split'])
            self.assertEqual(hashes.setdefault(e['context_sha256'],e['split']),e['split'])
        self.assertEqual(len(data['rows']),36)
        config=json.loads((OUT/'environment.json').read_text())['config']
        for row in data['rows']:
            self.assertNotEqual(row['split'],'reserved')
            path=OUT/row['arrays'];self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['arrays_sha256'])
            with np.load(path,allow_pickle=False) as a:
                x=seed_inputs(a['context']);np.testing.assert_array_equal(x['occupancy'],a['seed'])
                distance=teacher_distance(a['target'],a['seed']);np.testing.assert_array_equal(distance,a['distance'])
                np.testing.assert_array_equal(distance>=0,a['target'])
                for depth in range(int(distance.max())+1):
                    state=training_start(distance,depth,'train')
                    if depth:
                        old=training_start(distance,depth-1,'train')
                        near=neighbors6(torch.from_numpy(old)[None,None]).numpy()[0,0]
                        self.assertFalse(((state>old)&~near).any())
                scene=entries[row['id'].rsplit('-v',1)[0]]['scene'];f,d,_=target_context(scene,config)
                score,_=evaluate_targets(a['target'],scene,f,d);self.assertEqual(score,row['score'])
                self.assertEqual(score['resolved_cube_width_cells'],3)
        with self.assertRaises(ValueError): training_start(np.zeros((3,3,3)),0,'development')

    def test_seed_only_rollout_and_training_gradients(self):
        data=json.loads((OUT/'dataset.json').read_text());row=next(r for r in data['rows'] if r['split']=='train')
        with np.load(OUT/row['arrays'],allow_pickle=False) as a: c=a['context'].copy();target=a['target'].copy()
        torch.manual_seed(1201);model=ConnectedRepair();model.last.bias.data[0]=.1
        before=c.copy();first=generate(model,c,2101,4)['field'];target[:]=False
        second=generate(model,c,2101,4)['field'];self.assertTrue(torch.equal(first,second));np.testing.assert_array_equal(c,before)
        self.assertGreater(int(first.sum()),1)
        x=seed_inputs(c);self.assertEqual(int(x['occupancy'].sum()),1)
        with self.assertRaises(TypeError):generate(model,c,target=target)
        with np.load(OUT/row['arrays'],allow_pickle=False) as a:
            start=training_start(a['distance'],3,'train');label=torch.from_numpy(a['target'].astype(np.float32))[None,None]
        result=model.rollout(torch.from_numpy(start)[None,None],perceive(torch.from_numpy(c)[None]),
            torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(1201),4,target=label)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0)

    def test_teacher_replay_and_seed_rejection(self):
        data=json.loads((OUT/'dataset.json').read_text());row=next(r for r in data['rows'] if r['family']=='partial_obstruction')
        entries=json.loads((OUT/'split-manifest.json').read_text())['entries']
        scene=next(e['scene'] for e in entries if e['id']==row['id'].rsplit('-v',1)[0])
        config=json.loads((OUT/'environment.json').read_text())['config'];f,d,_=target_context(scene,config)
        target,_,_=generate_anchored_mass(scene,f,d,0,row['anchor'],CoverageGeneratorSpec(target_fraction=row['request']))
        with np.load(OUT/row['arrays'],allow_pickle=False) as a:
            np.testing.assert_array_equal(target,a['target'])
            bad=target.copy();bad[tuple(row['anchor'])]=False
            with self.assertRaises(ValueError):teacher_distance(bad,a['seed'])
            c=a['context'].copy();c[5]=0
            with self.assertRaises(ValueError):seed_inputs(c)

if __name__=='__main__':
    started=time.perf_counter();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests))
    report=dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),seconds=time.perf_counter()-started,
        success=r.wasSuccessful(),trained_model_quality=False,checkpoint_recovery_tested=False)
    with (OUT/'verification.json').open('x') as f:json.dump(report,f,indent=2)
    sys.exit(not r.wasSuccessful())
