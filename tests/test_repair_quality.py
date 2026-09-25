from pathlib import Path
from copy import deepcopy
import json,subprocess,sys,unittest,uuid
import numpy as np
from nca.repair_quality import SEEDS,FIRING,SETTINGS,primary_gate
from nca.quality_package import verify
from nca.experiments import digest
from nca.repair_portable import PortableSession
from scripts.evaluate_repair_quality import predict,read_models

ROOT=Path(__file__).resolve().parents[1]


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'.local-artifacts/testing/NR3'/uuid.uuid4().hex;self.root.mkdir(parents=True)
        self.examples=[];self.records=[]
        def metric(iou,valid):return {'iou':iou,'request_error_cells':0,'targets':{'contract_pass':valid,'domain_voxels':1000}}
        for case in range(18):
            for damage in ('intact','cube5','slab2'):
                self.examples.append({'case':str(case),'damage':damage,'split':'test',
                                      'metrics':{'unchanged':metric(.75,False),'closing3':metric(.85,True)}})
                for seed in SEEDS:
                    for firing in FIRING:self.records.append({'case':str(case),'damage':damage,'split':'test','seed':seed,
                        'firing_seed':firing,'steps':32,'checkpoint':256,'metrics':metric(1.,True)})

    def test_complete_primary_gate_passes_and_reports_each_seed(self):
        result=primary_gate(self.records,self.examples)
        self.assertTrue(result['admit_further_repair_study']);self.assertFalse(result['studio_promotion'])
        self.assertEqual(result['primary_observations'],486);self.assertEqual(len(result['by_seed']),3)

    def test_missing_duplicate_and_wrong_checkpoint_cannot_pass(self):
        for records in (self.records[:-1],self.records+[self.records[0]]):
            with self.assertRaises(ValueError):primary_gate(records,self.examples)
        self.records[0]['checkpoint']=192
        with self.assertRaises(ValueError):primary_gate(self.records,self.examples)

    def test_intact_regression_and_budget_violation_fail_even_with_high_iou(self):
        rows=deepcopy(self.records);rows[0]['metrics']['targets']['contract_pass']=False
        self.assertFalse(primary_gate(rows,self.examples)['admit_further_repair_study'])
        rows=deepcopy(self.records);next(x for x in rows if x['damage']=='cube5')['metrics']['request_error_cells']=11
        self.assertFalse(primary_gate(rows,self.examples)['admit_further_repair_study'])

    def test_good_mean_cannot_hide_failed_model_seed(self):
        for x in self.records:
            if x['seed']==1203 and x['damage']!='intact':x['metrics']['iou']=.86
        result=primary_gate(self.records,self.examples)
        self.assertEqual([x['passed'] for x in result['by_seed']],[True,True,False])

    def test_cpu_evaluator_preserves_nr2_rollout_and_projection(self):
        target=np.zeros((6,)*3,np.float32);target[1:5,1:5,1:5]=1
        damaged=target.copy();damaged[2:4,2:4,2:4]=0
        context=np.zeros((7,6,6,6),np.float32);context[:2]=1;context[6]=.24
        p=self.root/'example.npz'
        with p.open('xb') as f:np.savez_compressed(f,target=target,damaged=damaged,condition=context)
        row={'arrays':p.name,'arrays_sha256':digest(p),'split':'train'}
        session=PortableSession(self.root,[row],{'synthetic':'evaluator-parity'})
        before=session.payload();expected=session.evaluate()
        state,prob=predict(session.model.state_dict(),{'occupancy':damaged,'context':context},16,2101)
        np.testing.assert_array_equal(state,expected['state']);np.testing.assert_array_equal(prob,expected['probability'])
        self.assertEqual(session.completed,before['completed'])

    def test_test_evaluation_refuses_missing_final_models(self):
        with self.assertRaises(ValueError):read_models([], '0'*64)

    def test_gpu_command_is_disarmed_and_cpu_rehearsal_is_explicit(self):
        for args in (['--seed','1201'],['--seed','1201','--device','cpu']):
            result=subprocess.run([sys.executable,str(ROOT/'scripts/colab_repair_quality.py'),*args],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertTrue('approval required' in result.stderr or 'eight-update rehearsal' in result.stderr)

    def test_quality_package_rejects_heldout_rows(self):
        rows=[{'case':str(i//3),'split':'train','arrays':f'data/{i}.npz','arrays_sha256':'0'*64} for i in range(81)]
        rows[0]['split']='test'
        for name,obj in [('study.json',SETTINGS),('dataset.json',{'rows':rows})]:
            (self.root/name).write_text(json.dumps(obj),encoding='utf-8')
        m={'version':'NR3_quality_package_v1','files':{n:digest(self.root/n) for n in ('study.json','dataset.json')}}
        (self.root/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'TRAIN split differs'):verify(self.root)


if __name__=='__main__':unittest.main()
