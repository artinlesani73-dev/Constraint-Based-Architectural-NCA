import copy
import tempfile
from pathlib import Path
import unittest
import torch
from deploy.checkpoints import load_model_c
from scripts.diagnostic_inputs import load_inputs
from scripts.run_raw_access_training import parity_metadata_equal
from nca.raw_access_training import REPO,Session,make_metadata,next_horizon,cost_gate,RAW,CANDIDATE,objective_pair
from nca.access_training import make_metadata as f2_metadata,objective_pair as f2_objective
from nca.recovery import save_checkpoint
from nca.sensitivity import contexts
from nca.losses import LossSpec
from nca.objective import weighted_total


class RawTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg,_,cls.checkpoint=load_model_c();_,cls.inputs=load_inputs(REPO)

    def meta(self,objective=RAW):
        return make_metadata('mass_3','ref-01-ground-pair',self.inputs,self.cfg,self.checkpoint,objective)

    def test_fixed_horizon_and_metadata_guards(self):
        m=self.meta();self.assertEqual([next_horizon(m,u) for u in (0,1,63,64)],[16,16,16,None])
        for u in (-1,65,True,1.5):
            with self.assertRaises(ValueError):next_horizon(m,u)
        for key,value in [('horizon_schedule',[16,50]*32),('objective_version','unknown'),('code_sha256',{}),('seed',1)]:
            bad=copy.deepcopy(m);bad[key]=value
            with self.assertRaises(ValueError):Session(bad)

    def test_parity_requires_projected_objective_and_same_science(self):
        old=f2_metadata('mass_3','ref-01-ground-pair',self.inputs,self.cfg,self.checkpoint)
        self.assertTrue(parity_metadata_equal(self.meta(CANDIDATE),old))
        self.assertFalse(parity_metadata_equal(self.meta(),old))
        for key,value in [('seed',1),('horizon_schedule',[16,50]*32),('unexpected',True)]:
            m=self.meta(CANDIDATE);m[key]=value
            self.assertFalse(parity_metadata_equal(m,old))

    def test_only_access_term_changes(self):
        name='ref-01-ground-pair';state=self.inputs[name]['seed'].clone()
        ctx,allow=contexts(self.inputs,self.cfg,[name])[name]
        raw=torch.full_like(state[:,self.cfg['ch_structure']],-.02,requires_grad=True)
        state[:,self.cfg['ch_structure']]=raw.clamp(0,1)*ctx.permitted
        _,base,_=f2_objective(state,raw,ctx,self.cfg,allow,LossSpec())
        _,parity,_=objective_pair(state,raw,ctx,self.cfg,allow,LossSpec(),CANDIDATE)
        _,candidate,_=objective_pair(state,raw,ctx,self.cfg,allow,LossSpec(),RAW)
        self.assertEqual(len(candidate['terms']),9)
        for family in base['terms']:
            self.assertTrue(torch.equal(base['terms'][family],parity['terms'][family]))
            if family!='access':self.assertTrue(torch.equal(base['terms'][family],candidate['terms'][family]))
        for name in base['regularizers']:self.assertTrue(torch.equal(base['regularizers'][name],candidate['regularizers'][name]))
        self.assertGreater(float(candidate['terms']['access'][0].detach()),1)

    def test_restore_rejects_different_objective(self):
        m=self.meta();s=Session(m)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'u0.pt';save_checkpoint(p,s.model,s.optimizer,s.scheduler,s.generator,m,0)
            with self.assertRaises(ValueError):Session(self.meta(CANDIDATE),p)

    def test_cost_requires_final_grid_and_rejects_overruns(self):
        self.assertTrue(cost_gate([1],[1],[1],[1])['admitted'])
        self.assertFalse(cost_gate([100],[1],[1],[1])['admitted'])
        self.assertFalse(cost_gate([1],[1],[1000],[1])['admitted'])
        for i in range(4):
            for invalid in ([],[-1],[float('nan')]):
                groups=[[1] for _ in range(4)];groups[i]=invalid
                with self.assertRaises(ValueError):cost_gate(*groups)

    def test_reused_evaluation_terms_match_full_recomputation(self):
        from nca.experiments import read_json
        from nca.raw_access_training import PROPOSAL
        s=Session(self.meta());row,fields=s.score(3,2)
        name=s.metadata['scene'];ctx,allow=s.contexts[name]
        state=s.inputs[name]['seed'].clone();state[:,s.config['ch_structure']]=torch.from_numpy(fields['material'])
        with torch.no_grad():
            _,reference,details=objective_pair(state,torch.from_numpy(fields['raw']),ctx,s.config,allow,LossSpec(),RAW)
            self.assertEqual(row['raw_access'],float(reference['terms']['access'][0]))
            self.assertEqual(row['raw_details'],details[0])
            for name,c in read_json(REPO/PROPOSAL)['recipes'].items():
                self.assertEqual(row['raw_totals'][name],float(weighted_total(reference,c['family_weights'],c['regularizer_weights'])))
