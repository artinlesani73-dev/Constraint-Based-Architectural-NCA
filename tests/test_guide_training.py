import copy
from pathlib import Path
import tempfile
import unittest
import torch
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from scripts.diagnostic_inputs import load_inputs
from nca.conditioning import GuidedNCA,GuideContext,guided_rollout,BASELINE,GUIDED
from nca.interventions import experimental_rollout
from nca.guide_training import Session,make_metadata,REPO,cost_gate
from nca.raw_access_training import make_metadata as old_metadata
from nca.recovery import save_checkpoint,tree_equal
from scripts.run_guide_training import parity_metadata_equal

class GuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg,cls.weights,cls.checkpoint=load_model_c();_,cls.inputs=load_inputs(REPO)
        cls.scene='ref-06-minimal-smoke';cls.item=cls.inputs[cls.scene]

    def meta(self,architecture=GUIDED):
        return make_metadata('mass_3',self.scene,self.inputs,self.cfg,self.checkpoint,architecture)

    def pair(self):
        torch.manual_seed(52);base=UrbanPavilionNCA(self.cfg);base.load_state_dict(self.weights);rng=torch.get_rng_state().clone()
        torch.manual_seed(52);new=GuidedNCA(self.cfg);new.load_original(self.weights)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        return base,new

    def ctx(self,seed=None,scaffold=None):
        return GuideContext(self.scene,self.item['seed'] if seed is None else seed,self.item['scaffold'] if scaffold is None else scaffold,self.cfg)

    def test_zero_branch_forward_backbone_gradient_and_rng_parity(self):
        base,new=self.pair();x=self.item['seed'];g=self.item['scaffold'];ctx=self.ctx()
        rng=torch.get_rng_state().clone();a=torch.Generator().manual_seed(7);b=torch.Generator().manual_seed(7)
        out=experimental_rollout(base,x,g,'hard_preclamp',3,a)
        other=guided_rollout(new,x,g,'hard_preclamp',3,b,ctx,self.scene)
        for k in ('state','raw_material'):self.assertTrue(torch.equal(out[k],other[k]),k)
        self.assertTrue(torch.equal(a.get_state(),b.get_state()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        # Same scalar probes both projected trajectories and final raw material.
        (out['state'].square().mean()+out['raw_material'].square().mean()).backward()
        (other['state'].square().mean()+other['raw_material'].square().mean()).backward()
        for name,p in base.named_parameters():self.assertTrue(torch.equal(p.grad,dict(new.named_parameters())[name].grad),name)
        self.assertEqual(new.guide_weight.numel(),384)
        self.assertGreater(float(new.guide_weight.grad.norm()),0)

    def test_batch_two_and_repeated_backward_have_fresh_graphs(self):
        _,new=self.pair();seed=self.item['seed'].repeat(2,1,1,1,1);g=self.item['scaffold'].repeat(2,1,1,1);ctx=self.ctx(seed,g)
        grads=[]
        for _ in range(2):
            new.zero_grad(set_to_none=True)
            out=guided_rollout(new,seed,g,'hard_preclamp',2,torch.Generator().manual_seed(1),ctx,self.scene)
            out['raw_material'].square().mean().backward();grads.append(new.guide_weight.grad.clone())
        self.assertTrue(torch.equal(*grads))

    def test_context_owned_and_binding_rejects_changes(self):
        ctx=self.ctx();x=self.item['seed'];g=self.item['scaffold']
        f=ctx.features(self.scene,x,g,self.cfg);f.zero_()
        self.assertGreater(float(ctx.features(self.scene,x,g,self.cfg).abs().sum()),0)
        for scene,xx,gg,cfg in [(self.scene+'x',x,g,self.cfg),(self.scene,x+1,g,self.cfg),
            (self.scene,x,1-g,self.cfg),(self.scene,x,g,dict(self.cfg,fire_rate=.1))]:
            with self.assertRaises(ValueError):ctx.features(scene,xx,gg,cfg)
        ctx._features.zero_()
        with self.assertRaises(ValueError):ctx.features(self.scene,x,g,self.cfg)

    def test_invalid_context_and_rollout_fail_loudly(self):
        x=self.item['seed'];g=self.item['scaffold'];_,new=self.pair()
        for bad in (g[:,0],g.double(),g+2,torch.full_like(g,float('nan')),g.clone().requires_grad_()):
            with self.assertRaises(ValueError):self.ctx(x,bad)
        for steps in (0,-1,True,1.5):
            with self.assertRaises(ValueError):guided_rollout(new,x,g,'hard_preclamp',steps,torch.Generator(),self.ctx(),self.scene)
        with self.assertRaises(ValueError):guided_rollout(new,x,g,'hard_preclamp',1,torch.Generator())

    def test_disabled_path_exact_and_does_not_add_parameters(self):
        base,_=self.pair();x=self.item['seed'];g=self.item['scaffold']
        a=guided_rollout(base,x,g,'hard_preclamp',2,torch.Generator().manual_seed(0))
        b=experimental_rollout(base,x,g,'hard_preclamp',2,torch.Generator().manual_seed(0))
        self.assertTrue(tree_equal(a,b));self.assertNotIn('guide_weight',base.state_dict())

    def test_nonzero_guide_changes_evolving_state_only(self):
        base,new=self.pair();x=self.item['seed'];g=self.item['scaffold']
        with torch.no_grad():new.guide_weight.fill_(.01)
        a=guided_rollout(new,x,g,'hard_preclamp',2,torch.Generator().manual_seed(0),self.ctx(),self.scene)
        b=experimental_rollout(base,x,g,'hard_preclamp',2,torch.Generator().manual_seed(0))
        self.assertFalse(torch.equal(a['raw_material'],b['raw_material']))
        self.assertTrue(torch.equal(a['state'][:,:self.cfg['n_frozen']],x[:,:self.cfg['n_frozen']]))

    def test_migration_rejects_conditioned_or_missing_backbone(self):
        _,new=self.pair()
        with self.assertRaises(ValueError):new.load_original(new.state_dict())
        weights=dict(self.weights);weights.pop('update_net.0.weight')
        with self.assertRaises(ValueError):new.load_original(weights)

    def test_metadata_and_restore_reject_architecture_or_context_change(self):
        m=self.meta();s=Session(m)
        for key,value in [('architecture_version','bad'),('scene','ref-01-ground-pair'),('code_sha256',{}),('seed',1)]:
            bad=copy.deepcopy(m);bad[key]=value
            with self.assertRaises(ValueError):Session(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'initial.pt';save_checkpoint(path,s.model,s.optimizer,s.scheduler,s.generator,m,0)
            restored=Session(m,path);self.assertTrue(tree_equal(s.model.state_dict(),restored.model.state_dict()))
            with self.assertRaises(ValueError):Session(self.meta(BASELINE),path)

    def test_parity_metadata_preserves_objective_and_science(self):
        old=old_metadata('mass_3',self.scene,self.inputs,self.cfg,self.checkpoint)
        self.assertTrue(parity_metadata_equal(self.meta(BASELINE),old))
        self.assertFalse(parity_metadata_equal(self.meta(),old))
        bad=self.meta(BASELINE);bad['objective_version']='component_objective_v2'
        self.assertFalse(parity_metadata_equal(bad,old))

    def test_cost_gate_and_config_do_not_admit_missing_or_excessive_timing(self):
        self.assertTrue(cost_gate([1],[1],[1],[1])['admitted'])
        self.assertFalse(cost_gate([100],[1],[1],[1])['admitted'])
        for values in ([],[-1],[float('nan')]):
            with self.assertRaises(ValueError):cost_gate(values,[1],[1],[1])

    def test_legacy_model_entrypoint_cannot_silently_ignore_conditioning(self):
        _,new=self.pair()
        with self.assertRaises(ValueError):new(self.item['seed'],1)
        with self.assertRaises(ValueError):new.grow(self.item['seed'],1)

    def test_conditioned_evaluation_does_not_consume_training_or_global_rng(self):
        s=Session(self.meta());local=s.generator.get_state().clone();global_state=torch.get_rng_state().clone()
        a,fields=s.score(3,2);b,other=s.score(3,2)
        self.assertEqual(a,b)
        for key in fields:self.assertTrue((fields[key]==other[key]).all())
        self.assertTrue(torch.equal(local,s.generator.get_state()))
        self.assertTrue(torch.equal(global_state,torch.get_rng_state()))
