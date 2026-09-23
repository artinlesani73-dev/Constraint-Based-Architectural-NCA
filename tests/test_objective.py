import unittest
from dataclasses import replace
import torch
from nca.losses import LossSpec,FAMILIES
from nca.objective import research_terms,weighted_total,REGULARIZERS
from test_losses import fixture

class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def make_result(self):
        ctx=fixture(2);p=(torch.rand(ctx.permitted.shape,generator=torch.Generator().manual_seed(31))*.6*ctx.permitted).requires_grad_()
        return research_terms(p[:,None],p,ctx,{'ch_structure':0},torch.zeros_like(ctx.facade),LossSpec(reach_hops=8))

    def test_complete_recipe_equals_explicit_per_scene_sum(self):
        r=self.make_result();fw={n:float(i+1) for i,n in enumerate(FAMILIES)};rw={n:.25 for n in REGULARIZERS}
        expected=sum(v.mean()*fw[n] for n,v in r['terms'].items())+sum(v.mean()*rw[n] for n,v in r['regularizers'].items())
        self.assertTrue(torch.equal(weighted_total(r,fw,rw),expected))
        self.assertNotIn('cantilever_historical',r['regularizers'])

    def test_missing_or_disabled_family_and_unknown_regularizer_refused(self):
        r=self.make_result();fw=dict.fromkeys(FAMILIES,1.);rw=dict.fromkeys(REGULARIZERS,0.)
        for bad in [{k:v for k,v in fw.items() if k!='support'},dict(fw,support=0.),dict(fw,support=float('nan'))]:
            with self.assertRaises(ValueError):weighted_total(r,bad,rw)
        with self.assertRaises(ValueError):weighted_total(r,fw,dict(rw,cantilever_historical=5.))

    def test_joint_facade_conflict_refused_before_training_reduction(self):
        ctx=fixture();ctx=replace(ctx,facade=ctx.coverage.clone());p=ctx.coverage.float()
        r=research_terms(p[:,None],p,ctx,{'ch_structure':0},torch.zeros_like(ctx.facade),LossSpec(max_mass_ratio=.05,reach_hops=8))
        with self.assertRaises(ValueError):weighted_total(r,dict.fromkeys(FAMILIES,1.),dict.fromkeys(REGULARIZERS,0.))

    def test_preclamp_hinge_saturates_without_straight_through(self):
        from nca.interventions import guidance_loss
        raw=torch.tensor([[[[-.1,.5,1.2]]]],requires_grad=True);guide=torch.ones_like(raw,dtype=torch.bool)
        value=guidance_loss(raw.clamp(0,1)[:,None],raw,guide,{'ch_structure':0},'hard_preclamp').sum()
        grad,=torch.autograd.grad(value,raw)
        self.assertTrue(torch.allclose(grad,torch.tensor([[[[-1/3,-1/3,0.]]]])))
