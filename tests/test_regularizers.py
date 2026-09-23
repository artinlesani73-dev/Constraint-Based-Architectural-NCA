import ast,json,unittest
from pathlib import Path
import torch
from nca.regularizers import density_binary,total_variation,cantilever_historical,cantilever_boundary,regularizer_terms

class RegularizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        p=Path(__file__).resolve().parents[1]/'notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb'
        source=''.join(json.loads(p.read_text())['cells'][19]['source']);tree=ast.parse(source)
        cls.original={}
        for name in ('DensityPenalty','TotalVariation3D','CantileverLoss'):
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            space={'torch':torch,'nn':torch.nn,'F':torch.nn.functional}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),space);cls.original[name]=space[name]()

    def test_notebook_value_and_gradient_parity(self):
        p=torch.rand((2,7,6,5),generator=torch.Generator().manual_seed(12),dtype=torch.float64,requires_grad=True)
        for name,fn in [('DensityPenalty',density_binary),('TotalVariation3D',total_variation),('CantileverLoss',cantilever_historical)]:
            old=self.original[name](p);new=fn(p).mean()
            self.assertTrue(torch.allclose(old,new,atol=1e-14,rtol=1e-14),name)
            go,=torch.autograd.grad(old,p,retain_graph=True);gn,=torch.autograd.grad(new,p,retain_graph=True)
            self.assertTrue(torch.allclose(go,gn,atol=1e-14,rtol=1e-14),name)

    def test_boundary_handles_ground_and_floating_without_wraparound(self):
        p=torch.zeros((1,7,5,5));p[:,0,2,2]=1
        boundary=torch.zeros_like(p,dtype=torch.bool)
        self.assertGreater(float(cantilever_boundary(p,boundary)[0]),0)
        boundary[:,0,2,2]=True;self.assertEqual(float(cantilever_boundary(p,boundary)[0]),0)
        p.zero_();p[:,-1,2,2]=1;self.assertGreater(float(cantilever_boundary(p,boundary)[0]),0)
        boundary[:,-2,2,2]=True;self.assertEqual(float(cantilever_boundary(p,boundary)[0]),0)
        self.assertEqual(float(cantilever_boundary(torch.zeros_like(p),boundary)[0]),0)

    def test_batch_equals_separate_and_all_gradients_finite(self):
        p=torch.rand((2,7,5,5),generator=torch.Generator().manual_seed(8),requires_grad=True)
        boundary=torch.zeros_like(p,dtype=torch.bool);boundary[:,0]=True
        both=regularizer_terms(p,boundary)
        for name,value in both.items():
            grad,=torch.autograd.grad(value.mean(),p,retain_graph=True);self.assertTrue(torch.isfinite(grad).all())
            for i in range(2):self.assertTrue(torch.equal(value[i:i+1],regularizer_terms(p[i:i+1],boundary[i:i+1])[name]))

    def test_boundary_finite_difference(self):
        p=(torch.rand((1,5,4,4),generator=torch.Generator().manual_seed(4),dtype=torch.float64)*.8+.1).requires_grad_()
        fixed=torch.zeros_like(p,dtype=torch.bool);fixed[:,0]=True
        direction=torch.randn(p.shape,generator=torch.Generator().manual_seed(5),dtype=p.dtype);direction/=direction.norm()
        g,=torch.autograd.grad(cantilever_boundary(p,fixed).sum(),p);h=1e-6
        fd=(cantilever_boundary(p.detach()+h*direction,fixed)-cantilever_boundary(p.detach()-h*direction,fixed))/(2*h)
        self.assertAlmostEqual(float((g*direction).sum()),float(fd[0]),delta=1e-8)

    def test_density_is_binarization_not_density_cap(self):
        p=torch.full((1,3,3,3),.5)
        self.assertEqual(float(density_binary(p)[0]),.25)
        self.assertEqual(float(density_binary(torch.ones_like(p))[0]),0)
        self.assertEqual(float(density_binary(torch.zeros_like(p))[0]),0)
