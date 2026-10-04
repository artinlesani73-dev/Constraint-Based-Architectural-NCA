from pathlib import Path
import json,sys,shutil
import numpy as np,torch
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G9-Training-Diagnosis-2026-10-04')
sys.path.insert(0,str(OUT/'source'))
from nca.paced_generation import full_origins
from access_ranking_one_sided import access_ranking
torch.set_num_threads(2)
records=json.loads((OUT/'gradient-records.json').read_text());checks=[];cache={}
for r in records:
 case=r['case']
 if case not in cache:
  with np.load(OUT/f'inputs/examples/{case}.npz',allow_pickle=False) as a:cache[case]=full_origins(a['target'].astype(bool))
 with np.load(OUT/f'gradient-snapshots/{r["model"]}-{case}-{r["step"]:02d}.npz',allow_pickle=False) as a:
  z=torch.from_numpy(a['logits'].copy()).requires_grad_(True);e=torch.from_numpy(a['fired'])[None,None];p=torch.from_numpy(a['positive'])[None,None];o=torch.from_numpy(cache[case])[None,None]
  loss=access_ranking(z,e,o,p,r['phase']);g=torch.autograd.grad(loss,z)[0].numpy()[0,0]
  A=a['fired']&a['positive']&cache[case]
  assert np.allclose(g[A],a['ranking'][A],rtol=1e-6,atol=1e-7)
  assert (g[~A]==0).all()
  assert np.isclose(float(loss.detach()),r['loss_terms']['ranking'],rtol=1e-6,atol=1e-6)
  combined=a['base']+g
  assert np.array_equal(combined[~A],a['base'][~A])
  assert (combined[A]<=a['base'][A]+1e-7).all()
 checks.append(dict(model=r['model'],case=case,step=r['step'],same_loss_value=True,same_advancing_gradient=True,other_scores_no_auxiliary_gradient=True,base_gradient_retained_elsewhere=True))
with (OUT/'one-sided-check.json').open('x') as f:json.dump(dict(checked=len(checks),passed=True,optimizer_updates=0,model_inference=0,checks=checks),f,indent=2)
shutil.copyfile(Path(__file__).with_name('access_ranking_one_sided.py'),OUT/'access_ranking_one_sided.py')
shutil.copyfile(__file__,OUT/'check-one-sided.py')
print('Verified one-sided semi-gradient on',len(checks),'saved snapshots.')

