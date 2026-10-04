from pathlib import Path
import sys,time,json
ROOT=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');sys.path.insert(0,str(ROOT));out=Path(__file__).resolve().parent
import numpy as np,torch
from reversible_repair import ConnectedRepair as Reversible
from nca.connected_repair import ConnectedRepair as Original
from nca.repair_training import perceive
from nca.repair_benchmark import load_example
from nca.experiments import RunStore,digest
root=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020';assert not RunStore(root.parent).verify(root.name)
rows=sorted([r for r in json.loads((root/'study.json').read_bytes())['examples'] if r['split']=='train'],key=lambda r:(r['case'],r['damage']))
selected=[next(r for r in rows if r['damage']==d) for d in ['intact','cube5','slab2']];records=[];torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
for row in selected:
 inputs,target=load_example(root,row);o=torch.from_numpy(inputs['occupancy'])[None,None];c=torch.from_numpy(inputs['context'])[None];legal=(c[:,:1]>0)&(c[:,1:2]>0);features=perceive(c);t=torch.from_numpy(target)[None,None]
 for label,cls in [('CGR1',Original),('RGR1',Reversible)]:
  torch.manual_seed(1201);model=cls();start=time.perf_counter()
  with torch.no_grad():res=model.rollout(o,features,legal,torch.Generator().manual_seed(2101),32,capture=True)
  elapsed=time.perf_counter()-start
  path=out/f'{label}-{row["damage"]}-untrained.npz'
  with path.open('xb') as f:np.savez_compressed(f,**{k:v.numpy() for k,v in res.items()})
  start=time.perf_counter();loss=model.rollout(o,features,legal,torch.Generator().manual_seed(2101),32,target=t)['loss'];loss.backward();backward=time.perf_counter()-start
  assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  records.append(dict(model=label,damage=row['damage'],case=row['case'],forward_capture_seconds=elapsed,forward_backward_seconds=backward,arrays_sha256=digest(path),occupied=int(res['field'].sum())))
  print(label,row['damage'],round(elapsed,3),round(backward,3),flush=True)
# Untrained high-birth stress, no learned-quality interpretation.
for label,cls in [('CGR1',Original),('RGR1',Reversible)]:
 torch.manual_seed(1201);model=cls()
 with torch.no_grad():
  model.last.bias[0]=2.;start=time.perf_counter();res=model.rollout(o,features,legal,torch.Generator().manual_seed(2101),32);elapsed=time.perf_counter()-start
 records.append(dict(model=label,stress='fixed_positive_birth_bias',forward_seconds=elapsed,occupied=int(res['field'].sum())))
(out/'timings.json').write_text(json.dumps(dict(records=records,threads=2,torch=str(torch.__version__),numpy=np.__version__,scope='Single CPU timings on3TRAIN examples and one artificial high-birth case. Untrained fresh weights; no optimizer updates or GPU extrapolation.'),indent=2),encoding='utf-8')
