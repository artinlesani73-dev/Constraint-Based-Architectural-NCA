from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G6-Paced-Growth-2026-10-04';ROOT=OUT/'package';AUDIT=BASE/'G6-Objective-Audit-2026-10-04';sys.path.insert(0,str(ROOT))
from nca.paced_generation import PacedNCA
from nca.generation_package import verify
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.repair_portable import read_portable
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
_,data=verify(ROOT);row=data['rows'][0]
with np.load(ROOT/row['arrays'],allow_pickle=False) as a:c=a['condition'].copy()
identity=json.loads((AUDIT/'inputs/identity.json').read_text());checkpoint=read_portable(AUDIT/'inputs/checkpoint-0256.pt',identity)
model=PacedNCA().float();model.load_state_dict(checkpoint['model']);model.eval();x=seed_inputs(c)
with torch.no_grad():r=model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(2101),128,capture=True)
with np.load(AUDIT/f'paced-rollouts/{row["id"]}.npz',allow_pickle=False) as a:
    assert np.array_equal(r['field'].numpy()[0,0],a['field128']) and np.array_equal(r['admission_counts'].numpy(),a['admission_counts'])
    first64=x['occupancy'].astype(bool)|r['births'][:64].numpy()[:,0,0].any(0)
    assert np.array_equal(first64,a['field64'])
counts=r['admission_counts'].numpy();caps=r['step_ceilings'].numpy();C=int(r['budget'][2]);K=int(r['quota'])
assert np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))
assert (counts[:,0]+counts[:,6]<=caps).all()
old=(ROOT/'nca/block_generation.py').read_text();new=(ROOT/'nca/paced_generation.py').read_text()
assert old[old.index('def block_loss('):old.index('\nclass BlockNCA')]==new[new.index('def block_loss('):new.index('\nclass PacedNCA')]
result=dict(package_matches_independent_paced_rollout=True,case=row['id'],horizons=[64,128],field_and_all128counts_exact=True,quota=K,loss_function_source_exactly_equal_g4=True,no_new_parameters=True,optimizer_updates=0,development_or_reserved_access=False)
with (OUT/'implementation-checks.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
