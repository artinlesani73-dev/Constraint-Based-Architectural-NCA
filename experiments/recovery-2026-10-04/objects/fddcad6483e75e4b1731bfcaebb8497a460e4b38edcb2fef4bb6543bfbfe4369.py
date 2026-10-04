from pathlib import Path
import json,sys,shutil,hashlib,time
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G11-R1-Prototype-2026-10-04-v2';OUT=BASE/'G11-Scheduling-Diagnosis-2026-10-04'
OUT.mkdir(exist_ok=False);shutil.copytree(OLD/'source',OUT/'source');sys.dont_write_bytecode=True
p=OUT/'source/g11_reservation.py';s=p.read_text()
s=s.replace('provenance=np.zeros_like(field,dtype=np.uint8);blocked=0','provenance=np.zeros_like(field,dtype=np.uint8);blocked=0;reasons=dict(quota=0,reservation=0,facade=0)')
s=s.replace("if candidate.sum()>cap or union.sum()>C or (union&contact).sum()/union.sum()>.15+1e-10:\n                blocked+=1;continue","if candidate.sum()>cap:\n                reasons['quota']+=1;blocked+=1;continue\n            if union.sum()>C:\n                reasons['reservation']+=1;blocked+=1;continue\n            if (union&contact).sum()/union.sum()>.15+1e-10:\n                reasons['facade']+=1;blocked+=1;continue")
s=s.replace("hidden=(hidden+output", """# Diagnostic only: check remaining frozen/new frontier fits without changing admission.
        fits=dict(frozen_offer=0,frozen_unoffered=0,new_offer=0,new_unoffered=0)
        after,_=eligibility(field,valid)
        for idx in np.flatnonzero(after):
            p=np.unravel_index(idx,valid.shape);reg=tuple(slice(int(v),int(v)+3) for v in p)
            test=field.copy();test[reg]=True;u=test|W
            if test.sum()>cap or u.sum()>C or (u&contact).sum()/u.sum()>.15+1e-10:continue
            origin='frozen' if eligible[p] else 'new'
            offer=bool(wv[p] or (fn[p] and qn[p]>.5))
            fits[origin+('_offer' if offer else '_unoffered')]+=1
        hidden=(hidden+output""")
s=s.replace('reserved_missing=int((W&~field).sum())))','reserved_missing=int((W&~field).sum()),reasons=reasons,fits=fits))')
p.write_text(s,encoding='utf-8')
sys.path.insert(0,str(OUT/'source'))
import numpy as np,torch
from g11_reservation import rollout
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.paced_generation import PacedNCA
from nca.repair_portable import read_portable
from nca.massing_targets import neighbors
from nca.massing_cases import target_context
from nca.contract import entrance_masks
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
model=PacedNCA();model.load_state_dict(read_portable(OLD/'model/checkpoint-0427.pt',json.loads((OLD/'model/identity.json').read_text()))['model']);model.eval()
scenes=json.loads((OLD/'scenes.json').read_text());config=json.loads((OLD/'config.json').read_text())
audit=json.loads((OLD/'audit.json').read_text());cases=[x['case'] for x in audit['stability'] if not x['G11-R1']['passed']]
shutil.copyfile(__file__,OUT/'run.py')
records=[]
for case in cases:
 folder=OLD/'cases'/case
 with np.load(folder/'context.npz') as a:c=a['condition'].copy()
 with np.load(folder/'witness.npz') as a:W=a['field'].copy()
 x=seed_inputs(c);scene=scenes[case.rsplit('-v',1)[0]];fields,domain,_=target_context(scene,config)
 contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool);face=neighbors(fields['existing'].astype(bool));ends=entrance_masks(scene)
 for e in scene['entrances']:
  if e['kind']=='facade':contact &= ~(ends[e['id']]&face&x['allowed'])
 out=rollout(model,torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],W,contact,64)
 with np.load(folder/'G11-R1-trajectory.npz') as a:assert np.array_equal(out['births'],a['births'][:64])
 (OUT/f'{case}.json').write_text(json.dumps(out['trace'],indent=2),encoding='utf-8')
 unused=[t for t in out['trace'][1:] if t['cap']>t['mass']]
 rec=dict(case=case,unused=sum(t['cap']-t['mass'] for t in unused),steps=len(unused),new_offer_fits=sum(t['fits']['new_offer']>0 for t in unused),unoffered_fits=sum(t['fits']['new_unoffered']+t['fits']['frozen_unoffered']>0 for t in unused),frozen_offer_fits=sum(t['fits']['frozen_offer']>0 for t in unused),exact_replay=True)
 records.append(rec);print(rec,flush=True)
(OUT/'result.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

