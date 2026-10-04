from pathlib import Path
import sys,json
import numpy as np
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G11-R1-Prototype-2026-10-04-v2')
sys.dont_write_bytecode=True;sys.path.insert(0,str(OUT/'source'))
from nca.massing_targets import evaluate_targets
from nca.massing_cases import target_context
from nca.paced_generation import full_origins,connected
from g11_reservation import witness
case='g1-aligned-y0-v16';scene=json.loads((OUT/'scenes.json').read_text())[case.rsplit('-v',1)[0]]
fields,domain,_=target_context(scene,json.loads((OUT/'config.json').read_text()))
with np.load(OUT/'cases'/case/'witness.npz') as a:W=a['field'];route=a['route']
score,_=evaluate_targets(W,scene,fields,domain);assert score['contract_pass']
small,trace=witness(route,domain,scene,fields,domain,26)
assert small.sum()>26 # must fail the separate certificate gate, never claim impossible geometry
cut=W.copy();xs=np.flatnonzero(W.any((0,1)));cut[:,:,xs[len(xs)//2]]=False
cs,_=evaluate_targets(cut,scene,fields,domain)
assert not cs['family_pass']['access'] and not connected(cut)
blocked=domain.copy();blocked[:,:,xs[len(xs)//2]]=False
assert (W&~blocked).any() # reused witness rejected by legality certificate
missing=full_origins(blocked);assert not missing[:,:,max(0,xs[len(xs)//2]-2):xs[len(xs)//2]+1].any()
result=dict(valid_certificate=True,cap_shortage_rejected=True,critical_bridge_cut_detected=True,blocked_plane_invalidates_route=True,limitations='Boundary checks on certificate predicates, not proof of planner completeness')
(OUT/'boundary-checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
import shutil;shutil.copyfile(__file__,OUT/'boundary-checks.py')
print(result)

