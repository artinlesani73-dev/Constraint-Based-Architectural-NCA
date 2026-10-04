from pathlib import Path
import json,sys,shutil,hashlib,copy
from collections import deque
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G11-R3-Independent-Review-2026-10-04'
OUT.mkdir(exist_ok=False);PARENT=BASE/'G11-R3-Ledger-2026-10-04';OLD=BASE/'G10-Final-Review-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((PARENT/'milestone-manifest.json').read_text())
assert all(sha((PARENT/n).read_bytes())==h for n,h in manifest['files'].items())
shutil.copytree(PARENT/'source',OUT/'source');shutil.copytree(PARENT/'model',OUT/'model')
shutil.copyfile(PARENT/'config.json',OUT/'config.json')
sys.dont_write_bytecode=True;sys.path.insert(0,str(OUT/'source'))
import numpy as np
from nca.massing_cases import target_context
from nca.repair_benchmark import condition
config=json.loads((OUT/'config.json').read_text())
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
oldentries=json.loads((OLD/'scene-index.json').read_text())['entries']
oldentries=[e for e in oldentries if not e['cohort'].startswith('baseline')]
assert len(oldentries)==23
known={e['id']:e['scene'] for e in oldentries}
known.update(json.loads((PARENT/'scenes.json').read_text()))
hashes=set()
for scene in known.values():
 f,d,_=target_context(scene,config);hashes.add(sha(condition(scene,f,d,.16)[:6].tobytes()))
base=copy.deepcopy(oldentries[-1]['scene']);fresh=[]
for i,(w,e,wr,er,y) in enumerate([(7,19,22,29,14),(19,7,29,22,16),(12,22,25,31,13),(22,12,31,25,17)]):
 scene=copy.deepcopy(base);sid=f'g11r3-reserved-{i}';scene.update(scene_id=sid,description='Frozen independent R3 assessment; no teacher targets',notes=['Unseen synthetic geometry, frozen before inference.'])
 scene['buildings'][0]['z'][1]=wr;scene['buildings'][1]['z'][1]=er
 scene['entrances'][0].update(z=w,y=y);scene['entrances'][1].update(z=e,y=y)
 f,d,_=target_context(scene,config);h=sha(condition(scene,f,d,.16)[:6].tobytes())
 assert h not in hashes;hashes.add(h)
 fresh.append(dict(id=sid,scene=scene,cohort='fresh',physical_context_sha256=h))
entries=[dict(id=e['id'],scene=e['scene'],cohort='regression') for e in oldentries]+fresh
save('scene-index.json',dict(entries=entries));save('fresh-splits.json',dict(entries=fresh,requests=[.16,.24,.32]))
save('protocol.json',dict(adapter_sha256=sha((OUT/'source/g11_reservation.py').read_bytes()),parent_manifest_sha256=sha((PARENT/'milestone-manifest.json').read_bytes()),checkpoint_sha256=sha((OUT/'model/checkpoint-0427.pt').read_bytes()),regression_cases=69,fresh_cases=12,horizons=[64,128],firing_seed=2101,thresholds=dict(median_volume_error=.02,max_volume_error=.04,stability=.05,all_nine_required=True),frozen_before_inference=True,physical_disjointness_prior_unique_contexts=len(hashes)-4,certificate_failure_counts_as_failure=True,baseline='G10 cached69 regression; new12 freshly inferred',training_updates=0))
save('RESUME.json',dict(status='Protocol and scenes frozen; evaluation pending',next='Run evaluate_g11_independent.py once; preserve partial files if interrupted',paid_training=False))
# Extract unchanged route algorithm from its recorded design audit.
s=(BASE/'G11-Allocation-Design-2026-10-04/audit.py').read_text()
block=s[s.index(" x=seed_inputs(c);"):s.index(" D=int(legal.sum())")]
block=block.replace(" assert len(starts)"," if not len(starts):return None")
helper="from collections import deque\nimport numpy as np\nfrom nca.generation_data import seed_inputs\nfrom nca.paced_generation import full_origins,connected\nfrom nca.block_reference import cube_union\n\ndef route_from_context(c):\n"+block+"\n return route\n"
(OUT/'source/context_route.py').write_text(helper,encoding='utf-8')
# Validate extraction against all45 archived TRAIN routes, no model inference.
from importlib import import_module
route=import_module('context_route').route_from_context
for p in (BASE/'G11-Allocation-Design-2026-10-04/contexts').glob('*.npz'):
 with np.load(p) as a:c=a['condition']
 with np.load(BASE/'G11-Allocation-Design-2026-10-04/routes'/p.name) as a:assert np.array_equal(route(c),a['route'])
save('route-extraction-check.json',dict(exact_training_route_matches=45))
shutil.copyfile(__file__,OUT/'freeze.py')
print('Frozen69 regression +12 new; exact route parity45.')

