from pathlib import Path
import json,hashlib,zipfile,shutil
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G9-Training-Diagnosis-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
r=json.loads((OUT/'result.json').read_text());allcases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))];assert len(allcases)==90
aggregation={}
for model in ['G8','G9']:
 aggregation[model]={}
 for group in ['shared_original_train','added_g7_train']:
  cases=[c for c in allcases if c['model']==model and c['group']==group]
  steps=[s for c in cases for s in c['steps'] if not s['seed_phase'] and s['effective_cap']<s['cap']]
  wasted=[s for s in steps if s['unused_step_allowance']>0]
  categories={}
  for name,select in [('no_eligible_cube',lambda s:s['eligible']==0),('no_above_threshold',lambda s:s['eligible']>0 and s['above_threshold']==0),('firing_missed_above_threshold',lambda s:s['above_threshold']>0 and s['offered']==0),('offers_exhausted_no_rejection',lambda s:s['offered']>0 and s['allowance_rejected']==0),('whole_cube_allowance_rejections',lambda s:s['allowance_rejected']>0)]:
   part=[s for s in wasted if select(s)];categories[name]=dict(steps=len(part),unused_voxels=sum(s['unused_step_allowance'] for s in part))
  assert sum(v['steps'] for v in categories.values())==len(wasted)
  progress=[s for c in cases for s in c['steps'] if not s['seed_phase'] and not s['connected_before'] and s['mass_before']<s['cap'] and s['positive_progress'] and s['positive_other']]
  aggregation[model][group]=dict(full_quota_steps=len(steps),allowance_total=sum(s['quota'] for s in steps),added_total=sum(s['added'] for s in steps),unused_total=sum(s['unused_step_allowance'] for s in steps),unused_categories=categories,matched_teacher_positive_progress_steps=len(progress),other_positive_mean_exceeds_progress=sum(s['other_positive_mean_q']>s['progress_positive_mean_q'] for s in progress),progress_mean_below_threshold=sum(s['progress_positive_mean_q']<=.5 for s in progress))
save('allowance-analysis.json',aggregation)
print(json.dumps(dict(summary=r['summary'],training=r['training'],allowance=aggregation),indent=2))
shutil.copyfile(__file__,OUT/'summarize-diagnosis.py')
