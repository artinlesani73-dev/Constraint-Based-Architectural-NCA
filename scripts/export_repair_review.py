"""Export saved D078 fields for Studio. No model loading, training or inference."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.experiments import read_json,digest
from nca.repair_benchmark import load_example,closing_repair,repair_metrics

def main(review,output):
 review=Path(review);source=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020'
 study=read_json(source/'study.json');examples={(r['case'],r['damage']):r for r in study['examples'] if r['split']=='validation'};targets={r['case']:r for r in study['targets']};cases=[]
 for path in sorted((review/'observations').glob('*.json')):
  obs=read_json(path);assert obs['split']=='validation';assert digest(path.with_suffix('.npz'))==obs['arrays_sha256']
  row=examples.pop((obs['case'],obs['damage']));inputs,target=load_example(source,row,split='validation');target=target.astype(bool);damaged=inputs['occupancy'].astype(bool)
  t=targets[row['case']];context=read_json(source/t['json'])
  with np.load(source/t['arrays'],allow_pickle=False) as a:domain=a['domain'];permitted=a['permitted']
  with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:learned=a['field'];assert np.array_equal(learned,a['probability']>.5)
  closed=closing_repair(damaged,domain,permitted);results=[]
  for field,metrics in [(damaged,obs['baselines']['unchanged']),(closed,obs['baselines']['closing3']),(learned,obs['metrics']),(target,None)]:
   check=repair_metrics(field,target,damaged,domain,context['generation']['spec']['target_fraction'])
   if metrics is not None:
    assert all(metrics[k]==v for k,v in check.items()),'Export geometry differs from recorded scores'
   else:metrics={**check,'targets':context['targets']}
   assert int(field.sum())==metrics['targets']['occupied_voxels']
   results.append({'occupied_zyx':np.argwhere(field).tolist(),'added_zyx':np.argwhere(field&~target).tolist(),'missing_zyx':np.argwhere(target&~field).tolist(),'metrics':metrics})
  fraction=context['generation']['spec']['target_fraction'];label=f"{len(cases)+1:02d} / {fraction:.0%} volume / {row['case'].split('__')[-1]} / {row['damage']}"
  cases.append({'case':row['case'],'damage':row['damage'],'label':label,'scene':context['scene'],'results':results,'observation_sha256':digest(path),'arrays_sha256':obs['arrays_sha256']})
 assert len(cases)==27 and not examples
 data={'version':'NR3_review_view_v1','run_id':'20260926T073942Z_47ef54dc1d67','checkpoint_sha256':read_json(review/'sealed-model.json')['checkpoint_sha256'],'cases':cases}
 Path(output).write_text(json.dumps(data,separators=(',',':')),encoding='utf-8');print('Exported 27 complete saved cases; geometry matches recorded metrics.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.review,a.output)
