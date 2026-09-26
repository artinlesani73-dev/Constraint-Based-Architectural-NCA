"""NR3/NR4 saved-field diagnosis and display export; no model inference."""
from pathlib import Path
import json,sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.experiments import digest
from nca.massing_targets import evaluate_targets
from nca.repair_benchmark import repair_metrics

def build(review,output):
 review=Path(review);out=Path(output);out.mkdir(parents=True,exist_ok=True)
 base=json.loads((ROOT/'deploy/static/repair/study.json').read_text(encoding='utf-8'))
 source=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020';study=json.loads((source/'study.json').read_text());targets={t['case']:t for t in study['targets']}
 observations={}
 for p in (review/'observations').glob('*.json'):
  row=json.loads(p.read_text());assert row['split']=='validation';assert digest(p.with_suffix('.npz'))==row['arrays_sha256'];observations[(row['case'],row['damage'])]=(row,p)
 assert len(observations)==27
 diagnoses=[]
 for case in base['cases']:
  obs,path=observations.pop((case['case'],case['damage']))
  with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:field=a['field'];assert np.array_equal(field,a['probability']>.5)
  t=targets[case['case']];context=json.loads((source/t['json']).read_text())
  with np.load(source/t['arrays'],allow_pickle=False) as a:
   target=a['target'].astype(bool);domain=a['domain'];report,masks=evaluate_targets(field,context['scene'],{k:a[k] for k in ['permitted','existing','protected','support_boundary']},domain)
  assert report==obs['metrics']['targets']
  damaged=np.zeros_like(field);coords=np.array(case['results'][0]['occupied_zyx']);damaged[tuple(coords.T)]=True
  metrics=repair_metrics(field,target,damaged,domain,context['generation']['spec']['target_fraction']);assert all(obs['metrics'][k]==v for k,v in metrics.items())
  excess=field&~target;unreached=field&~masks['raw_reached']
  diagnosis={'unmet':[k for k,v in report['family_pass'].items() if not v],'raw_unreached':int(unreached.sum()),'unreached_excess':int((unreached&excess).sum()),'unreached_target':int((unreached&target).sum()),'bulk_unreached':int((masks['bulk']&~masks['bulk_reached']).sum()),'raw_hits':report['raw_interface_hits'],'bulk_hits':report['bulk_interface_hits'],'bulk_fraction':report['bulk_fraction'],'unsupported_voxels':report['support']['unsupported_voxels']}
  new={'occupied_zyx':np.argwhere(field).tolist(),'added_zyx':np.argwhere(excess).tolist(),'missing_zyx':np.argwhere(target&~field).tolist(),'isolated_zyx':np.argwhere(unreached).tolist(),'metrics':obs['metrics']}
  old=case['results'];case['results']=[old[0],old[3],old[2],new,old[1]];case['diagnosis']=diagnosis;case['nr4_arrays_sha256']=obs['arrays_sha256']
  case['label']=case['label'].replace('slab2','slab damage').replace('cube5','cube damage')+(' / CHECKS UNMET' if diagnosis['unmet'] else '')
  if diagnosis['unmet']:diagnoses.append({'case':case['case'],'damage':case['damage'],**diagnosis})
 assert not observations and len(diagnoses)==8
 base['version']='NR34_comparison_v1';base['nr4_run_id']='20260926T083017Z_51b29cc254ff';base['nr4_checkpoint_sha256']=json.loads((review/'sealed-model.json').read_text())['checkpoint_sha256']
 (out/'study.json').write_text(json.dumps(base,separators=(',',':')),encoding='utf-8');(out/'diagnosis.json').write_text(json.dumps(diagnoses,indent=2),encoding='utf-8')
 print('27 exported comparisons,8 diagnosed failures; archived metrics reproduced, no inference.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.review,a.output)
