"""Diagnose saved NR5 fields; no inference, training, threshold tuning or admission."""
from pathlib import Path
import argparse,json,sys
from hashlib import sha256
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.massing_targets import evaluate_targets
from nca.repair_benchmark import repair_metrics
from nca.evaluation import flood_fill
from nca.experiments import write_once

def main(review,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);review=Path(review)
 source=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020'
 study=json.loads((source/'study.json').read_bytes());targets={t['case']:t for t in study['targets']}
 display=json.loads((ROOT/'deploy/static/repair/study.json').read_bytes())
 inputs={(c['case'],c['damage']):c['results'][0]['occupied_zyx'] for c in display['cases']}
 rows=[]
 for p in sorted((review/'observations').glob('*.json')):
  obs=json.loads(p.read_bytes());assert obs['split']=='validation'
  raw=p.with_suffix('.npz').read_bytes();assert sha256(raw).hexdigest()==obs['arrays_sha256']
  with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:
   field=a['field'].copy();prob=a['probability'].copy();assert np.array_equal(field,prob>.5)
  t=targets[obs['case']];context=json.loads((source/t['json']).read_bytes())
  with np.load(source/t['arrays'],allow_pickle=False) as a:
   target=a['target'].astype(bool);domain=a['domain'].copy();fields={k:a[k].copy() for k in ['permitted','existing','protected','support_boundary']}
  damaged=np.zeros_like(field);coords=np.asarray(inputs[(obs['case'],obs['damage'])]);damaged[tuple(coords.T)]=True
  report,masks=evaluate_targets(field,context['scene'],fields,domain);assert report==obs['metrics']['targets']
  measured=repair_metrics(field,target,damaged,domain,context['generation']['spec']['target_fraction'])
  assert all(obs['metrics'][k]==v for k,v in measured.items())
  excess=field&~target;missing=target&~field;detached=field&~masks['raw_reached']
  unsupported=field&~flood_fill(field|fields['support_boundary'],fields['support_boundary'])
  cf={};arrays=dict(raw=field,excess=excess,missing=missing,detached=detached,unsupported=unsupported,bulk_unreached=masks['bulk']&~masks['bulk_reached'])
  for label,value in [('remove_detached',field&~detached),('oracle_remove_excess',field&target),('oracle_restore_missing',field|target)]:
   r,_=evaluate_targets(value,context['scene'],fields,domain)
   cf[label]=dict(passed=r['contract_pass'],unmet=[k for k,v in r['family_pass'].items() if not v],changed=int((field!=value).sum()))
   arrays[label]=value
  row=dict(case=obs['case'],damage=obs['damage'],arrays_sha256=obs['arrays_sha256'],passed=report['contract_pass'],
   unmet=[k for k,v in report['family_pass'].items() if not v],raw_hits=report['raw_interface_hits'],bulk_hits=report['bulk_interface_hits'],
   detached=int(detached.sum()),detached_target=int((detached&target).sum()),detached_excess=int((detached&excess).sum()),
   unsupported=int(unsupported.sum()),unsupported_target=int((unsupported&target).sum()),bulk_unreached=int(arrays['bulk_unreached'].sum()),
   bulk_fraction=report['bulk_fraction'],excess=int(excess.sum()),missing=int(missing.sum()),counterfactuals=cf,
   detached_probabilities=prob[detached].tolist(),detached_zyx=np.argwhere(detached).tolist())
  key=p.stem;np.savez_compressed(out/(key+'.npz'),**arrays);row['diagnostic_arrays']=key+'.npz';row['diagnostic_sha256']=sha256((out/(key+'.npz')).read_bytes()).hexdigest();rows.append(row)
 assert len(rows)==27
 summary=dict(version='NR5_saved_failure_diagnosis_v1',cases=rows,raw_passes=sum(r['passed'] for r in rows),
  counterfactual_passes={k:sum(r['counterfactuals'][k]['passed'] for r in rows) for k in cf},
  scope='Fixed saved27 validation outputs at32steps and threshold0.5. Counterfactuals are diagnostic derived fields, not learned results. Oracle operations use reference targets and cannot be deployed.')
 write_once(out/'diagnosis.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
 for r in rows:
  if not r['passed']:print(json.dumps({k:v for k,v in r.items() if k not in ['detached_probabilities','detached_zyx','diagnostic_sha256','arrays_sha256']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.review,a.output)
