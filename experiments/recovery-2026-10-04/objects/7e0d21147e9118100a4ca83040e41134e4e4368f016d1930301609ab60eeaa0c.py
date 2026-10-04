"""Frozen CGR2 final-checkpoint development review; no training or TEST."""
from pathlib import Path
import sys,json,zipfile,hashlib,io,shutil,time
ROOT=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from nca.bulk_repair import ConnectedRepair,SETTINGS,LOSS,VERSION
from nca.repair_training import perceive
from nca.repair_benchmark import load_example,repair_metrics
from nca.massing_targets import evaluate_targets
from nca.experiments import digest,write_once,snapshot_source,RunStore
from nca.recovery import metadata_hash

def main(archive,output):
 archive=Path(archive);out=Path(output);out.mkdir(parents=True,exist_ok=False)
 receipt=json.loads(archive.with_suffix('.receipt.json').read_bytes());assert digest(archive)==receipt['sha256']
 with zipfile.ZipFile(archive) as z:
  m=json.loads(z.read('evidence-manifest.json'));assert len(m)==receipt['files'];assert len(z.namelist())==len(set(z.namelist()));assert set(z.namelist())==set(m)|{'evidence-manifest.json'}
  for n,h in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
  result=json.loads(z.read('result.json'));request=result['request'];assert result['status']=='completed' and result['worker']['completed']==256
  assert request['settings']==SETTINGS and request['seed']==1201 and request['updates']==256 and not request['cpu_rehearsal']
  manifest='b3d15f8625637e75ba4b3a59aea8f243facb5d572b0653436acba90afe23b22a';assert request['manifest_sha256']==manifest
  raw=z.read('worker/checkpoint-0256.pt');meta=json.loads(z.read('worker/checkpoint-0256.json'));assert hashlib.sha256(raw).hexdigest()==meta['sha256'] and len(raw)==meta['bytes']
  payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);identity=payload['identity']
  assert payload['completed']==256 and identity['seed']==1201 and identity['model_semantics']==VERSION and identity['objective']==LOSS and identity['train_steps']==32
  assert identity['experiment']=={'study':SETTINGS,'manifest_sha256':manifest} and identity['runtime']['device']=='cuda:0'
  assert metadata_hash(identity)==meta['identity_sha256'] and payload['sampler']['consumed']==256 and len(payload['trace'])==256
  assert all(int(v['step'])==256 for v in payload['optimizer']['state'].values())
  write_once(out/'sealed-model.json',dict(checkpoint_sha256=meta['sha256'],identity=identity))
 for p in [archive,archive.with_suffix('.receipt.json')]:shutil.copy2(p,out/p.name);assert digest(out/p.name)==digest(p)
 write_once(out/'imports.json',dict(receipt=receipt,result=result,execution_provenance='User supplied completed run; no new compute approval inferred.'))
 snapshot_source(ROOT,out/'source.zip');shutil.copy2(__file__,out/'review-script.py')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 model=ConnectedRepair().float();model.load_state_dict(payload['model'],strict=True);model.eval()
 source=ROOT/'.local-artifacts/runs'/SETTINGS['dataset_run'];assert not RunStore(source.parent).verify(source.name)
 study=json.loads((source/'study.json').read_bytes());targets={t['case']:t for t in study['targets']};examples=[r for r in study['examples'] if r['split']=='validation'];assert len(examples)==27
 write_once(out/'request.json',dict(checkpoint=256,steps=32,firing=2101,split='validation',rows=27,torch=str(torch.__version__),numpy=np.__version__,method='Accepted constructive occupancy, no cleanup; enforced preservation/attachment.'))
 rows=[];start=time.monotonic();(out/'observations').mkdir()
 for i,row in enumerate(examples):
  inputs,target=load_example(source,row,split='validation');o=torch.from_numpy(inputs['occupancy'])[None,None];context=torch.from_numpy(inputs['context'])[None];allowed=(context[:,:1]>0)&(context[:,1:2]>0)
  with torch.no_grad():r=model.rollout(o,perceive(context),allowed,torch.Generator().manual_seed(2101),32,capture=True)
  field=r['field'].numpy()[0,0];t=targets[row['case']];ctx=json.loads((source/t['json']).read_bytes())
  with np.load(source/t['arrays'],allow_pickle=False) as a:domain=a['domain'];fields={k:a[k] for k in ['permitted','existing','protected','support_boundary']}
  report,masks=evaluate_targets(field,ctx['scene'],fields,domain);metrics=repair_metrics(field,target.astype(bool),inputs['occupancy'].astype(bool),domain,ctx['generation']['spec']['target_fraction']);metrics['targets']=report
  detached=field&~masks['raw_reached'];metrics['detached_correct']=int((detached&target.astype(bool)).sum());metrics['detached_excess']=int((detached&~target.astype(bool)).sum())
  p=out/'observations'/f'{i:02d}.npz'
  with p.open('xb') as f:np.savez_compressed(f,**{k:v.detach().cpu().numpy() for k,v in r.items()})
  record=dict(case=row['case'],damage=row['damage'],split='validation',metrics=metrics,baselines=row['metrics'],arrays_sha256=digest(p));write_once(p.with_suffix('.json'),record);rows.append(record)
  print(i+1,row['damage'],round(metrics['iou'],4),report['contract_pass'],flush=True)
 summary={}
 for group in ['all','damaged','intact']:
  subset=[r for r in rows if group=='all' or (r['damage']=='intact')==(group=='intact')];summary[group]={}
  for method in ['model','unchanged','closing3']:
   ms=[r['metrics'] if method=='model' else r['baselines'][method] for r in subset]
   summary[group][method]=dict(n=len(ms),median_iou=float(np.median([x['iou'] for x in ms])),all_nine_pass=sum(x['targets']['contract_pass'] for x in ms),median_absolute_request_error_cells=float(np.median([abs(x['request_error_cells']) for x in ms])),**{key:sum(x[key] for x in ms) for key in ['recovered_cells','false_positive_cells','surviving_cells_removed']})
 d=summary['damaged']['model'];intact=[r['metrics'] for r in rows if r['damage']=='intact']
 checks=dict(intact_overlap=all(x['iou']>=.99 for x in intact),intact_validity=all(x['targets']['contract_pass'] for x in intact),damaged_validity=d['all_nine_pass']>=17,damaged_overlap=d['median_iou']>=.9705768039313023,damaged_excess=d['false_positive_cells']<=325,damaged_recovery=d['recovered_cells']>=1945,preservation=summary['all']['model']['surviving_cells_removed']==0,volume_error=d['median_absolute_request_error_cells']<=19)
 final=dict(status='completed',summary=summary,checks=checks,all_conditions_met=all(checks.values()),seconds=time.monotonic()-start,family_pass_counts={k:sum(r['metrics']['targets']['family_pass'][k] for r in rows) for k in rows[0]['metrics']['targets']['family_pass']},scope='Repeated development cases; one seed. Structured accepted output, not unconstrained proposal. No TEST or automatic admission.')
 write_once(out/'result.json',final);print(json.dumps(final,indent=2))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
