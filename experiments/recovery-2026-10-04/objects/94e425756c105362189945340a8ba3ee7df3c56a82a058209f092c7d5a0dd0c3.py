from pathlib import Path
import sys,json,zipfile,hashlib,io,shutil,time,importlib.util
ROOT=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA');sys.path.insert(0,str(ROOT))
import numpy as np,torch
from nca.connected_repair import ConnectedRepair as Control,VERSION as CV,LOSS as CL
from nca.recovery import tree_equal,metadata_hash
from nca.repair_benchmark import load_example,repair_metrics
from nca.repair_training import perceive
from nca.massing_targets import evaluate_targets
from nca.experiments import digest,write_once,RunStore,snapshot_source
out=Path('C:/Users/artin/Documents/Codex/outputs/RGR1-Paired-Review-2026-10-03');out.mkdir(exist_ok=False);(out/'observations').mkdir()
a=Path('C:/Users/artin/Downloads/20261003T142251Z_20c919d80cee.zip');receipt=json.loads(a.with_suffix('.receipt.json').read_bytes());assert digest(a)==receipt['sha256']
package=Path('C:/Users/artin/Documents/Codex/outputs/RGR1-Paired-Comparison/NCA-RGR1-Paired-Package.zip')
with zipfile.ZipFile(package) as z:
 raw=z.read('manifest.json');manifest=hashlib.sha256(raw).hexdigest();assert manifest=='a847c631cdf7b48b303b9d22be93f8e8a5759cd685db17797979cef0241b1f23';pm=json.loads(raw)
 for n,h in pm['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 (out/'reversible_repair.py').write_bytes(z.read('nca/reversible_repair.py'))
 plan=json.loads(z.read('paired-plan.json'))
 for n in ['nca/connected_repair.py','nca/repair_training.py','nca/repair_portable.py']:assert (ROOT/n).read_bytes()==z.read(n)
spec=importlib.util.spec_from_file_location('candidate',out/'reversible_repair.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
models={};payloads={};initials={};identities={}
with zipfile.ZipFile(a) as z:
 m=json.loads(z.read('evidence-manifest.json'));assert len(m)==receipt['files'] and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m)|{'evidence-manifest.json'}
 for n,h in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 result=json.loads(z.read('result.json'));assert result['status']=='completed' and result['request']['manifest_sha256']==manifest and result['request']['updates_per_arm']==256 and not result['request']['cpu_rehearsal']
 for label,cls,version,loss in [('CGR1',Control,CV,CL),('RGR1',candidate.ConnectedRepair,candidate.VERSION,candidate.LOSS)]:
  for number in [0,256]:
   raw=z.read(f'worker/{label}/checkpoint-{number:04d}.pt');meta=json.loads(z.read(f'worker/{label}/checkpoint-{number:04d}.json'));assert hashlib.sha256(raw).hexdigest()==meta['sha256'] and len(raw)==meta['bytes']
   p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);ident=p['identity'];assert metadata_hash(ident)==meta['identity_sha256']
   assert p['completed']==p['sampler']['consumed']==len(p['trace'])==number and ident['model_semantics']==version and ident['objective']==loss and ident['seed']==1201 and ident['train_steps']==32
   assert ident['experiment']=={'paired_plan':plan,'arm':label,'manifest_sha256':manifest}
   assert all(ident['runtime'][k]==v for k,v in {'device':'cuda:0','gpu_name':'Tesla T4','torch':'2.11.0+cu130','cuda_build':'13.0','cudnn':92700}.items())
   if number==0:initials[label]=p
   else:payloads[label]=p;identities[label]=ident
  assert all(int(v['step'])==256 for v in p['optimizer']['state'].values())
  for i,t in enumerate(p['trace'],1):
   assert t['update']==i
   saved=json.loads(z.read(f'worker/{label}/update-{i:04d}.json'));assert all(saved[k]==v for k,v in t.items())
  model=cls().float();model.load_state_dict(p['model']);model.eval();models[label]=model
 assert tree_equal(initials['CGR1']['model'],initials['RGR1']['model'])
 assert [t['row_index'] for t in payloads['CGR1']['trace']]==[t['row_index'] for t in payloads['RGR1']['trace']]
 assert tree_equal(payloads['CGR1']['rng'],payloads['RGR1']['rng'])
 assert identities['CGR1']['ordered_training_rows']==identities['RGR1']['ordered_training_rows']
for p in [a,a.with_suffix('.receipt.json')]:shutil.copy2(p,out/p.name);assert digest(p)==digest(out/p.name)
write_once(out/'imports.json',dict(receipt=receipt,result=result,identities=identities,independent_pairing_verified=True));shutil.copy2(__file__,out/'review-script.py');snapshot_source(ROOT,out/'source.zip')
source=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020';assert not RunStore(source.parent).verify(source.name)
study=json.loads((source/'study.json').read_bytes());targets={t['case']:t for t in study['targets']};rows=[r for r in study['examples'] if r['split']=='validation'];assert len(rows)==27
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);records=[]
write_once(out/'request.json',dict(split='validation',rows=27,checkpoints=256,steps=32,firing_seed=2101,device='cpu',dtype='float32',torch=str(torch.__version__),numpy=np.__version__,cleanup='RGR1 built-in anchor projection only; no postprocessing or tuning'))
for i,row in enumerate(rows):
 inputs,target=load_example(source,row,split='validation');o=torch.from_numpy(inputs['occupancy'])[None,None];c=torch.from_numpy(inputs['context'])[None];allowed=(c[:,:1]>0)&(c[:,1:2]>0);t=targets[row['case']];ctx=json.loads((source/t['json']).read_bytes())
 with np.load(source/t['arrays'],allow_pickle=False) as p:domain=p['domain'];fields={k:p[k] for k in ['permitted','existing','protected','support_boundary']}
 def evaluate(field):
  report,_=evaluate_targets(field,ctx['scene'],fields,domain);metrics=repair_metrics(field,target.astype(bool),inputs['occupancy'].astype(bool),domain,ctx['generation']['spec']['target_fraction']);metrics['targets']=report;return metrics
 for label,model in models.items():
  start=time.monotonic()
  with torch.no_grad():r=model.rollout(o,perceive(c),allowed,torch.Generator().manual_seed(2101),32,capture=True)
  seconds=time.monotonic()-start;values={k:v.detach().cpu().numpy() for k,v in r.items()};metrics=evaluate(values['field'][0,0]);extra={}
  if label=='RGR1':
   correct=target.astype(bool)[None,None,None];wrong=~correct
   for kind in ['direct_removed','projection_removed']:
    extra[kind+'_correct_events']=int((values[kind]&correct).sum());extra[kind+'_wrong_events']=int((values[kind]&wrong).sum())
   total_removed=values['direct_removed']|values['projection_removed'];extra['birth_events']=int(values['births'].sum());extra['cells_born_more_than_once']=int((values['births'].sum(0)>1).sum());extra['input_removed_events']=int((total_removed&inputs['occupancy'].astype(bool)[None,None,None]).sum())
   extra['pre_projection_final']=evaluate(values['candidates'][-1,0,0])
  p=out/'observations'/f'{i:02d}-{label}.npz'
  with p.open('xb') as f:np.savez_compressed(f,**values)
  record=dict(case=row['case'],damage=row['damage'],model=label,metrics=metrics,diagnostics=extra,seconds=seconds,baselines=row['metrics'],arrays_sha256=digest(p));write_once(p.with_suffix('.json'),record);records.append(record)
  print(i+1,label,row['damage'],round(metrics['iou'],4),metrics['targets']['contract_pass'],flush=True)
summary={};gates={}
for label in models:
 summary[label]={}
 for group in ['all','damaged','intact']:
  ms=[r['metrics'] for r in records if r['model']==label and (group=='all' or (r['damage']=='intact')==(group=='intact'))]
  summary[label][group]=dict(n=len(ms),median_iou=float(np.median([x['iou'] for x in ms])),all_nine_pass=sum(x['targets']['contract_pass'] for x in ms),median_absolute_request_error_cells=float(np.median([abs(x['request_error_cells']) for x in ms])),**{k:sum(x[k] for x in ms) for k in ['recovered_cells','false_positive_cells','surviving_cells_removed']})
 d=summary[label]['damaged'];intact=[r['metrics'] for r in records if r['model']==label and r['damage']=='intact']
 gates[label]=dict(intact_overlap=all(x['iou']>=.99 for x in intact),intact_validity=all(x['targets']['contract_pass'] for x in intact),damaged_validity=d['all_nine_pass']>=17,damaged_overlap=d['median_iou']>=.9705768039313023,damaged_excess=d['false_positive_cells']<=325,damaged_recovery=d['recovered_cells']>=1945,preservation=summary[label]['all']['surviving_cells_removed']==0,volume_error=d['median_absolute_request_error_cells']<=19)
write_once(out/'result.json',dict(summary=summary,gates=gates,all_conditions_met={k:all(v.values()) for k,v in gates.items()},scope='One paired seed; reused development examples; no TEST,quality promotion or new training.'))
print(json.dumps(dict(summary=summary,gates=gates),indent=2))
