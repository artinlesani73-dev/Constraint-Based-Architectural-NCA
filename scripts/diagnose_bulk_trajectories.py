"""TRAIN-only paired trajectory diagnosis. No optimization or heldout evaluation."""
from pathlib import Path
import sys,json,hashlib,zipfile,io,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from nca.connected_repair import ConnectedRepair,neighbors6
from nca.bulk_repair import ConnectedRepair as BulkRepair
from nca.repair_training import perceive
from nca.repair_benchmark import load_example
from nca.experiments import digest,write_once,RunStore,snapshot_source

def main(output):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'observations').mkdir()
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 locations={'CGR1':('CGR1-Final-Review-2026-09-28','20260928T153419Z_44ad70fb1464',ConnectedRepair),'CGR2':('CGR2-Final-Review-2026-09-29','20260929T062933Z_465972e5c250',BulkRepair)}
 models={};provenance={}
 for label,(folder,run,cls) in locations.items():
  root=Path('C:/Users/artin/Documents/Codex/outputs')/folder;a=root/(run+'.zip');receipt=json.loads(a.with_suffix('.receipt.json').read_bytes());assert digest(a)==receipt['sha256']
  seal=json.loads((root/'sealed-model.json').read_bytes())
  with zipfile.ZipFile(a) as z:raw=z.read('worker/checkpoint-0256.pt')
  assert hashlib.sha256(raw).hexdigest()==seal['checkpoint_sha256']
  payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);assert payload['completed']==256
  model=cls().float();model.load_state_dict(payload['model']);model.eval();models[label]=model;provenance[label]=dict(archive_sha256=receipt['sha256'],checkpoint=seal)
 source=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020';assert not RunStore(source.parent).verify(source.name)
 study=json.loads((source/'study.json').read_bytes());rows=[x for x in study['examples'] if x['split']=='train'];assert len(rows)==81
 write_once(out/'request.json',dict(split='train',rows=81,steps=32,firing_seed=2101,checkpoint=256,torch=str(torch.__version__),numpy=np.__version__,provenance=provenance,method='Unmodified target-free rollout; independently reconstruct same firing masks, assert every birth and final field match.'))
 snapshot_source(ROOT,out/'source.zip');records=[];start=time.monotonic()
 for i,row in enumerate(rows):
  inputs,target=load_example(source,row);o=torch.from_numpy(inputs['occupancy'])[None,None];c=torch.from_numpy(inputs['context'])[None];t=torch.from_numpy(target)[None,None].bool();allowed=(c[:,:1]>0)&(c[:,1:2]>0)
  ideal=o.bool().clone();depth=0
  while (t&~ideal).any() and depth<32:ideal=ideal|(neighbors6(ideal)&t&allowed);depth+=1
  assert torch.equal(ideal,t)
  for label,model in models.items():
   with torch.no_grad():r=model.rollout(o,perceive(c),allowed,torch.Generator().manual_seed(2101),32,capture=True)
   g=torch.Generator().manual_seed(2101);m=o.bool().clone();ever_front=torch.zeros_like(m);ever_fire=m&False;rejected=torch.zeros_like(o,dtype=torch.int16);front_count=rejected.clone();history=[]
   for step in range(32):
    q=r['proposals'][step];front=allowed&~m&neighbors6(m);fire=torch.rand(m.shape,generator=g)<.5;eligible=front&fire;born=eligible&(q>.5)
    assert torch.equal(born,r['births'][step]);ever_front|=front;ever_fire|=eligible;rejected+=(eligible&(q<=.5)).short();front_count+=eligible.short();m|=born
    history.append(dict(step=step+1,correct_births=int((born&t).sum()),wrong_births=int((born&~t).sum()),missing=int((t&~m).sum()),eligible_missing=int((eligible&t).sum()),rejected_missing=int((eligible&t&(q<=.5)).sum())))
   assert torch.equal(m,r['field']);missing=t&~m
   metrics=dict(missing=int(missing.sum()),recovered=int((m&t&~o.bool()).sum()),excess=int((m&~t).sum()),never_frontier=int((missing&~ever_front).sum()),frontier_never_fired=int((missing&ever_front&~ever_fire).sum()),fired_rejected=int((missing&ever_fire).sum()),rejected_at_least8=int((missing&(rejected>=8)).sum()),last8_correct_births=sum(h['correct_births'] for h in history[-8:]),last8_wrong_births=sum(h['wrong_births'] for h in history[-8:]))
   assert metrics['missing']==sum(metrics[k] for k in ['never_frontier','frontier_never_fired','fired_rejected'])
   p=out/'observations'/f'{i:02d}-{label}.npz'
   with p.open('xb') as f:np.savez_compressed(f,initial=o.numpy(),target=t.numpy(),field=m.numpy(),proposals=r['proposals'].numpy(),births=r['births'].numpy(),state=r['state'].numpy(),eligible_count=front_count.numpy(),rejected_count=rejected.numpy())
   record=dict(case=row['case'],damage=row['damage'],model=label,ideal_target_growth_steps=depth,metrics=metrics,history=history,arrays_sha256=digest(p));write_once(p.with_suffix('.json'),record);records.append(record)
  if (i+1)%9==0:print('Completed paired TRAIN examples',i+1,'/81',flush=True)
 summary={}
 for label in models:
  summary[label]={}
  for group in ['all','damaged','intact']:
   selected=[x for x in records if x['model']==label and (group=='all' or (x['damage']=='intact')==(group=='intact'))]
   summary[label][group]={k:sum(x['metrics'][k] for x in selected) for k in selected[0]['metrics']};summary[label][group]['rows']=len(selected)
 write_once(out/'result.json',dict(summary=summary,max_ideal_target_growth_steps=max(x['ideal_target_growth_steps'] for x in records),seconds=time.monotonic()-start,scope='Training-set mechanism diagnosis, not generalization; no interventions, extra horizon, threshold tuning or training.'))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main(sys.argv[1])
