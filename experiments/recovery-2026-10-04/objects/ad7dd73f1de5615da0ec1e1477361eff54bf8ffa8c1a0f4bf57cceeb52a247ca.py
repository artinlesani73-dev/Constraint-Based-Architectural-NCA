from pathlib import Path
import json,hashlib,numpy as np
out=Path('C:/Users/artin/Documents/Codex/outputs/CGR2-Train-Diagnosis-2026-09-29')
assert (out/'result.json').exists()
rows=[];paired={'correct_in_CGR1_only':0,'correct_in_CGR2_only':0,'excess_in_CGR1_only':0,'excess_in_CGR2_only':0};prob={};by_damage={}
for i in range(81):
 fields={}
 for name in ['CGR1','CGR2']:
  p=out/'observations'/f'{i:02d}-{name}.npz';record=json.loads(p.with_suffix('.json').read_bytes());assert hashlib.sha256(p.read_bytes()).hexdigest()==record['arrays_sha256'];rows.append(record)
  with np.load(p,allow_pickle=False) as z:
   field=z['field'];target=z['target'];fields[name]=field;missing=target&~field;q=z['proposals'][-1]
   bins=prob.setdefault(name,dict(missing_q_le_01=0,missing_q_le_025=0,missing_q_le_05=0,missing_q_gt_05=0))
   for key,mask in [('missing_q_le_01',q<=.1),('missing_q_le_025',q<=.25),('missing_q_le_05',q<=.5),('missing_q_gt_05',q>.5)]:bins[key]+=int((missing&mask).sum())
  agg=by_damage.setdefault(name,{}).setdefault(record['damage'],{k:0 for k in record['metrics']})
  for key,val in record['metrics'].items():agg[key]+=val
 paired['correct_in_CGR1_only']+=int((fields['CGR1']&~fields['CGR2']&target).sum());paired['correct_in_CGR2_only']+=int((fields['CGR2']&~fields['CGR1']&target).sum())
 paired['excess_in_CGR1_only']+=int((fields['CGR1']&~fields['CGR2']&~target).sum());paired['excess_in_CGR2_only']+=int((fields['CGR2']&~fields['CGR1']&~target).sum())
result=dict(verified_arrays=len(rows),paired=paired,final_probability_cumulative_bins=prob,by_damage=by_damage)
with (out/'comparison.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
