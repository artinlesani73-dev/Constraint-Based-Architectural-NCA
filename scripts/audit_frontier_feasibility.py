"""TRAIN-only oracle reachability audit; no learned rollout or TEST access."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nca.repair_benchmark import load_example
from nca.experiments import write_once
source=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020'
study=json.loads((source/'study.json').read_bytes());rows=[]
for row in study['examples']:
 if row['split']!='train':continue
 inputs,target=load_example(source,row);target=target.astype(bool);occupied=inputs['occupancy'].astype(bool)
 assert not (occupied & ~target).any();reached=occupied.copy();steps=0
 while (target & ~reached).any() and steps<32:
  p=np.pad(reached,1);near=p[:-2,1:-1,1:-1]|p[2:,1:-1,1:-1]|p[1:-1,:-2,1:-1]|p[1:-1,2:,1:-1]|p[1:-1,1:-1,:-2]|p[1:-1,1:-1,2:]
  updated=reached | (near & target);steps+=1
  if np.array_equal(updated,reached):break
  reached=updated
 rows.append(dict(case=row['case'],damage=row['damage'],input_sha256=row['arrays_sha256'],oracle_parallel_steps=steps,unreachable=int((target & ~reached).sum())))
assert len(rows)==81
result=dict(version='frontier_feasibility_v1',rows=rows,max_oracle_steps=max(x['oracle_parallel_steps'] for x in rows),all_reachable=all(x['unreachable']==0 for x in rows),scope='TRAIN-only geometric lower bound with perfect decisions and all cells firing; not learned or stochastic convergence.')
write_once(ROOT/'experiments/reports/frontier-feasibility.json',result)
print({k:v for k,v in result.items() if k!='rows'})
