"""Freeze and audit the proposed CGR3 schedule without optimization."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from nca.repair_curriculum import training_start,VERSION
from nca.repair_portable import TrainingOrder
from nca.repair_benchmark import load_example
from nca.experiments import RunStore,write_once,digest,snapshot_source

def main(output):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'starts').mkdir()
 root=ROOT/'.local-artifacts/runs/20260925T094341Z_316cff241020';assert not RunStore(root.parent).verify(root.name)
 study=json.loads((root/'study.json').read_bytes());rows=sorted((r for r in study['examples'] if r['split']=='train'),key=lambda r:(r['case'],r['damage']));assert len(rows)==81
 sampler=TrainingOrder(81,1203);visits=[0]*81;records=[]
 for update in range(256):
  index=sampler.next();row=rows[index];inputs,target=load_example(root,row);o=inputs['occupancy'];allowed=(inputs['context'][0]>0)&(inputs['context'][1]>0)
  kw=dict(split='train',row_hash=row['arrays_sha256'],visit=visits[index]);start,meta=training_start(o,target,allowed,**kw);again,meta2=training_start(o,target,allowed,**kw)
  assert np.array_equal(start,again) and meta==meta2 and (start>=o).all() and (start<=target).all()
  if meta['mode']=='intermediate':assert not np.array_equal(start,target)
  if visits[index]%2==0 or row['damage']=='intact':assert np.array_equal(start,o)
  p=out/'starts'/f'{update+1:04d}.npz'
  with p.open('xb') as f:np.savez_compressed(f,occupancy=start)
  records.append(dict(update=update+1,row_index=index,case=row['case'],damage=row['damage'],source_sha256=row['arrays_sha256'],start_sha256=digest(p),**meta));visits[index]+=1
 # Guard against split leakage; actual model inputs remain occupancy+unchanged context.
 try:training_start(o,target,allowed,split='validation',row_hash=row['arrays_sha256'],visit=1)
 except ValueError:pass
 else:raise AssertionError('Heldout guard failed')
 summary=dict(version=VERSION,updates=256,rows=81,original=sum(x['mode']=='original' for x in records),intermediate=sum(x['mode']=='intermediate' for x in records),fallback=sum(x['mode']=='original_fallback' for x in records),intact_updates=sum(x['damage']=='intact' for x in records),total_added_start_cells=sum(x['added'] for x in records),max_added_start_cells=max(x['added'] for x in records),visits=visits,training_executed=False)
 write_once(out/'schedule.json',records);write_once(out/'result.json',summary);snapshot_source(ROOT,out/'source.zip');print(json.dumps(summary,indent=2))
if __name__=='__main__':main(sys.argv[1])
