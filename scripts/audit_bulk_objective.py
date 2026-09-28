from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from nca.bulk_package import extract
from nca.bulk_repair import bulk_completion,neighbors6,step_loss
from nca.repair_benchmark import load_example
from nca.experiments import write_once
out=Path(sys.argv[1]);receipt=json.loads((out/'package-receipt.json').read_bytes())
_,data=extract(out/receipt['archive'],out/'rehearsal',receipt['archive_sha256'])
torch.set_num_threads(2);records=[]
for row in data['rows']:
 inputs,target=load_example(out/'rehearsal',row,split='train')
 m=torch.from_numpy(inputs['occupancy'])[None,None].bool();t=torch.from_numpy(target)[None,None].float();c=torch.from_numpy(inputs['context'])[None]
 eligible=~m & neighbors6(m) & (c[:,:1]>0)&(c[:,1:2]>0)
 x=torch.zeros_like(t,requires_grad=True);soft=m.float()+eligible*x.sigmoid()
 b=bulk_completion(soft,eligible,t);g=torch.autograd.grad(b,x)[0]
 assert torch.isfinite(g).all() and (g<=0).all() and (g[~t.bool()]==0).all()
 loss,front,volume=step_loss(x,m,eligible,t,row['damage']=='intact')
 records.append(dict(case=row['case'],damage=row['damage'],bulk=float(b.detach()),bulk_gradient_l1=float(g.abs().sum()),frontier=float(front.detach()),volume=float(volume.detach()),total=float(loss.detach())))
write_once(out/'train-gradient-audit.json',dict(rows=records,count=len(records),scope='TRAIN only, initial masks, logits zero, all frontier eligible; not a learned performance result.'))
print('Audited',len(records),'TRAIN rows; nonzero bulk gradients:',sum(x['bulk_gradient_l1']>0 for x in records))
