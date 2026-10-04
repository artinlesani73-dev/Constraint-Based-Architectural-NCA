"""G11-R1: explicit global witness reservation with procedural priority admission."""
import numpy as np
import torch,math
from torch.nn import functional as F
from nca.paced_generation import eligibility,full_origins
from nca.coverage_mass_generator import box_counts,coverage_parts
from nca.massing_targets import neighbors
from nca.contract import entrance_masks
from nca.repair_training import perceive
from nca.budget_reference import budget

def witness(route,allowed,scene,fields,domain,C):
    w=route.copy();valid=full_origins(allowed)
    parts,minima=coverage_parts(domain,.08)
    contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool)
    face=neighbors(fields['existing'].astype(bool));exempt=np.zeros_like(w)
    endpoints=entrance_masks(scene)
    for e in scene['entrances']:
        if e['kind']=='facade':exempt |= endpoints[e['id']]&face&allowed
    contact &= ~exempt
    trace=[]
    for _ in range(C):
        counts=np.array([(w&p).sum() for p in parts]);deficit=np.maximum(minima-counts,0)
        if not deficit.any() and (w&contact).sum()/w.sum()<=.15+1e-10:return w,trace
        eligible,_=eligibility(w,valid);ids=np.flatnonzero(eligible)
        added=box_counts(~w,3).ravel()
        by=np.stack([box_counts((~w)&p,3).ravel() for p in parts])
        extra=box_counts((~w)&contact,3).ravel()
        ids=ids[(added[ids]>0)&(int(w.sum())+added[ids]<=C)]
        if not len(ids):return w,trace
        gain=np.minimum(by[:,ids],deficit[:,None]).sum(0)/added[ids]
        ratio=((w&contact).sum()+extra[ids])/(w.sum()+added[ids])
        order=np.lexsort((ids,ratio,-gain))
        idx=ids[order[0]];p=tuple(int(v) for v in np.unravel_index(idx,valid.shape))
        w[tuple(slice(v,v+3) for v in p)]=True
        trace.append(p)
    return w,trace

def rollout(model,occ,static,allowed,witness,contact,steps=128):
    field=occ[0,0].numpy().astype(bool).copy();legal=allowed[0,0].numpy()
    W=witness;valid=full_origins(legal);wv=full_origins(W)
    D=int(legal.sum());B,C=budget(D,float(static[0,6,0,0,0]),3);K=max(9,math.ceil((C-27)/63))
    hidden=torch.zeros((1,7,*field.shape));rng=torch.Generator().manual_seed(2101)
    born=[];labels=[];trace=[];states={}
    tensor=lambda a:torch.from_numpy(a.copy())[None,None]
    with torch.no_grad():
      for step in range(1,steps+1):
        eligible,seed=eligibility(field,valid);m=tensor(field)
        remaining=occ.new_tensor((B-int(field.sum()))/D).expand_as(occ)
        output=model.last(F.relu(model.first(torch.cat((perceive(torch.cat((m.float(),hidden),1)),static,remaining),1))))
        q=torch.sigmoid(output[:,:1,1:-1,1:-1,1:-1]);fire=torch.rand(q.shape,generator=rng)<.5
        qn=q[0,0].numpy();fn=fire[0,0].numpy()
        old=field.copy();cap=C if seed else min(C,int(field.sum())+K)
        provenance=np.zeros_like(field,dtype=np.uint8);blocked=0
        # Procedural witness cubes ignore score/firing. Eligibility frozen at step start.
        forced=np.flatnonzero(eligible&wv)
        learned=np.flatnonzero(eligible&fn&(qn>.5)&~wv)
        learned=learned[np.lexsort((learned,-qn.ravel()[learned]))]
        for kind,ids in [(2,forced),(1,learned)]:
          for idx in ids:
            p=np.unravel_index(idx,valid.shape);region=tuple(slice(int(v),int(v)+3) for v in p)
            delta=~field[region]
            if not delta.any():continue
            candidate=field.copy();candidate[region]=True
            union=candidate|W
            if candidate.sum()>cap or union.sum()>C or (union&contact).sum()/union.sum()>.15+1e-10:
                blocked+=1;continue
            field=candidate;provenance[region][delta]=kind
            if seed:break
          if seed and field.sum()>1:break
        hidden=(hidden+output[:,1:]*F.pad(fire.float(),(1,1,1,1,1,1)))*allowed
        assert not (old&~field).any() and field.sum()<=cap and (field|W).sum()<=C
        born.append(field&~old);labels.append(provenance)
        trace.append(dict(step=step,mass=int(field.sum()),cap=cap,learned_voxels=int((provenance==1).sum()),procedural_voxels=int((provenance==2).sum()),blocked=blocked,reserved_missing=int((W&~field).sum())))
        if step in [64,128]:states[step]=torch.cat((tensor(field).float(),hidden),1).numpy()
    return dict(states=states,births=np.stack(born),provenance=np.stack(labels),trace=trace)

