"""G4 deterministic NumPy block-admission reference; no learned model."""
import numpy as np

VERSION='overlapping_cube_growth_reference_v1'

def full_origins(field,width=3):
    field=np.asarray(field)
    if field.dtype!=bool or field.ndim!=3 or type(width) is not int or width<1 or min(field.shape)<width:
        raise ValueError('Boolean3D field and fitting positive width required')
    return np.lib.stride_tricks.sliding_window_view(field,(width,)*3).all(axis=(-3,-2,-1))

def cube_union(origins,width=3):
    origins=np.asarray(origins,dtype=bool);result=np.zeros(tuple(n+width-1 for n in origins.shape),bool)
    for z in range(width):
        for y in range(width):
            for x in range(width):result[z:z+origins.shape[0],y:y+origins.shape[1],x:x+origins.shape[2]]|=origins
    return result

def adjacent_origins(origins):
    p=np.pad(origins,1);n=origins.shape
    return p[:-2,1:-1,1:-1]|p[2:,1:-1,1:-1]|p[1:-1,:-2,1:-1]|p[1:-1,2:,1:-1]|p[1:-1,1:-1,:-2]|p[1:-1,1:-1,2:]

def eligible_origins(field,allowed,width=3):
    if field.dtype!=bool or allowed.dtype!=bool or field.shape!=allowed.shape or (field&~allowed).any():
        raise ValueError('Legal matching Boolean state required')
    valid=full_origins(allowed,width);full=full_origins(field,width)
    if field.sum()==1:
        seed=np.argwhere(field)[0];coords=np.indices(valid.shape)
        contains=np.ones_like(valid)
        for axis in range(3):contains&=(coords[axis]<=seed[axis])&(seed[axis]<coords[axis]+width)
        return valid&contains,True
    if not full.any() or not np.array_equal(cube_union(full,width),field):
        raise ValueError('After seed, state must be a complete cube union')
    return valid & ~full & adjacent_origins(full),False

def transition(field,allowed,probability,fire,ceiling,width=3):
    eligible,seed_phase=eligible_origins(field,allowed,width)
    q=np.asarray(probability);fire=np.asarray(fire)
    if q.shape!=eligible.shape or fire.shape!=eligible.shape or fire.dtype!=bool or not np.isfinite(q).all() or ((q<0)|(q>1)).any():
        raise ValueError('Matching finite probabilities and Boolean firing required')
    if type(ceiling) is not int or ceiling<int(field.sum()):raise ValueError('Invalid ceiling; no deletion allowed')
    offered=np.flatnonzero(eligible&fire&(q>.5));order=offered[np.lexsort((offered,-q.ravel()[offered]))]
    result=field.copy();mass=int(result.sum());accepted=[];rejected=[];redundant=[];trace=[]
    for index in order:
        origin=tuple(int(v) for v in np.unravel_index(index,q.shape));region=tuple(slice(v,v+width) for v in origin)
        delta=int((~result[region]).sum())
        if delta==0:redundant.append(list(origin));continue
        if mass+delta>ceiling:rejected.append(dict(origin=list(origin),new_cells=delta));continue
        result[region]=True;mass+=delta;accepted.append(list(origin));trace.append(dict(origin=list(origin),new_cells=delta,mass=mass))
        if seed_phase:break # One seed-containing cube only in this step.
    return result,dict(seed_phase=seed_phase,offered_count=len(offered),accepted=accepted,rejected_budget=rejected,redundant=redundant,trace=trace,
        initial_mass=int(field.sum()),final_mass=mass,unused_capacity=ceiling-mass,
        deferred_after_first_cube=len(order)-len(accepted)-len(rejected)-len(redundant) if seed_phase and accepted else 0)
