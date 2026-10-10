"""urban_context_v1: rectangular, multi-site massing inputs, not an evaluator.

Coordinates XYZ, arrays ZYX, half-open integer voxel bounds, Z up.
Existing/context geometry is fixed. Occupied generated cells mean building volume.
No trained-model compatibility, feasibility or structural claim is implied.
"""
import copy
import hashlib
import json
import math
import re
import numpy as np

VERSION = 'urban_context_v1'
AXES = ('x', 'y', 'z')

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _keys(value, expected):
    _require(isinstance(value, dict) and set(value) == set(expected), 'Unexpected or missing fields')

def _id(value):
    _require(isinstance(value,str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}',value), 'Invalid identifier')

def _box(value, shape):
    _keys(value,AXES)
    for i,axis in enumerate(AXES):
        bounds=value[axis]
        _require(isinstance(bounds,list) and len(bounds)==2 and all(type(v) is int for v in bounds)
                 and 0<=bounds[0]<bounds[1]<=shape[i], 'Invalid half-open bounds: '+axis)

def _slice(bounds):
    return tuple(slice(*bounds[a]) for a in reversed(AXES))

def _collections(scene):
    shape=scene['shape_xyz']
    for name in ('buildings','opportunity_regions','protected_regions'):
        entries=scene[name]
        _require(isinstance(entries,list),'Region collection must be a list')
        ids=set()
        for entry in entries:
            _keys(entry,('id','bounds'));_id(entry['id']);_box(entry['bounds'],shape)
            _require(entry['id'] not in ids,'Duplicate '+name+' identifier');ids.add(entry['id'])
    _require(len(scene['buildings'])>=2 and len(scene['opportunity_regions'])>=2,
             'Declare multiple buildings and opportunity regions')

def _interface_box(entry,buildings):
    b=buildings[entry['building']]['bounds'];axis=entry['axis']
    position=b[axis][1] if entry['side']=='max' else b[axis][0]-1
    result={axis:[position,position+1]}
    result.update(entry['spans'])
    return result

def validate(scene):
    """Return an independent validated scene; reject silent extra semantics."""
    _keys(scene,('version','scene_id','shape_xyz','voxel_size_m','origin_xyz_m',
                 'buildings','opportunity_regions','protected_regions','interfaces','connections'))
    s=copy.deepcopy(scene)
    _require(s['version']==VERSION,'Unknown urban scene version');_id(s['scene_id'])
    shape=s['shape_xyz']
    _require(isinstance(shape,list) and len(shape)==3 and all(type(v) is int and 1<=v<=512 for v in shape)
             and math.prod(shape)<=16_777_216,'Invalid or excessive grid dimensions')
    size=s['voxel_size_m']
    _require(type(size) in (int,float) and math.isfinite(size) and size>0,'Invalid voxel size')
    origin=s['origin_xyz_m']
    _require(isinstance(origin,list) and len(origin)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in origin),'Invalid world origin')
    _collections(s)
    buildings={b['id']:b for b in s['buildings']};regions={r['id']:r for r in s['opportunity_regions']}
    existing=np.zeros(tuple(reversed(shape)),dtype=bool)
    for b in s['buildings']:
        sl=_slice(b['bounds']);_require(not existing[sl].any(),'Existing buildings overlap');existing[sl]=True
    opportunity=np.zeros_like(existing)
    for r in s['opportunity_regions']:
        sl=_slice(r['bounds']);_require(not opportunity[sl].any(),'Opportunity regions overlap');opportunity[sl]=True
    protected=np.zeros_like(existing)
    for r in s['protected_regions']:protected[_slice(r['bounds'])]=True
    allowed=opportunity & ~existing & ~protected
    _require(allowed.any(),'Empty growth domain')
    for r in s['opportunity_regions']:_require(allowed[_slice(r['bounds'])].any(),'Empty opportunity region')
    _require(isinstance(s['interfaces'],list) and isinstance(s['connections'],list),'Interfaces/connections must be lists')
    interfaces={}
    for entry in s['interfaces']:
        _keys(entry,('id','building','region','axis','side','spans'));_id(entry['id'])
        _id(entry['building']);_id(entry['region'])
        _require(entry['id'] not in interfaces,'Duplicate interface')
        _require(entry['building'] in buildings and entry['region'] in regions,'Unknown interface reference')
        axis=entry['axis'];_require(isinstance(axis,str) and axis in AXES and entry['side'] in ('min','max'),'Invalid face')
        _keys(entry['spans'],[a for a in AXES if a!=axis])
        bounds=_interface_box(entry,buildings);_box(bounds,shape)
        for a in AXES:
            lo,hi=bounds[a];r=regions[entry['region']]['bounds'][a]
            _require(r[0]<=lo<hi<=r[1],'Interface is outside its opportunity region')
            if a!=axis:
                b=buildings[entry['building']]['bounds'][a]
                _require(b[0]<=lo<hi<=b[1],'Interface exceeds building face')
        _require(allowed[_slice(bounds)].all(),'Interface collides with fixed/protected geometry')
        interfaces[entry['id']]=entry
    pairs=set();link_ids=set()
    for entry in s['connections']:
        _keys(entry,('id','source','target'));_id(entry['id'])
        _require(entry['id'] not in link_ids,'Duplicate connection identifier');link_ids.add(entry['id'])
        a,b=entry['source'],entry['target']
        _id(a);_id(b)
        _require(a in interfaces and b in interfaces and a!=b,'Invalid connection endpoints')
        pair=tuple(sorted((a,b)));_require(pair not in pairs,'Duplicate connection pair');pairs.add(pair)
    _require(bool(pairs),'Declare at least one connection request')
    return s

def fields(scene):
    s=validate(scene);shape=tuple(reversed(s['shape_xyz']))
    result={name:np.zeros(shape,dtype=bool) for name in ('existing','opportunity','protected')}
    for collection,name in [('buildings','existing'),('opportunity_regions','opportunity'),('protected_regions','protected')]:
        for entry in s[collection]:result[name][_slice(entry['bounds'])]=True
    result['allowed']=result['opportunity'] & ~result['existing'] & ~result['protected']
    buildings={b['id']:b for b in s['buildings']}
    result['interfaces']={}
    for entry in s['interfaces']:
        mask=np.zeros(shape,dtype=bool);mask[_slice(_interface_box(entry,buildings))]=True
        result['interfaces'][entry['id']]=mask
    return result

def scene_hash(scene):
    s=validate(scene)
    return hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def summary(scene):
    s=validate(scene);f=fields(s)
    return dict(version=VERSION,scene_id=s['scene_id'],sha256=scene_hash(s),shape_xyz=s['shape_xyz'],
        extent_xyz_m=[v*s['voxel_size_m'] for v in s['shape_xyz']],
        buildings=len(s['buildings']),opportunity_regions=len(s['opportunity_regions']),
        interfaces=len(s['interfaces']),connection_requests=len(s['connections']),
        allowed_cells=int(f['allowed'].sum()),allowed_volume_m3=float(f['allowed'].sum()*s['voxel_size_m']**3),
        representation_valid=True,geometric_feasibility_evaluated=False,trained_model_evaluated=False)
