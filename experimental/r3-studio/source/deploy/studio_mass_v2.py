"""MS2: explicit generator versions, bounded scale presets and queued replay."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import base64
import json
import time
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictInt
from typing import Literal
from deploy import studio_mass as legacy
from nca.contract import scene_hash
from nca.incremental_mass_generator import generate_incremental_mass, CoverageGeneratorSpec, VERSION as GENERATOR
from nca.massing_targets import evaluate_targets, MassingTargetSpec, VERSION as EVALUATOR

ROOT=Path(__file__).resolve().parents[1]
VERSION='studio_mass_v2'
FORMAT='studio_mass_portable_v2'
MAX_IMPORT_BYTES=20_000_000
router=APIRouter(prefix='/api/mass-v2')
encode=legacy.encode


def contexts():
    return json.loads((ROOT/'deploy/mass_v2_contexts.json').read_bytes())['contexts']


def context_for(name):
    for item in contexts():
        if item['case']==name:return item
    raise ValueError('Unknown evaluated site preset')


class MassRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scene_case: str
    request_fraction: Literal[.16,.24,.32]=.24
    seed: StrictInt=0


def checked_request(value):
    settings=MassRequest.model_validate(value).model_dump()
    context=context_for(settings['scene_case'])
    if settings['seed'] not in context['seeds'] or settings['request_fraction'] not in context['requests']:
        raise ValueError('Choose a tested seed and volume for this preset')
    return settings


def grid(coords,size):
    if type(size) is not int or size not in (32,48,64):raise ValueError('Unsupported grid')
    if not isinstance(coords,list) or len(coords)>size**3:raise ValueError('Too many coordinates')
    seen=set()
    for cell in coords:
        if not isinstance(cell,list) or len(cell)!=3 or any(type(v) is not int or not 0<=v<size for v in cell):
            raise ValueError('Coordinates must be bounded integer ZYX triples')
        key=tuple(cell)
        if key in seen:raise ValueError('Duplicate coordinate')
        seen.add(key)
    result=np.zeros((size,)*3,bool)
    if coords:result[tuple(np.asarray(coords).T)]=True
    return result


def arrays(context):
    if context['size']==32:
        old=legacy.context_for(context['legacy_context'])
        return legacy.grid(old['domain_zyx']),{k:legacy.grid(v) for k,v in old['masks'].items()}
    raw=base64.b64decode(context['arrays_npz_b64'],validate=True)
    if sha256(raw).hexdigest()!=context['arrays_sha256']:raise ValueError('Context checksum mismatch')
    with np.load(BytesIO(raw),allow_pickle=False) as packed:
        values={k:packed[k] for k in packed.files}
    if any(a.dtype!=np.bool_ or a.shape!=(context['size'],)*3 for a in values.values()):
        raise ValueError('Context shape/type mismatch')
    return values.pop('domain'),values


def generate(settings,progress=None):
    progress=progress or (lambda message:None);started=time.perf_counter()
    settings=checked_request(settings);context=context_for(settings['scene_case'])
    progress('Loading evaluated site and physical scale')
    scene=context['scene'];domain,fields=arrays(context)
    spec=CoverageGeneratorSpec(target_fraction=settings['request_fraction'],max_seconds={32:15,48:45,64:120}[context['size']])
    progress('Growing building volume with incremental accounting')
    field,route,generation=generate_incremental_mass(scene,fields,domain,settings['seed'],spec)
    progress('Checking all nine massing families')
    target,_=evaluate_targets(field,scene,fields,domain,MassingTargetSpec())
    return {'scene':scene,'scene_hash':scene_hash(scene),'mass_request':settings,'method':GENERATOR,
        'learned':False,'interpretation':'Overall building volume; interiors and construction deferred.',
        'occupied_zyx':np.argwhere(field).tolist(),'route_zyx':np.argwhere(route).tolist(),
        'field_sha256':sha256(field.tobytes()).hexdigest(),'targets':target,'generation':generation,
        'input_identity':{'context_sha256':sha256(encode(context)).hexdigest(),'domain_sha256':sha256(domain.tobytes()).hexdigest(),
            'generator':GENERATOR,'evaluator':EVALUATOR,'generator_spec':asdict(spec),'evaluator_spec':asdict(MassingTargetSpec())},
        'elapsed_seconds':time.perf_counter()-started}


def prepare_import(payload):
    if len(encode(payload))>MAX_IMPORT_BYTES:raise ValueError('Export exceeds 20 MB')
    value=payload['record'];version=value.get('version')
    expected={VERSION:FORMAT,legacy.VERSION:legacy.FORMAT}
    if version not in expected or payload.get('format')!=expected[version]:raise ValueError('Unsupported record format/version')
    if value.get('kind')!='mass_result' or value.get('learned') is not False:raise ValueError('Not a procedural mass record')
    if payload['sha256']!=sha256(encode(value)).hexdigest():raise ValueError('Import checksum mismatch')
    legacy.verify_sources(payload['source_files'],payload['source_provenance'])
    claimed=value.get('import_source',{}).get('claimed_provenance',value['provenance'])
    if claimed!=payload['source_provenance']:raise ValueError('Source provenance mismatch')
    checker=legacy.checked_request if version==legacy.VERSION else checked_request
    lookup=legacy.context_for if version==legacy.VERSION else context_for
    settings=checker(value['mass_request']);context=lookup(settings['scene_case'])
    scene=context['scene']
    if value['scene']!=scene or value['scene_hash']!=scene_hash(scene):raise ValueError('Imported site differs from known preset')
    if value['method']!=(legacy.GENERATOR if version==legacy.VERSION else GENERATOR):raise ValueError('Generator version mismatch')
    grid(value['occupied_zyx'],scene['grid_size']);grid(value['route_zyx'],scene['grid_size'])
    return settings,scene,version


def replay_import(payload,progress=None):
    settings,_,version=prepare_import(payload)
    fn=legacy.generate if version==legacy.VERSION else generate
    if progress:progress('Replaying the recognized generator version')
    computed=fn(settings,progress=progress);value=payload['record']
    for key in ('scene','scene_hash','mass_request','method','learned','interpretation','occupied_zyx',
                'route_zyx','field_sha256','targets','input_identity'):
        if computed[key]!=value.get(key):raise ValueError('Imported '+key+' differs from replay')
    actual=deepcopy(computed['generation']);claimed=deepcopy(value['generation'])
    actual.pop('wall_seconds',None);claimed.pop('wall_seconds',None)
    if actual!=claimed:raise ValueError('Imported growth decisions differ from replay')
    computed['import_source']={'id':value['id'],'sha256':payload['sha256'],
        'claimed_provenance':payload['source_provenance'],
        'verification':'Retained source hashes and recognized local version replay verified. Source origin is not authenticated.'}
    return computed


def manager(request):return request.app.state.mass_v2_jobs


def stores(request):return [manager(request),request.app.state.mass_jobs]


def locate(request,record_id):
    from deploy.studio import read_record
    matches=[]
    for m in stores(request):
        if (m.store/record_id/'receipt.json').exists():matches.append((read_record(m.store,record_id),m))
    if len(matches)!=1:raise ValueError('Record missing or ambiguous')
    value,m=matches[0]
    if value['version'] not in (VERSION,legacy.VERSION) or value['kind']!='mass_result':raise ValueError('Unsupported mass record')
    return value,m


def checked(call):
    try:return call()
    except (ValueError,KeyError,TypeError,OSError,OverflowError) as error:raise HTTPException(422,str(error)) from error


@router.get('/presets')
def presets():
    return {'version':VERSION,'generator':GENERATOR,'learned':False,'contexts':[
        {k:c[k] for k in ('case','label','size','seeds','requests','scene','domain_voxels')} for c in contexts()]}


@router.post('/jobs',status_code=202)
def submit(value:MassRequest,request:Request):
    return checked(lambda:manager(request).submit_mass(value.model_dump()))


@router.get('/jobs')
def jobs(request:Request):return manager(request).list()


@router.post('/jobs/{job_id}/cancel')
def cancel(job_id:str,request:Request):return checked(lambda:manager(request).cancel(job_id))


@router.post('/jobs/{job_id}/retry',status_code=202)
def retry(job_id:str,request:Request):return checked(lambda:manager(request).retry(job_id))


@router.post('/import',status_code=202)
def import_record(payload:dict,request:Request):
    return checked(lambda:manager(request).submit_mass(payload=payload))


@router.get('/records')
def records(request:Request):
    from deploy.studio import read_record
    items=[];issues=[]
    for m in stores(request):
        for p in sorted(m.store.glob('*'),reverse=True):
            if not p.is_dir():continue
            try:
                value=read_record(m.store,p.name)
                if value['version'] not in (VERSION,legacy.VERSION):raise ValueError('Unsupported version')
                items.append({k:value[k] for k in ('id','created_at','mass_request','targets','version','method')} |
                    {'status':value['generation']['status']})
            except (OSError,ValueError,TypeError,KeyError):issues.append(p.name)
    return {'records':sorted(items,key=lambda v:v['created_at'],reverse=True),'integrity_issues':issues}


@router.get('/records/{record_id}')
def record(record_id:str,request:Request):return checked(lambda:locate(request,record_id)[0])


@router.get('/records/{record_id}/export')
def export(record_id:str,request:Request):
    def run():
        value,m=locate(request,record_id);sources=legacy.source_files(m,value)
        provenance=value.get('import_source',{}).get('claimed_provenance',value['provenance'])
        legacy.verify_sources(sources,provenance)
        return {'format':FORMAT if value['version']==VERSION else legacy.FORMAT,'record':value,
            'sha256':sha256(encode(value)).hexdigest(),'source_files':sources,'source_provenance':provenance}
    return checked(run)


@router.get('/compare')
def compare(a:str,b:str,request:Request):
    def run():
        left,_=locate(request,a);right,_=locate(request,b)
        if left['targets']['spec']!=right['targets']['spec'] or left['targets']['version']!=right['targets']['version']:
            raise ValueError('Different evaluation definitions')
        same=(left['scene']==right['scene'] and left['input_identity']['domain_sha256']==right['input_identity']['domain_sha256'])
        ca=set(map(tuple,left['occupied_zyx']));cb=set(map(tuple,right['occupied_zyx']))
        return {'a':left,'b':right,'same_context':same,'added_voxels':len(cb-ca) if same else None,
            'removed_voxels':len(ca-cb) if same else None,'shared_voxels':len(ca&cb) if same else None,
            'note':'Same physical context; complete fields compared.' if same else 'Different physical sites; volume and checks are descriptive only.'}
    return checked(run)
