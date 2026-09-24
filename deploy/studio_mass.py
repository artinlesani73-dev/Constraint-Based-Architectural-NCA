"""MS1: bounded live mass studies with explicit semantics and portable replay."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
import json
import time
import zipfile
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictInt
from typing import Literal
from nca.contract import scene_hash
from nca.budget_mass_generator import generate_budget_mass, BudgetGeneratorSpec, VERSION as GENERATOR
from nca.massing_targets import evaluate_targets, MassingTargetSpec, VERSION as EVALUATOR

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'studio_mass_v1'
FORMAT = 'studio_mass_portable_v1'
IMPORT_GATE = Lock()
router = APIRouter(prefix='/api/mass')


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def contexts():
    return json.loads((ROOT/'deploy/mass_contexts.json').read_bytes())['contexts']


def context_for(name):
    return next(c for c in contexts() if c['case'] == name)


class MassRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scene_case: Literal['aligned','wide_gap','offset_interfaces','blocked_gap','partial_obstruction']
    request_fraction: Literal[0.16,0.24,0.32] = 0.24
    seed: StrictInt = 0


def checked_request(value):
    parsed = MassRequest.model_validate(value).model_dump()
    if not 0 <= parsed['seed'] <= 2147483647:
        raise ValueError('Seed must be an integer between 0 and 2147483647')
    return parsed


def grid(coords):
    from deploy.studio_portable import coordinates
    coordinates(coords)
    result=np.zeros((32,32,32),bool)
    if coords: result[tuple(np.array(coords).T)]=True
    return result


def generate(request, progress=None):
    started=time.perf_counter(); progress=progress or (lambda s:None)
    settings=checked_request(request); progress('Loading fixed site and volume request')
    context=context_for(settings['scene_case']);scene=context['scene']
    domain=grid(context['domain_zyx']);fields={k:grid(v) for k,v in context['masks'].items()}
    spec=BudgetGeneratorSpec(target_fraction=settings['request_fraction'])
    progress('Growing building volume within the contact budget')
    field,route,generation=generate_budget_mass(scene,fields,domain,settings['seed'],spec)
    progress('Checking all nine massing families')
    targets,masks=evaluate_targets(field,scene,fields,domain,MassingTargetSpec())
    return {'scene':scene,'scene_hash':scene_hash(scene),'mass_request':settings,
        'method':GENERATOR,'learned':False,'interpretation':'Overall building volume; interiors and construction deferred.',
        'occupied_zyx':np.argwhere(field).tolist(),'route_zyx':np.argwhere(route).tolist(),
        'field_sha256':sha256(field.tobytes()).hexdigest(),'targets':targets,'generation':generation,
        'input_identity':{'context_sha256':sha256(encode(context)).hexdigest(),
            'domain_sha256':sha256(domain.tobytes()).hexdigest(),'generator':GENERATOR,
            'evaluator':EVALUATOR,'generator_spec':asdict(spec),'evaluator_spec':asdict(MassingTargetSpec())},
        'elapsed_seconds':time.perf_counter()-started}


def manager(request):return request.app.state.mass_jobs


def load(request, record_id):
    from deploy.studio import read_record
    try:
        record=read_record(manager(request).store,record_id)
        if record.get('version')!=VERSION or record.get('kind')!='mass_result':
            raise ValueError('Not a building-mass record')
        return record
    except FileNotFoundError as error:raise HTTPException(404,'Mass record not found') from error
    except (ValueError,KeyError,TypeError) as error:raise HTTPException(409,str(error)) from error


@router.get('/presets')
def presets():
    return {'version':VERSION,'generator':GENERATOR,'learned':False,
        'requests':[.16,.24,.32],'contexts':[{'case':c['case'],'scene':c['scene'],
            'domain_voxels':len(c['domain_zyx'])} for c in contexts()]}


@router.post('/jobs',status_code=202)
def submit(value:MassRequest, request:Request):
    try:
        settings=checked_request(value.model_dump());scene=context_for(settings['scene_case'])['scene']
        return manager(request).submit(scene,scene_hash(scene),mass_request=settings)
    except (ValueError,KeyError,TypeError) as error:raise HTTPException(422,str(error)) from error


@router.get('/jobs')
def jobs(request:Request):return manager(request).list()


@router.post('/jobs/{job_id}/cancel')
def cancel(job_id:str,request:Request):
    try:return manager(request).cancel(job_id)
    except (ValueError,OSError,KeyError) as error:raise HTTPException(409,str(error)) from error


@router.post('/jobs/{job_id}/retry',status_code=202)
def retry(job_id:str,request:Request):
    try:return manager(request).retry(job_id)
    except (ValueError,OSError,KeyError) as error:raise HTTPException(409,str(error)) from error


@router.get('/records')
def records(request:Request):
    items=[];issues=[]
    from deploy.studio import read_record
    for directory in sorted(manager(request).store.glob('*'),reverse=True):
        if not directory.is_dir():continue
        try:
            record=read_record(manager(request).store,directory.name)
            if record['kind']!='mass_result' or record['version']!=VERSION:raise ValueError('Wrong record type')
            items.append({k:record[k] for k in ('id','created_at','mass_request','targets')} |
                         {'status':record['generation']['status']})
        except (ValueError,OSError,KeyError,TypeError):issues.append(directory.name)
    return {'records':items,'integrity_issues':issues}


@router.get('/records/{record_id}')
def record(record_id:str,request:Request):return load(request,record_id)


def source_files(job_manager,value):
    # Export actual retained bytes; never substitute the running server's code.
    if value.get('import_source'):
        payload=json.loads((job_manager.store/value['id']/'import.json').read_bytes())
        return payload['source_files']
    with zipfile.ZipFile(job_manager.directory(value['job_id'])/'source.zip') as archive:
        return {name:archive.read(name).decode('utf-8') for name in archive.namelist()}


def verify_sources(sources, provenance):
    expected=provenance['code_sha256']
    if not isinstance(sources,dict) or sources.keys()!=expected.keys():raise ValueError('Source manifest mismatch')
    for name,text in sources.items():
        if not isinstance(text,str) or sha256(text.encode('utf-8')).hexdigest()!=expected[name]:
            raise ValueError('Source checksum mismatch')
    # Files are retained as data; incoming paths are never extracted or executed.


@router.get('/records/{record_id}/export')
def export(record_id:str,request:Request):
    value=load(request,record_id)
    try:
        sources=source_files(manager(request),value)
        provenance=value.get('import_source',{}).get('claimed_provenance',value['provenance'])
        verify_sources(sources,provenance)
        return {'format':FORMAT,'record':value,'sha256':sha256(encode(value)).hexdigest(),
                'source_files':sources,'source_provenance':provenance}
    except (ValueError,KeyError,OSError) as error:raise HTTPException(409,str(error)) from error


def import_verified(payload,job_manager,provenance):
    from deploy.studio import identifier,read_record,save_record
    from deploy.studio_jobs import write_once
    if payload.get('format')!=FORMAT:raise ValueError('Use a building-mass export, not a material-scaffold record')
    value=payload['record'];digest=sha256(encode(value)).hexdigest()
    if payload.get('sha256')!=digest:raise ValueError('Import checksum mismatch')
    if value.get('kind')!='mass_result' or value.get('version')!=VERSION or value.get('learned') is not False:
        raise ValueError('Unsupported mass record')
    verify_sources(payload['source_files'],payload['source_provenance'])
    claimed=value.get('import_source',{}).get('claimed_provenance',value['provenance'])
    if payload['source_provenance']!=claimed:raise ValueError('Record/source provenance mismatch')
    settings=checked_request(value['mass_request']);grid(value['occupied_zyx']);grid(value['route_zyx'])
    computed=generate(settings)
    for key in ('scene','scene_hash','mass_request','method','learned','interpretation',
                'occupied_zyx','route_zyx','field_sha256','targets','input_identity'):
        if value.get(key)!=computed[key]:raise ValueError('Imported '+key+' differs from procedural replay')
    actual=deepcopy(computed['generation']);submitted=deepcopy(value['generation'])
    actual.pop('wall_seconds',None);submitted.pop('wall_seconds',None)
    if actual!=submitted:raise ValueError('Imported growth decisions differ from procedural replay')
    for receipt in job_manager.store.glob('*/receipt.json'):
        try:local=read_record(job_manager.store,receipt.parent.name)
        except (OSError,ValueError,KeyError,TypeError):continue
        if sha256(encode(local)).hexdigest()==digest or local.get('import_source',{}).get('sha256')==digest:
            return {'record':local,'duplicate':True,'verification':'checksum, source bytes and procedural replay'}
    result={'id':identifier(),'created_at':datetime.now(timezone.utc).isoformat(),
        'version':VERSION,'kind':'mass_result','provenance':provenance,**computed,
        'import_source':{'id':value['id'],'sha256':digest,'claimed_provenance':claimed,
            'verification':'Retained source hashes verified; current procedural replay matches. Source origin is not authenticated.'}}
    directory=job_manager.store/result['id'];directory.mkdir(parents=True)
    write_once(directory/'import.json',payload)
    save_record(job_manager.store,result)
    return {'record':result,'duplicate':False,'verification':result['import_source']['verification']}


@router.post('/import')
def import_record(payload:dict,request:Request):
    if not IMPORT_GATE.acquire(blocking=False):raise HTTPException(429,'An import replay is already running')
    try:return import_verified(payload,manager(request),request.app.state.provenance)
    except (ValueError,KeyError,TypeError,OverflowError) as error:raise HTTPException(422,str(error)) from error
    finally:IMPORT_GATE.release()


@router.get('/compare')
def compare(a:str,b:str,request:Request):
    left=load(request,a);right=load(request,b)
    if left['input_identity']['evaluator']!=right['input_identity']['evaluator'] or left['targets']['spec']!=right['targets']['spec']:
        raise HTTPException(422,'Different mass evaluation definitions')
    ca=set(map(tuple,left['occupied_zyx']));cb=set(map(tuple,right['occupied_zyx']))
    same=left['input_identity']['context_sha256']==right['input_identity']['context_sha256']
    return {'a':left,'b':right,'same_context':same,'added_voxels':len(cb-ca),
        'removed_voxels':len(ca-cb),'shared_voxels':len(ca&cb),
        'note':'Same fixed context.' if same else 'Different contexts: descriptive comparison only.'}
