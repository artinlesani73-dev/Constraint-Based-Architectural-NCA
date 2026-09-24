"""Local Studio S1: procedural design studies, separate from historical serving.

Run: python -m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001
No learned model, training, cloud storage, or public hosting in this service.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
import json
import logging
import os
import re
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanSceneGenerator
from nca.contract import (validate_scene, scene_hash, to_generator_params,
                          fields_from_state, load_reference_set)
from nca.legal_corridor import route_legal_corridor
from nca.losses import LossSpec, material_envelope, context_from_scenes
from nca.facade import endpoint_allowance
from nca.constructive import build_witness
from nca.objective import research_terms
from nca.evaluation import endpoint_connectivity, material_legality, geometric_support

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / '.local-artifacts' / 'studio'
VERSION = 'studio_s2'
GATE = Lock()


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def identifier():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:12]


def save_record(root, record):
    """Unique directory; fsync payload before publishing a hashed receipt.

    Incomplete directories remain inspectable and never appear as saved results.
    No overwrite/delete API. Files are the authority, not browser storage.
    """
    directory = Path(root) / record['id']
    directory.mkdir(parents=True, exist_ok=True)
    payload = encode(record)
    with (directory / 'record.json').open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {'id': record['id'], 'sha256': sha256(payload).hexdigest(),
               'kind': record['kind'], 'scene_id': record['scene']['scene_id'],
               'created_at': record['created_at']}
    with (directory / 'receipt.json').open('xb') as stream:
        stream.write(encode(receipt))
        stream.flush()
        os.fsync(stream.fileno())
    return record


def read_record(root, record_id):
    if not re.fullmatch(r'\d{8}T\d{6}Z_[0-9a-f]{12}', record_id):
        raise ValueError('Invalid saved record ID')
    directory = Path(root) / record_id
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    payload = (directory / 'record.json').read_bytes()
    if sha256(payload).hexdigest() != receipt['sha256']:
        raise ValueError('Saved record failed its integrity check')
    record = json.loads(payload)
    if receipt['id'] != record_id or record['id'] != record_id:
        raise ValueError('Saved record identity mismatch')
    return record


def checked_scene(value):
    # Bound allocation before calling the general-purpose contract validator.
    if value.get('grid_size') != 32 or value.get('street_levels') != 6:
        raise ValueError('Studio S1 requires grid_size=32 and street_levels=6')
    if value.get('voxel_size_m') != 0.8 or value.get('ceiling_z') is not None:
        raise ValueError('Studio S1 uses 0.8 m voxels and no height ceiling')
    if value.get('legacy_relaxations'):
        raise ValueError('Studio editing does not enable legacy scene relaxations')
    if not 1 <= len(value.get('buildings', [])) <= 12 or not 2 <= len(value.get('entrances', [])) <= 8:
        raise ValueError('Use 1–12 buildings and 2–8 entrances')
    scene = validate_scene(value)
    for building in scene['buildings']:
        face = building['gap_facing_x']
        if face is not None and face != building['x'][1 if building['side'] == 'left' else 0]:
            raise ValueError('Gap-facing coordinate must match the declared building face')
    to_generator_params(scene)  # enforces supported entrance extent
    return scene


def plan_scene(scene, config, progress=None):
    """Same W1 construction/proxies, recomputed for the submitted scene."""
    started = time.perf_counter()
    progress = progress or (lambda stage: None)
    progress('Validating scene')
    scene = checked_scene(scene)
    with torch.inference_mode():
        state, _ = UrbanSceneGenerator(dict(config)).generate(to_generator_params(scene))
        fields = fields_from_state(state, config, scene)
        progress('Routing connections')
        routed = route_legal_corridor(fields['permitted'], fields['endpoints'])
        guide = torch.from_numpy(routed['centerline'])[None]
        permitted = torch.from_numpy(fields['permitted'])[None]
        envelope = material_envelope(guide, permitted, 6)
        context = context_from_scenes(state, config, [scene], guide, envelope,
                    torch.tensor([routed['report']['all_endpoints_connected']]))
        allowance, _ = endpoint_allowance(scene, permitted)
        progress('Constructing scaffold')
        witness = build_witness(context, allowance)
        material = witness['material'].float()
        state[:, config['ch_structure']] = material
        progress('Evaluating nine families')
        values = research_terms(state, material, context, config, allowance)
        binary = material[0].numpy() > 0.5
        connectivity = endpoint_connectivity(binary & fields['permitted'],
                             fields['endpoints'], sorted(fields['endpoints'])[0])
        ratio = float(values['mass_ratio'][0])
        valid = bool(values['context_valid'][0])
        in_budget = 0.03 - 1e-6 <= ratio <= 0.12 + 1e-6
        diagnostics = {
            'metric_version': 'binary_v1', 'objective_version': 'research_objective_v1',
            'context_valid': valid, 'route_feasible': routed['report']['all_endpoints_connected'],
            'connectivity': connectivity, 'material_ratio': ratio,
            'material_voxels': int(binary.sum()), 'envelope_voxels': int(envelope.sum()),
            'budget_range': [0.03, 0.12], 'in_budget': in_budget,
            'joint_budget_connectivity': valid and connectivity['all_connected'] and in_budget,
            'families': {k: float(v[0]) for k, v in values['terms'].items()},
            'regularizers': {k: float(v[0]) for k, v in values['regularizers'].items()},
            'legality': material_legality(binary, fields['permitted']),
            'support': geometric_support(binary, fields['support_boundary']),
            'interpretation': 'Geometric proxies only; not walkability or mechanical safety.',
        }
    return {'scene': scene, 'scene_hash': scene_hash(scene),
            'method': 'budgeted_witness_v1', 'learned': False,
            'construction_status': witness['status'], 'routing': routed['report'],
            'material_zyx': np.argwhere(binary).tolist(),
            'guide_zyx': np.argwhere(routed['centerline']).tolist(),
            'diagnostics': diagnostics, 'elapsed_seconds': time.perf_counter() - started,
            'settings': {'envelope_radius': 6, 'threshold': 0.5,
                         'loss_spec': vars(LossSpec()), 'random_seed': None}}


class SceneRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scene: dict


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    app.state.config, _, checkpoint = load_model_c(device='cpu')
    sources = list((ROOT / 'nca').glob('*.py')) + list((ROOT / 'deploy').glob('studio*.py')) + [
                ROOT / 'deploy/model_utils.py', ROOT / 'deploy/checkpoints.py', ROOT / 'deploy/mass_contexts.json']
    sources += list((ROOT / 'deploy/static/live').glob('*'))
    app.state.provenance = {
        'checkpoint_config_source_sha256': sha256(checkpoint.read_bytes()).hexdigest(),
        'checkpoint_weights_used': False, 'config': app.state.config,
        'code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sources},
        'torch': str(torch.__version__), 'numpy': str(np.__version__), 'threads': 2}
    app.state.store = STORE
    from deploy.studio_jobs import JobManager
    app.state.jobs = JobManager(STORE, app.state.provenance)
    try:
        app.state.mass_jobs = JobManager(STORE.parent / (STORE.name + '-mass'), app.state.provenance)
        try:
            yield
        finally:
            app.state.mass_jobs.close()
    finally:
        app.state.jobs.close()


app = FastAPI(title='NCA Studio — local design studies', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT / 'deploy/static'), name='static')


@app.middleware('http')
async def bounded_local_requests(request, call_next):
    from fastapi.responses import JSONResponse
    if request.method == 'POST':
        # Same-origin JSON API; reject browser cross-origin writes and huge bodies.
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Cross-origin writes are disabled'}, status_code=403)
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            limit = (20_000_000 if request.url.path == '/api/mass/import' else
                     2_000_000 if request.url.path == '/api/studio/import' else 100_000)
            if size > limit:
                return JSONResponse({'detail': 'Scene request too large'}, status_code=413)
            chunks.append(chunk)
        request._body = b''.join(chunks)
    return await call_next(request)


@app.get('/')
def index():
    return FileResponse(ROOT / 'deploy/studio.html')


@app.get('/api/studio/scenes')
def presets():
    return {'version': VERSION, 'scenes': list(load_reference_set().values())}


@app.post('/api/studio/validate')
def validate(request: SceneRequest):
    try:
        scene = checked_scene(request.scene)
        return {'scene': scene, 'scene_hash': scene_hash(scene)}
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


@app.post('/api/studio/records')
def create(request: SceneRequest, plan: bool = True):
    try:
        scene = checked_scene(request.scene)
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, str(error)) from error
    if not GATE.acquire(blocking=False):
        raise HTTPException(429, 'A local study is running. Please retry after it finishes.')
    record_id = identifier()
    base = {'id': record_id, 'created_at': datetime.now(timezone.utc).isoformat(),
            'version': VERSION, 'kind': 'result' if plan else 'scene', 'scene': scene,
            'scene_hash': scene_hash(scene), 'provenance': app.state.provenance}
    try:
        # A crash must retain the submitted scene even before a result exists.
        directory = app.state.store / record_id
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / 'request.json').open('xb') as stream:
            stream.write(encode(base))
            stream.flush()
            os.fsync(stream.fileno())
        if plan:
            base.update(plan_scene(scene, app.state.config))
        return save_record(app.state.store, base)
    except Exception as error:
        logging.getLogger(__name__).exception('Studio attempt %s failed', record_id)
        # Preserve computation failures; leave any incomplete write untouched.
        failure = {**base, 'id': identifier(), 'kind': 'failure',
                   'failed_attempt_id': record_id, 'error': str(error)}
        try:
            save_record(app.state.store, failure)
        except OSError:
            logging.getLogger(__name__).exception('Could not preserve Studio failure record')
        raise HTTPException(500, 'Study failed; inspect local Studio records and server logs.') from error
    finally:
        GATE.release()


@app.get('/api/studio/records')
def records():
    items, issues = [], []
    if app.state.store.exists():
        for directory in sorted(app.state.store.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            try:
                record = read_record(app.state.store, directory.name)
                items.append({k: record[k] for k in ('id', 'kind', 'scene_hash', 'created_at', 'scene')})
            except (ValueError, OSError, KeyError):
                issues.append(directory.name)
    return {'records': items, 'integrity_issues': issues}


@app.get('/api/studio/records/{record_id}')
def record(record_id: str):
    try:
        return read_record(app.state.store, record_id)
    except FileNotFoundError as error:
        raise HTTPException(404, 'Saved record not found or incomplete') from error
    except (ValueError, KeyError) as error:
        raise HTTPException(409, str(error)) from error


@app.post('/api/studio/jobs', status_code=202)
def submit_job(request: SceneRequest):
    try:
        scene = checked_scene(request.scene)
        return app.state.jobs.submit(scene, scene_hash(scene))
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


@app.get('/api/studio/jobs')
def list_jobs():
    return app.state.jobs.list()


@app.post('/api/studio/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    try:
        return app.state.jobs.cancel(job_id)
    except (ValueError, OSError, KeyError) as error:
        raise HTTPException(409, str(error)) from error


@app.post('/api/studio/jobs/{job_id}/retry', status_code=202)
def retry_job(job_id: str):
    try:
        return app.state.jobs.retry(job_id)
    except (ValueError, OSError, KeyError) as error:
        raise HTTPException(409, str(error)) from error


@app.get('/api/studio/records/{record_id}/export')
def export_record(record_id: str):
    value = record(record_id)
    return {'format': 'studio_portable_v1', 'sha256': sha256(encode(value)).hexdigest(), 'record': value}


@app.post('/api/studio/import')
def import_record(payload: dict):
    from deploy.studio_portable import import_verified
    if not GATE.acquire(blocking=False):
        raise HTTPException(429, 'Another record operation is running')
    try:
        return import_verified(payload, app.state.store, app.state.config, app.state.provenance)
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        GATE.release()


@app.get('/api/studio/compare')
def compare_records(a: str, b: str):
    from deploy.studio_portable import compare
    try:
        return compare(record(a), record(b))
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(422, str(error)) from error


from deploy.studio_mass import router as mass_router
app.include_router(mass_router)
