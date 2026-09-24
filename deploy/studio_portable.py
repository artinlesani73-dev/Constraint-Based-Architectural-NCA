"""Checked portable records and descriptive comparisons; no model inference."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from deploy.studio import (checked_scene, scene_hash, encode, identifier,
                           plan_scene, read_record, save_record)


def coordinates(value):
    if not isinstance(value, list) or len(value) > 32**3:
        raise ValueError('Expected bounded voxel coordinates')
    for cell in value:
        if not isinstance(cell, list) or len(cell) != 3 or any(type(n) is not int or not 0 <= n < 32 for n in cell):
            raise ValueError('Coordinates must be integer (z,y,x) cells inside the grid')
    cells = set(map(tuple, value))
    if len(cells) != len(value):
        raise ValueError('Duplicate voxel coordinates')
    return cells


def import_verified(payload, store, config, provenance):
    legacy = payload.get('format') is None
    if legacy:
        value = payload
    else:
        if payload.get('format') != 'studio_portable_v1':
            raise ValueError('Unsupported portable format')
        value = payload['record']
        if payload.get('sha256') != sha256(encode(value)).hexdigest():
            raise ValueError('Import checksum mismatch; no study was accepted')
    if not isinstance(value, dict) or value.get('version') not in ('studio_s1', 'studio_s2'):
        raise ValueError('Unsupported Studio record version')
    if value.get('kind') not in ('scene', 'result'):
        raise ValueError('Only scene and completed result records can be imported')
    scene = checked_scene(value['scene'])
    if scene_hash(scene) != value.get('scene_hash'):
        raise ValueError('Scene hash does not match the submitted geometry')
    digest = sha256(encode(value)).hexdigest()
    # Old raw S1 exports have no transport checksum. Accept only an exact known
    # local record; portable cross-workspace transfers require the new envelope.
    if legacy:
        try:
            local = read_record(store, value['id'])
        except (OSError, ValueError, KeyError) as error:
            raise ValueError('Legacy export lacks a checksum. Re-export it from its original Studio first.') from error
        if sha256(encode(local)).hexdigest() != digest:
            raise ValueError('Legacy export differs from the trusted local record')
        return {'record': local, 'duplicate': True, 'verification': 'exact local legacy match'}

    computed = None
    if value['kind'] == 'result':
        if value.get('method') != 'budgeted_witness_v1' or value.get('learned') is not False:
            raise ValueError('Only the supported procedural method can be verified here')
        coordinates(value['material_zyx']); coordinates(value['guide_zyx'])
        computed = plan_scene(scene, config)
        for key in ('material_zyx', 'guide_zyx', 'diagnostics', 'settings', 'routing', 'construction_status'):
            if value.get(key) != computed[key]:
                raise ValueError(f'Imported {key} differs from the recomputed procedural result')

    for receipt in Path(store).glob('*/receipt.json'):
        try:
            local = read_record(store, receipt.parent.name)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if sha256(encode(local)).hexdigest() == digest or local.get('import_source', {}).get('sha256') == digest:
            return {'record': local, 'duplicate': True, 'verification': 'checksum and supported semantics verified'}
    record = {'id': identifier(), 'created_at': datetime.now(timezone.utc).isoformat(),
              'version': 'studio_s2', 'kind': value['kind'], 'scene': scene,
              'scene_hash': scene_hash(scene), 'provenance': provenance,
              'import_source': {'id': value.get('id'), 'sha256': digest,
                                'claimed_provenance': value.get('provenance'),
                                'verification': 'checksum plus scene validation and procedural replay when applicable'}}
    if computed:
        record.update(computed)
    directory = Path(store) / record['id']
    directory.mkdir(parents=True, exist_ok=False)
    from deploy.studio_jobs import write_once
    write_once(directory / 'import.json', payload)
    save_record(store, record)
    return {'record': record, 'duplicate': False, 'verification': record['import_source']['verification']}


def compare(a, b):
    if a.get('kind') != 'result' or b.get('kind') != 'result':
        raise ValueError('Choose two completed result records')
    if any(a['scene'][key] != b['scene'][key] for key in ('contract_version', 'grid_size', 'voxel_size_m')):
        raise ValueError('Geometry uses incompatible grids or units')
    if any(a['diagnostics'][key] != b['diagnostics'][key] for key in ('metric_version', 'objective_version')):
        raise ValueError('Metric definitions differ; numeric comparison is not supported')
    ca, cb = coordinates(a['material_zyx']), coordinates(b['material_zyx'])
    changes = []
    for group in ('buildings', 'entrances'):
        left = {v['id']: v for v in a['scene'][group]}
        right = {v['id']: v for v in b['scene'][group]}
        for item in sorted(left.keys() | right.keys()):
            if left.get(item) != right.get(item):
                changes.append({'group': group, 'id': item, 'before': left.get(item), 'after': right.get(item)})
    return {'a': a, 'b': b, 'same_scene': a['scene_hash'] == b['scene_hash'],
            'scene_changes': changes, 'added_voxels': len(cb-ca), 'removed_voxels': len(ca-cb),
            'shared_voxels': len(ca & cb),
            'note': 'Descriptive geometry comparison; different scene revisions are not a controlled model comparison.'}
