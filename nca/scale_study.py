"""MG6 physical-scale study helpers; no generator or evaluator changes."""
from copy import deepcopy
import math
import numpy as np
from nca.contract import validate_scene, to_generator_params, fields_from_state, verify_state_matches_scene, declared_existing, entrance_masks
from nca.massing import opportunity_region
from nca.massing_cases import target_scenes

VERSION = 'physical_scale_study_v1'


def decode_grid(coords, size):
    if type(size) is not int or size < 4:
        raise ValueError('Plain integer scene size required')
    a = np.zeros((size,) * 3, bool)
    if not coords:
        return a
    indices = np.asarray(coords)
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in 'iu':
        raise ValueError('Integer ZYX triples required')
    if (indices < 0).any() or (indices >= size).any():
        raise ValueError('Coordinates outside scene')
    a[tuple(indices.T)] = True
    return a


def embed_field(field, size):
    if field.shape != (32, 32, 32) or size not in (48, 64):
        raise ValueError('MG6 embeds 32 into 48 or 64 only')
    shift = (size - 32) // 2
    result = np.zeros((size,) * 3, bool)
    result[:32, shift:shift+32, shift:shift+32] = field
    return result


def embedded_scene(scene, size):
    scene = validate_scene(scene)
    if scene['grid_size'] != 32 or size not in (48, 64):
        raise ValueError('MG6 embeds 32 into 48 or 64 only')
    result = deepcopy(scene);shift = (size - 32) // 2
    result['grid_size'] = size;result['scene_id'] = f'mg6-embedded-{size}'
    result['description'] = 'Same site translated in XY into a larger grid; not a larger opportunity region.'
    result['notes'] = ['Z and physical dimensions unchanged. No generation equivariance claim.']
    for b in result['buildings']:
        for axis in ('x', 'y'):b[axis] = [v+shift for v in b[axis]]
        if b['gap_facing_x'] is not None:b['gap_facing_x'] += shift
    for e in result['entrances']:
        for axis in ('x', 'y'):e[axis] += shift
    return validate_scene(result)


def scale_sites(size):
    if size not in (48, 64):raise ValueError('Frozen sizes: 48, 64')
    base = deepcopy(dict(target_scenes())['aligned'])
    base['grid_size'] = size
    base['buildings'][0].update(x=[0, 8], y=[6, size-6], z=[0, size-7])
    base['buildings'][1].update(x=[size-8, size], y=[6, size-6], z=[0, size-7], gap_facing_x=size-8)
    base['entrances'][0].update(x=8, y=size//2-1, z=12)
    base['entrances'][1].update(x=size-10, y=size//2-1, z=12)
    sites = []
    for kind in ('compact', 'offset_obstacle', 'blocked'):
        scene = deepcopy(base);scene['scene_id'] = f'mg6-{size}-{kind}'
        scene['description'] = f'MG6 {size} grid at 0.8m/cell: {kind}'
        scene['notes'] = ['Designed scale development case; no held-out or architectural-quality claim.']
        if kind == 'offset_obstacle':
            scene['entrances'][0].update(y=size//2-7)
            scene['entrances'][1].update(y=size//2+5, z=20)
            scene['buildings'].append({'id':'B_obstacle','x':[size//2-1,size//2+2],
                'y':[size//2-3,size//2+3],'z':[6,23],'side':None,'gap_facing_x':None})
        if kind == 'blocked':
            scene['buildings'].append({'id':'B_partition','x':[size//2-1,size//2+1],
                'y':[0,size],'z':[0,size],'side':None,'gap_facing_x':None})
        sites.append({'case':f'{size}__{kind}','kind':kind,'partition_control':kind=='blocked','scene':validate_scene(scene)})
    return sites


def scale_context(scene, historical_config, padding=(6.4, 6.4, 0.0)):
    """Explicit scene-sized historical context only; checkpoint weights unused."""
    from deploy.model_utils import UrbanSceneGenerator
    scene = validate_scene(scene);config = dict(historical_config)
    config.update(grid_size=scene['grid_size'], street_levels=scene['street_levels'])
    state, _ = UrbanSceneGenerator(config).generate(to_generator_params(scene))
    problems = verify_state_matches_scene(state, config, scene)
    if problems:raise ValueError(f'Context mismatch: {problems}')
    fields = fields_from_state(state, config, scene)
    domain, region = opportunity_region(scene, fields['permitted'], padding)
    masks = {k:fields[k] for k in ('existing','anchors','permitted','protected','support_boundary')}
    if not np.array_equal(masks['existing'], declared_existing(scene)):raise ValueError('Existing mask mismatch')
    size = scene['voxel_size_m'];width = math.ceil(2.4/size-1e-10)
    report = {'version':VERSION,'effective_context_config':config,'weights_used':False,
        'world_extent_m_zyx':[scene['grid_size']*size]*3,'voxel_volume_m3':size**3,
        'ground_band_m':scene['street_levels']*size,'bulk_width_cells':width,'bulk_width_m':width*size,
        'interfaces':{e['id']:{'extent_m':e['extent']*size,'cells':int(entrance_masks(scene)[e['id']].sum())} for e in scene['entrances']},
        'domain':region,'state_scene_problems':problems}
    return masks, domain, report
