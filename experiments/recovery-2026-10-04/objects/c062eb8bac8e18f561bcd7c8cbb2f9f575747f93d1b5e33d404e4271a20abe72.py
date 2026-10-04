"""Frozen MG1 development contexts and separate evaluator challenge fields."""
from copy import deepcopy
import numpy as np
from nca.contract import validate_scene
from nca.massing_cases import target_scenes, target_controls


def generation_scenes():
    scenes = target_scenes()
    scene = deepcopy(dict(scenes)['aligned'])
    scene['scene_id'] = 'mg1-partial-obstruction'
    scene['description'] = 'MG1 development context with a partial central obstacle.'
    scene['buildings'].append({'id': 'B_partial', 'x': [15, 17], 'y': [13, 18],
                              'z': [6, 14], 'side': None, 'gap_facing_x': None})
    return scenes + [('partial_obstruction', validate_scene(scene))]


def challenge_fields(scene, domain):
    base = target_controls(scene, domain)['compact_mass']
    appendage = base.copy(); appendage[9, 19:24, 16] = True
    lattice = np.zeros_like(domain)
    # Complete 3-cell bars deliberately satisfy the local cube definition.
    for z in (7, 13):
        for y in (12, 18):
            lattice[z:z+3, y:y+3, 8:24] = True
    for x in (8, 14, 21):
        lattice[7:16, 12:15, x:x+3] = True
        lattice[7:16, 18:21, x:x+3] = True
        lattice[7:10, 12:21, x:x+3] = True
        lattice[13:16, 12:21, x:x+3] = True
    z, y, x = np.indices(domain.shape)
    diagonal = (z >= 7) & (z < 13) & (x >= 8) & (x < 24) & (abs(y-x) <= 3)
    return {'thin_appendage': appendage, 'bulky_lattice': lattice,
            'diagonal_band': diagonal, 'quarter_turn_field': np.rot90(base, axes=(1, 2)).copy()}
