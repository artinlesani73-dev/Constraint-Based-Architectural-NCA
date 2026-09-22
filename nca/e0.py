"""Frozen E0 diagnostic protocol and aggregation; no fitting or training."""
from collections import defaultdict
import numpy as np
from nca.contract import fields_from_state
from nca.evaluation import (endpoint_connectivity, material_legality, ground_openness,
                            geometric_support, eroded_core)
from nca.rollout import historical_training, historical_evaluation, historical_serving

PROTOCOL_VERSION = 'E0_v1'


def variants(config):
    training = historical_training(config)
    serving = historical_serving(config)
    return [
        (training, (0, 1, 2)),
        (historical_evaluation(config), (0, 1, 2)),
        (serving, (0, 1, 2)),
        (serving.replace(name='serving-seed-015', corridor_seed_scale=0.15), (0,)),
        (serving.replace(name='serving-no-noise', noise_std=0.0), (0,)),
        (serving.replace(name='serving-no-mask', corridor_mask='none'), (0,)),
        (training.replace(name='training-fire-1', fire_rate=1.0), (0,)),
        (serving.replace(name='serving-state-fire-065', firing='state_blend', fire_rate=0.65), (0,)),
        (serving.replace(name='serving-delta-fire-065', module_mode='train',
                         firing='delta_mask', fire_rate=0.65), (0,)),
    ]


def evaluate(state, config, scene):
    fields = fields_from_state(state, config, scene, threshold=0.5)
    material = fields['material']
    source_id = sorted(fields['endpoints'])[0]
    try:
        connectivity = endpoint_connectivity(material, fields['endpoints'], source_id)
        connectivity['status'] = 'scored'
    except ValueError as error:
        # A fragmented source cannot become several origins. Keep and expose the
        # unscorable case instead of silently changing the metric or omitting it.
        connectivity = {'status': 'unscorable', 'reason': str(error),
                        'source_id': source_id, 'all_connected': None}
    raw = state.detach().cpu().numpy()[0, config['ch_structure']]
    return {
        'metric_version': 'binary_v1', 'threshold': 0.5,
        'legality': material_legality(material, fields['permitted']),
        'connectivity': connectivity,
        'ground': ground_openness(material, fields['existing'], fields['protected']),
        'support': geometric_support(material, fields['support_boundary']),
        'thickness_proxy': eroded_core(material),
        'threshold_material_counts': {str(t): int((raw > t).sum()) for t in (0.3, 0.5, 0.7)},
        'interpretation': 'Spatial connectivity and geometric proxies; not walkability or structural safety.',
    }


def aggregate(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[(case['scene_set'], case['profile_name'])].append(case)
    result = []
    for (scene_set, profile), items in sorted(groups.items()):
        successful = [c for c in items if c['status'] == 'completed']
        scored = [c for c in successful if c['metrics']['connectivity']['status'] == 'scored']
        counts = [c['metrics']['legality']['material_voxels'] for c in successful]
        result.append({
            'scene_set': scene_set, 'profile': profile, 'cases': len(items),
            'completed': len(successful), 'failed': len(items) - len(successful),
            'seeds': sorted({c['seed'] for c in items}),
            'mean_material_voxels': float(np.mean(counts)) if counts else None,
            'min_material_voxels': min(counts) if counts else None,
            'max_material_voxels': max(counts) if counts else None,
            'empty_cases': sum(n == 0 for n in counts),
            'connected_cases': sum(c['metrics']['connectivity']['all_connected'] for c in scored),
            'connectivity_scored_cases': len(scored),
            'connectivity_unscorable_cases': len(successful) - len(scored),
            'illegal_voxels_total': sum(c['metrics']['legality']['illegal_voxels'] for c in successful),
            'blocked_protected_voxels_total': sum(c['metrics']['ground']['blocked_voxels'] for c in successful),
            'unsupported_voxels_total': sum(c['metrics']['support']['unsupported_voxels'] for c in successful),
            'mean_rollout_seconds': float(np.mean([c['rollout_seconds'] for c in successful])) if successful else None,
        })
    return result
