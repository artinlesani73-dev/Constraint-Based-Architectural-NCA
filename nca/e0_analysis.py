"""Post-run target diagnostics derived from saved E0 inputs; no new rollouts."""
from pathlib import Path
from hashlib import sha256
import json
from zipfile import ZipFile
import numpy as np
from nca.contract import fields_from_state, scene_hash
from nca.evaluation import endpoint_connectivity, material_legality
from nca.experiments import read_json


def target_audit(repo, directory, protocol, cases):
    rows = []
    config = protocol['effective_model_config']
    # Read the experiment's own source snapshot, not today's checkout. This
    # remains exact even when Git normalizes manifest line endings on restore.
    sources = [event['details']['path'] for path in (directory / 'events').glob('*.json')
               for event in [read_json(path)]
               if event['kind'] == 'artifact' and event['details']['role'] == 'source_snapshot']
    if len(sources) != 1:
        raise ValueError('Expected one recorded source snapshot')
    recorded_sets = {}
    with ZipFile(directory / sources[0]) as snapshot:
        for set_name, expected_hash in protocol['scene_manifest_sha256'].items():
            prefix = 'experiments/scenes/' + set_name + '/'
            raw = snapshot.read(prefix + 'manifest.json')
            if sha256(raw).hexdigest() != expected_hash:
                raise ValueError('Recorded scene manifest hash mismatch')
            scenes = {}
            for item in json.loads(raw)['scenes']:
                payload = snapshot.read(prefix + item['file'])
                scene = json.loads(payload)
                if sha256(payload).hexdigest() != item['sha256'] or scene_hash(scene) != item['scene_hash']:
                    raise ValueError('Recorded scene hash mismatch')
                scenes[scene['scene_id']] = scene
            recorded_sets[set_name] = scenes
    for set_name, scenes in sorted(recorded_sets.items()):
        for sid, scene in sorted(scenes.items()):
            matches = [c for c in cases if (c['scene_set'], c['scene_id'], c['profile_name'], c['seed'])
                       == (set_name, sid, 'historical-training', 0)]
            if len(matches) != 1:
                raise ValueError('Expected exactly one saved training/seed-0 input per scene')
            case = matches[0]
            with np.load(directory / case['fields']['path'], allow_pickle=False) as data:
                fields = fields_from_state(data['seed_state'], config, scene)
                corridor = data['corridor_target'][0] > 0.5
            def connected(material):
                try:
                    return endpoint_connectivity(material, fields['endpoints'], sorted(fields['endpoints'])[0])
                except ValueError as error:
                    return {'all_connected': None, 'reason': str(error)}
            rows.append({'scene_set': set_name, 'scene_id': sid,
                         'input_artifact': case['fields'],
                         'permitted_connectivity': connected(fields['permitted']),
                         'legal_target_connectivity': connected(corridor & fields['permitted']),
                         'raw_target_legality': material_legality(corridor, fields['permitted']),
                         'note': 'Permitted-space connectivity is a spatial upper bound only, not architectural feasibility.'})
    return rows
