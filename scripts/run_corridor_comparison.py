"""C1_v1: separate target corrections and matched frozen-checkpoint diagnostics.

Local only. A retry is a fresh linked run; existing evidence is never replaced.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA, UrbanSceneGenerator, compute_corridor_target_v31
from nca.contract import (load_reference_set, to_generator_params, verify_state_matches_scene,
                          scene_hash, fields_from_state)
from nca.corridor import compute_corridor_target_bounded_v1, BOUNDED_VERSION
from nca.legal_corridor import compute_legal_corridor_v1, LEGAL_VERSION
from nca.evaluation import endpoint_connectivity, material_legality
from nca.e0 import evaluate, aggregate
from nca.experiments import RunStore, provenance, snapshot_source, write_once, digest, read_json
from nca.legacy_scenes import legacy_seed_state
from nca.rollout import run_rollout, historical_training, historical_serving, ROLLOUT_VERSION

VERSIONS = ('legacy_v31', BOUNDED_VERSION, LEGAL_VERSION)
E0_RUN = '20260922T230120Z_76f3b4677e8f'


def connectivity(mask, endpoints):
    try:
        return endpoint_connectivity(mask, endpoints, sorted(endpoints)[0])
    except ValueError as error:
        return {'all_connected': None, 'reason': str(error)}


def audit(target, fields, legacy, bounded):
    mask = target[0].cpu().numpy() > 0.5
    z = np.flatnonzero(mask.any(axis=(1, 2)))
    return {'legality': material_legality(mask, fields['permitted']),
            'z_range_inclusive': [int(z[0]), int(z[-1])] if len(z) else None,
            'entrance_contact_voxels': {name: int((mask & region).sum()) for name, region in fields['endpoints'].items()},
            'legal_connectivity': connectivity(mask & fields['permitted'], fields['endpoints']),
            'permitted_connectivity': connectivity(fields['permitted'], fields['endpoints']),
            'differences': {name: {'added': int((mask & ~other).sum()), 'removed': int((other & ~mask).sum())}
                            for name, other in [('legacy', legacy), ('bounded', bounded)]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config, weights, checkpoint = load_model_c()
    profiles = (historical_training(config), historical_serving(config))
    sets = {name: load_reference_set(REPO / 'experiments/scenes' / name)
            for name in ('reference_v1', 'legacy_easy_v1')}
    protocol = {'protocol': 'C1_v1', 'versions': VERSIONS, 'rollout_version': ROLLOUT_VERSION,
                'effective_model_config': config, 'checkpoint_sha256': digest(checkpoint),
                'scene_manifest_sha256': {name: digest(REPO / 'experiments/scenes' / name / 'manifest.json') for name in sets},
                'profiles': [p.as_dict() for p in profiles], 'seeds': [0], 'steps': 50,
                'schedule_position': 60, 'corridor_width': 1, 'vertical_envelope': 1,
                'device': 'cpu', 'torch_threads': 2, 'deterministic_algorithms': True,
                'training': False, 'optimizer_updates': 0, 'metric_version': 'binary_v1',
                'threshold': 0.5, 'neighborhood': 6, 'expected_targets': 54, 'expected_cases': 108,
                'e0_replay_reference': E0_RUN,
                'limits': 'Single-seed development diagnosis of one historical checkpoint. Target correctness does not establish learned architectural quality. No training or production promotion.'}
    store = RunStore(REPO / '.local-artifacts/runs')
    origin = provenance(REPO)
    run = store.create('C1 bounded expansion and legal routing', 'corridor_diagnostic', protocol,
                       0, origin, parent_run=args.parent_run)
    directory = store.path(run)
    print(f'RUN_ID={run}', flush=True)
    cases, targets, prepared = [], [], []
    started = time.perf_counter()
    status, error = 'completed', None
    def attach_arrays(name, **arrays):
        path = directory / (name + '.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        reference = store.attach(run, path, 'case_fields')
        path.unlink()  # verified registered copy persists
        return reference
    def attach_record(name, record, role):
        path = directory / 'cases' / (name + '.json')
        write_once(path, record)
        store.attach(run, path, role)
    try:
        source = directory / 'source-snapshot.zip'
        snapshot_source(REPO, source)
        store.attach(run, source, 'source_snapshot')
        source.unlink()
        write_once(directory / 'protocol.json', protocol)
        store.attach(run, directory / 'protocol.json', 'protocol')
        # Verify registered historical evidence before using it as a replay oracle.
        verified = store.verify(E0_RUN)
        if verified:
            raise ValueError('E0 artifact verification failed')
        e0_directory = store.path(E0_RUN)
        e0_protocol = read_json(e0_directory / 'run.json')['config']
        if e0_protocol['checkpoint_sha256'] != protocol['checkpoint_sha256'] or e0_protocol['effective_model_config'] != config:
            raise ValueError('Historical checkpoint/config differs from E0')
        e0_cases = {}
        for path in (e0_directory / 'events').glob('*.json'):
            event = read_json(path)
            if event['kind'] == 'artifact' and event['details']['role'] == 'case_record':
                record = read_json(e0_directory / event['details']['path'])
                if record['seed'] == 0:
                    e0_cases[(record['scene_set'], record['scene_id'], record['profile_name'])] = record
        for set_name, scenes in sets.items():
            for sid, scene in sorted(scenes.items()):
                if set_name == 'legacy_easy_v1':
                    seed_state = legacy_seed_state(scene, config)
                    seed_builder = 'nca.legacy_scenes.legacy_seed_state'
                else:
                    seed_state, _ = UrbanSceneGenerator(config).generate(to_generator_params(scene), device='cpu')
                    seed_builder = 'deploy.model_utils.UrbanSceneGenerator'
                problems = verify_state_matches_scene(seed_state, config, scene)
                if problems:
                    raise ValueError('; '.join(problems))
                before = seed_state.clone()
                zero_parity = torch.equal(compute_corridor_target_v31(seed_state, config, 1, 0),
                                          compute_corridor_target_bounded_v1(seed_state, config, 1, 0))
                if not zero_parity:
                    raise ValueError(f'Radius-zero parity failed for {sid}')
                tick = time.perf_counter()
                old = compute_corridor_target_v31(seed_state, config, 1, 1)
                old_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                bounded = compute_corridor_target_bounded_v1(seed_state, config, 1, 1)
                bounded_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                legal = compute_legal_corridor_v1(seed_state, config, [scene], 1, 1)
                legal_seconds = time.perf_counter() - tick
                corridors = dict(zip(VERSIONS, (old, bounded, legal['target'])))
                fields = fields_from_state(seed_state, config, scene)
                ref = attach_arrays(set_name + '__' + sid + '__targets', seed_state=seed_state.numpy(),
                                    permitted=fields['permitted'], legal_centerline=legal['centerline'].numpy(),
                                    **{name: target.numpy() for name, target in corridors.items()})
                if not torch.equal(before, seed_state):
                    raise ValueError('Target generation mutated seed state')
                for version, seconds in zip(VERSIONS, (old_seconds, bounded_seconds, legal_seconds)):
                    row = {'scene_set': set_name, 'scene_id': sid, 'scene_hash': scene_hash(scene),
                           'seed_builder': seed_builder, 'version': version, 'fields': ref,
                           'radius_zero_parity': zero_parity, 'target_seconds': seconds,
                           **audit(corridors[version], fields, old[0].numpy() > 0.5, bounded[0].numpy() > 0.5)}
                    if version == LEGAL_VERSION:
                        row['router'] = legal['reports'][0]
                        if row['legality']['illegal_voxels'] or (row['legal_connectivity']['all_connected'] != row['router']['all_endpoints_connected']):
                            raise ValueError('Legal target failed independent validity check')
                    attach_record(set_name + '__' + sid + '__' + version + '__target', row, 'target_record')
                    targets.append(row)
                prepared.append((set_name, sid, scene, seed_state, corridors, ref))
                print(f'TARGETS {len(targets)}/54 {sid}', flush=True)
        model = UrbanPavilionNCA(dict(config))
        model.load_state_dict(weights, strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for set_name, sid, scene, seed_state, corridors, ref in prepared:
            for profile in profiles:
                for version in VERSIONS:
                    cid = f'{set_name}__{sid}__{profile.name}__{version}__seed-0'
                    row = {'case_id': cid, 'scene_set': set_name, 'scene_id': sid,
                           'scene_hash': scene_hash(scene), 'version': version,
                           'profile_name': profile.name + '__' + version, 'base_profile': profile.name,
                           'seed': 0, 'input_fields': ref, 'status': 'started'}
                    try:
                        random.seed(0)
                        np.random.seed(0)
                        torch.manual_seed(0)
                        tick = time.perf_counter()
                        with torch.inference_mode():
                            result = run_rollout(model, seed_state, profile, corridor_target=corridors[version],
                                                 steps=50, schedule_position=60)
                        row['rollout_seconds'] = time.perf_counter() - tick
                        final = result.pop('state')
                        row['fields'] = attach_arrays(cid, final_state=final.cpu().numpy())
                        row['rollout'] = result
                        if not torch.isfinite(final).all() or not torch.equal(final[:, :config['n_frozen']], seed_state[:, :config['n_frozen']]):
                            raise ValueError('Output finite/frozen-context invariant failed')
                        if version == 'legacy_v31':
                            previous = e0_cases[(set_name, sid, profile.name)]
                            with np.load(e0_directory / previous['fields']['path'], allow_pickle=False) as original:
                                row['e0_replay'] = {'run_id': E0_RUN, 'fields': previous['fields'],
                                                    'bitwise_equal': np.array_equal(final.numpy(), original['final_state'])}
                            if not row['e0_replay']['bitwise_equal']:
                                raise ValueError('Legacy replay differs from E0')
                        row['metrics'] = evaluate(final, config, scene)
                        row['status'] = 'completed'
                    except Exception:
                        row['status'], row['error'] = 'failed', traceback.format_exc()
                        status = 'failed'
                    attach_record(cid, row, 'case_record')
                    cases.append(row)
                    print(f'ROLLOUT {len(cases)}/108 {cid} {row["status"]}', flush=True)
        if len(targets) != 54 or len(cases) != 108:
            raise ValueError('Incomplete protocol')
    except KeyboardInterrupt:
        status, error = 'interrupted', traceback.format_exc()
    except Exception:
        status, error = 'failed', traceback.format_exc()
    summary = {'run_id': run, 'protocol': 'C1_v1', 'status': status,
               'recorded_targets': len(targets), 'recorded_cases': len(cases), 'groups': aggregate(cases),
               'e0_bitwise_equal_cases': sum(c.get('e0_replay', {}).get('bitwise_equal', False) for c in cases),
               'wall_seconds': time.perf_counter() - started, 'error': error, 'provenance': origin,
               'artifact_location': f'.local-artifacts/runs/{run}', 'drive_backup': 'not_uploaded',
               'limits': protocol['limits']}
    write_once(directory / 'summary.json', summary)
    store.attach(run, directory / 'summary.json', 'summary')
    if error:
        store.event(run, 'error', 'Run stopped; prior case evidence retained', traceback=error)
    store.finish(run, status, {'recorded_targets': len(targets), 'recorded_cases': len(cases),
                               'failed_cases': sum(c['status'] != 'completed' for c in cases)}, protocol['limits'])
    write_once(REPO / 'experiments/records' / (run + '.json'), summary)
    print(json.dumps({'run_id': run, 'status': status, 'error': error}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
