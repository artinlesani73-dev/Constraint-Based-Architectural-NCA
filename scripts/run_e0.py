"""Run the local CPU E0 diagnostic and retain every case, field and failure.

Run without arguments for the frozen protocol. --parent-run links a complete
fresh retry to prior retained evidence; it never overwrites or silently skips
old cases. Interrupted processes retain completed case files and registered
artifacts even if finalization could not run. No cloud access is performed.
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
from nca.contract import load_reference_set, to_generator_params, verify_state_matches_scene, scene_hash
from nca.e0 import PROTOCOL_VERSION, variants, evaluate, aggregate
from nca.experiments import RunStore, provenance, snapshot_source, write_once, digest
from nca.legacy_scenes import legacy_seed_state
from nca.rollout import run_rollout, ROLLOUT_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', help='Retain and link a prior attempt; rerun the frozen protocol')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config, weights, checkpoint = load_model_c()
    profile_seeds = variants(config)
    sets = {name: load_reference_set(REPO / 'experiments/scenes' / name)
            for name in ('reference_v1', 'legacy_easy_v1')}
    manifests = {name: digest(REPO / 'experiments/scenes' / name / 'manifest.json') for name in sets}
    protocol = {
        'protocol': PROTOCOL_VERSION, 'rollout_version': ROLLOUT_VERSION,
        'device': 'cpu', 'torch_threads': 2, 'deterministic_algorithms': True,
        'training': False, 'optimizer_updates': 0, 'steps': 50, 'schedule_position': 60,
        'threshold': 0.5, 'neighborhood': 6, 'metric_version': 'binary_v1',
        'checkpoint_sha256': digest(checkpoint), 'effective_model_config': config,
        'scene_manifest_sha256': manifests,
        'profiles': [{'profile': p.as_dict(), 'seeds': list(seeds)} for p, seeds in profile_seeds],
        'expected_cases': sum(len(s) for s in sets.values()) * sum(len(seeds) for _, seeds in profile_seeds),
        'corridor_operator': 'unchanged legacy compute_corridor_target_v31 (known vertical-envelope defect)',
        'training_profile_note': 'Fixed 50 steps and epoch 60 (fully annealed mask); forward dynamics only.',
        'seed_policy': 'Three seeds for each historical profile; ablations use predetermined seed 0 only.',
        'limits': 'Diagnostic development scenes, one trained checkpoint; not a held-out generalization study. Single-seed ablations do not establish robust effects.',
        'timing_scope': 'CPU rollout wall time only; no warmup, no GPU/deployment benchmark claim.',
        'backup': 'Local only; user declined Drive upload. No cloud API or sync.',
    }
    store = RunStore(REPO / '.local-artifacts/runs')
    origin = provenance(REPO)
    run_id = store.create('E0 historical rollout diagnosis', 'baseline_diagnostic', protocol,
                          0, origin, parent_run=args.parent_run)
    print(f'RUN_ID={run_id} EXPECTED_CASES={protocol["expected_cases"]}', flush=True)
    directory = store.path(run_id)
    cases = []
    status = 'completed'
    error = None
    started = time.perf_counter()
    try:
        source = directory / 'source-snapshot.zip'
        snapshot_source(REPO, source)
        store.attach(run_id, source, 'source_snapshot')
        source.unlink()  # The verified registered copy is retained.
        write_once(directory / 'protocol.json', protocol)
        store.attach(run_id, directory / 'protocol.json', 'protocol')
        model = UrbanPavilionNCA(dict(config))
        model.load_state_dict(weights, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for set_name, scenes in sets.items():
            for scene_id, scene in sorted(scenes.items()):
                if set_name == 'legacy_easy_v1':
                    seed_state = legacy_seed_state(scene, config)
                    seed_builder = 'nca.legacy_scenes.legacy_seed_state'
                else:
                    seed_state, _ = UrbanSceneGenerator(config).generate(to_generator_params(scene), device='cpu')
                    seed_builder = 'deploy.model_utils.UrbanSceneGenerator'
                problems = verify_state_matches_scene(seed_state, config, scene)
                if problems:
                    raise ValueError(f'{scene_id}: {problems}')
                for profile, seeds in profile_seeds:
                    needs_corridor = profile.corridor_seed_scale > 0 or profile.corridor_mask != 'none'
                    corridor = compute_corridor_target_v31(seed_state, config,
                        corridor_width=profile.corridor_width,
                        vertical_envelope=profile.vertical_envelope) if needs_corridor else None
                    for seed in seeds:
                        case_id = f'{set_name}__{scene_id}__{profile.name}__seed-{seed}'
                        record = {'case_id': case_id, 'scene_set': set_name, 'scene_id': scene_id,
                                  'scene_hash': scene_hash(scene), 'seed_builder': seed_builder,
                                  'profile_name': profile.name, 'seed': seed, 'status': 'started'}
                        try:
                            random.seed(seed)
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            tick = time.perf_counter()
                            with torch.inference_mode():
                                result = run_rollout(model, seed_state, profile, corridor_target=corridor,
                                                     steps=50, schedule_position=60)
                            record['rollout_seconds'] = time.perf_counter() - tick
                            final = result.pop('state')
                            # Preserve the continuous fields before scoring so a
                            # failed invariant/metric still leaves inspectable evidence.
                            field_path = directory / (case_id + '.npz')
                            with field_path.open('xb') as stream:
                                np.savez_compressed(stream, seed_state=seed_state.cpu().numpy(),
                                    final_state=final.cpu().numpy(),
                                    corridor_target=(corridor.cpu().numpy() if corridor is not None else np.empty(0, np.float32)),
                                    material=(final[0, config['ch_structure']].cpu().numpy() > 0.5))
                            record['fields'] = store.attach(run_id, field_path, 'case_fields')
                            field_path.unlink()
                            record['rollout'] = result
                            if not torch.isfinite(final).all():
                                raise ValueError('Nonfinite output')
                            if not torch.equal(final[:, :config['n_frozen']], seed_state[:, :config['n_frozen']]):
                                raise ValueError('Frozen context was modified')
                            record['metrics'] = evaluate(final, config, scene)
                            record['status'] = 'completed'
                        except Exception:
                            record['status'] = 'failed'
                            record['error'] = traceback.format_exc()
                            status = 'failed'
                        path = directory / 'cases' / (case_id + '.json')
                        write_once(path, record)
                        store.attach(run_id, path, 'case_record')
                        cases.append(record)
                        print(f'{len(cases)}/{protocol["expected_cases"]} {case_id} {record["status"]} '
                              f'{record.get("rollout_seconds", 0):.3f}s', flush=True)
        if len(cases) != protocol['expected_cases']:
            raise RuntimeError('Incomplete protocol')
    except KeyboardInterrupt:
        status, error = 'interrupted', traceback.format_exc()
    except Exception:
        status, error = 'failed', traceback.format_exc()
    summary = {'run_id': run_id, 'protocol': PROTOCOL_VERSION, 'status': status,
               'expected_cases': protocol['expected_cases'], 'recorded_cases': len(cases),
               'wall_seconds': time.perf_counter() - started, 'groups': aggregate(cases),
               'error': error, 'provenance': origin,
               'artifact_location': f'.local-artifacts/runs/{run_id}', 'drive_backup': 'not_uploaded',
               'limits': protocol['limits']}
    write_once(directory / 'summary.json', summary)
    store.attach(run_id, directory / 'summary.json', 'summary')
    if error:
        store.event(run_id, 'error', 'Run stopped; all completed case evidence retained', traceback=error)
    store.finish(run_id, status, {'recorded_cases': len(cases), 'expected_cases': protocol['expected_cases'],
                                 'failed_cases': sum(c['status'] != 'completed' for c in cases),
                                 'wall_seconds': summary['wall_seconds']}, protocol['limits'])
    write_once(REPO / 'experiments/records' / (run_id + '.json'), summary)
    print(json.dumps({'run_id': run_id, 'status': status, 'recorded_cases': len(cases), 'error': error}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
