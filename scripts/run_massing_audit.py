"""MA1: original/derived completion controls under explicit building-mass semantics."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from nca.experiments import RunStore, provenance, snapshot_source, write_once
from nca.contract import declared_existing, fields_from_state, to_generator_params, scene_hash
from nca.massing import complete_mass, measure_mass, opportunity_region
from nca.spatial import legacy_diagnostics
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanSceneGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', help='Previous MA1 attempt; source VA1 is separately retained')
    args = parser.parse_args()
    recipe = json.loads((REPO / 'experiments/configs/MA1-massing.json').read_text())
    store = RunStore(REPO / '.local-artifacts/runs')
    config, _, checkpoint = load_model_c(device='cpu')
    torch.set_num_threads(2); torch.manual_seed(recipe['random_seed']); np.random.seed(recipe['random_seed'])
    metadata = {**provenance(REPO), 'effective_config': config, 'checkpoint_weights_used': False,
                'checkpoint_config_source_sha256': sha256(checkpoint.read_bytes()).hexdigest()}
    run = store.create('MA1 building mass completion comparison', 'geometry_objective_audit',
                       recipe, recipe['random_seed'], metadata, args.parent_run or recipe['source_run'])
    directory = store.path(run); print('RUN_ID=' + run, flush=True); started = time.perf_counter()
    try:
        snapshot_source(REPO, directory / 'source.zip'); store.attach(run, directory / 'source.zip', 'source_snapshot')
        parent = store.path(recipe['source_run'])
        assert not store.verify(recipe['source_run']), 'Parent archive failed verification'
        parent_bytes = (parent / 'study.json').read_bytes()
        originals = list((parent / 'artifacts').glob('*_study.json'))
        assert len(originals) == 1 and originals[0].read_bytes() == parent_bytes, 'Parent study differs from archived artifact'
        (directory / 'parent-study.json').write_bytes(parent_bytes)
        store.attach(run, directory / 'parent-study.json', 'exact_parent_study')
        previous = json.loads(parent_bytes); scene = previous['scene']; existing = declared_existing(scene)
        state, _ = UrbanSceneGenerator(dict(config)).generate(to_generator_params(scene))
        permitted = fields_from_state(state, config, scene)['permitted']
        domain, domain_report = opportunity_region(scene, permitted, recipe['padding_m_zyx'])
        fields = {}
        parent_by = {c['case']: c for c in previous['cases']}
        for c in previous['cases']:
            field = np.zeros_like(existing)
            for coord in c['material_zyx']: field[tuple(coord)] = True
            fields[c['case']] = field
        plates = np.zeros_like(existing); plates[7, 11:20, 8:24] = True; plates[15, 11:20, 8:24] = True
        fields['detached_plates'] = plates
        blocks = np.zeros_like(existing); blocks[7:10, 11:20, 8:24] = True; blocks[13:16, 11:20, 8:24] = True
        fields['separated_blocks'] = blocks
        domain_record = {**domain_report, 'scene_hash': scene_hash(scene), 'domain_zyx': np.argwhere(domain).tolist()}
        write_once(directory / 'domain.json', domain_record); store.attach(run, directory / 'domain.json', 'fixed_domain')
        records = []; checks = {'fixed_domain_count': int(domain.sum()) == recipe['expected_domain_voxels']}
        for case, field in fields.items():
            for method in recipe['methods']:
                result, operation, masks = complete_mass(field, existing, domain, method,
                    voxel_size_m=scene['voxel_size_m'], axis=recipe['axis_zyx'], max_gap_m=recipe['max_gap_m'])
                mass = measure_mass(result, existing, domain, scene['voxel_size_m'])
                legacy = legacy_diagnostics(scene, result, config)
                record = {'case': case, 'method': method, 'parent_run': recipe['source_run'] if case in parent_by else None,
                    'source_kind': 'saved_VA1_probe' if case in parent_by else 'MA1_gap_counterexample',
                    'scene_hash': scene_hash(scene), 'source_zyx': np.argwhere(field).tolist(),
                    'occupied_zyx': np.argwhere(result).tolist(), 'added_zyx': np.argwhere(masks['added']).tolist(),
                    'requested_zyx': np.argwhere(masks['requested']).tolist(),
                    'rejected_zyx': np.argwhere(masks['rejected']).tolist(),
                    'operation': operation, 'massing': mass, 'legacy': legacy, 'learned': False}
                write_once(directory / f'{case}__{method}.json', record)
                store.attach(run, directory / f'{case}__{method}.json', 'individual_completion_and_diagnostics')
                records.append(record)
                checks[f'{case}/{method}/preserves_source'] = bool(np.all(result[field]))
                checks[f'{case}/{method}/additions_eligible'] = not bool((masks['added'] & (~domain | existing)).any())
                if method == 'identity' and case in parent_by:
                    checks[f'{case}/legacy_parity'] = legacy == parent_by[case]['legacy']
                print(f'{case}/{method}: {operation["source_voxels"]} + {operation["added_voxels"]} = {mass["occupied_voxels"]}; blocked={operation["rejected_additions"]}', flush=True)
        by = {(c['case'], c['method']): c for c in records}
        checks.update({
            'closed_shell_exact_686_added': by['closed_shell_610', 'sealed_cavities']['operation']['added_voxels'] == 686,
            'open_tube_cavity_fill_unchanged': by['open_ends_512', 'sealed_cavities']['operation']['added_voxels'] == 0,
            'open_tube_vertical_span_1296': by['open_ends_512', 'axis_span']['massing']['occupied_voxels'] == 1296,
            'aperture_vertical_span_1296': by['side_aperture_487', 'axis_span']['massing']['occupied_voxels'] == 1296,
            'plates_share_tube_completion': by['detached_plates', 'axis_span']['occupied_zyx'] == by['open_ends_512', 'axis_span']['occupied_zyx'],
            'blocks_share_tube_completion': by['separated_blocks', 'axis_span']['occupied_zyx'] == by['open_ends_512', 'axis_span']['occupied_zyx'],
            'same_legacy_denominator': len({c['legacy']['envelope_voxels'] for c in records}) == 1,
        })
        study = {'version': recipe['version'], 'run_id': run, 'training': False,
                 'parent_study_sha256': sha256(parent_bytes).hexdigest(), 'scene': scene,
                 'recipe': recipe, 'domain': domain_record, 'cases': records, 'checks': checks,
                 'note': 'Analytical building-mass completion controls; interiors deferred. No chosen training objective or acceptance gate.'}
        write_once(directory / 'study.json', study); store.attach(run, directory / 'study.json', 'gallery_data')
        metrics = {'fields': len(fields), 'comparisons': len(records), 'checks': checks,
                   'all_checks_passed': all(checks.values()), 'wall_seconds': time.perf_counter() - started}
        status = 'completed' if all(checks.values()) else 'failed'
        store.finish(run, status, metrics, study['note'])
        write_once(REPO / 'experiments/records' / f'{run}.json', {'run_id': run, 'status': status,
            'config': recipe, 'metrics': metrics, 'provenance': metadata,
            'artifact_location': directory.relative_to(REPO).as_posix(), 'drive_backup': 'pending'})
        print(json.dumps({'run_id': run, 'status': status, 'wall_seconds': metrics['wall_seconds'], 'checks': len(checks)}, indent=2))
        return 0 if status == 'completed' else 1
    except Exception:
        message = traceback.format_exc(); store.event(run, 'error', message)
        if not (directory / 'result.json').exists(): store.finish(run, 'failed', {}, message)
        raise


if __name__ == '__main__': raise SystemExit(main())
