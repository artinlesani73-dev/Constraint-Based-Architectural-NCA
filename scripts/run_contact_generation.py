"""MG2: frozen contact-aware recipe against all45 original MG1 cases."""
from pathlib import Path
from hashlib import sha256
import argparse, json, sys, time, traceback
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from nca.experiments import RunStore, write_once, provenance, snapshot_source, digest
from nca.contact_mass_generator import generate_contact_mass, ContactGeneratorSpec
from nca.massing_targets import evaluate_targets, MassingTargetSpec
from run_mass_generation import summarize


def grid(coords, size):
    a = np.zeros((size,) * 3, bool)
    if coords: a[tuple(np.array(coords).T)] = True
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run'); args = parser.parse_args()
    recipe = json.loads((REPO / 'experiments/configs/MG2-contact.json').read_bytes())
    torch.set_num_threads(recipe['threads'])
    store = RunStore(REPO / '.local-artifacts/runs'); meta = provenance(REPO)
    run = store.create('MG2 complete contact-aware mass matrix', 'procedural_massing', recipe, 0, meta,
                       args.parent_run or recipe['parent_run'])
    d = store.path(run); print('RUN_ID=' + run, flush=True); start = time.perf_counter()
    records = []; contexts = []; status = 'completed'; failure = None
    tasks = [f'{scene}__v{int(request*100)}__s{seed}' for scene in recipe['contexts']
             for request in recipe['volume_requests'] for seed in recipe['seeds']]
    def save(name, value, role):
        write_once(d / name, value); store.attach(run, d / name, role)
    def check_time():
        if time.perf_counter() - start >= recipe['study_seconds_cap']:
            raise TimeoutError('MG2 study cap at case boundary')
    try:
        snapshot_source(REPO, d / 'source.zip'); store.attach(run, d / 'source.zip', 'exact_source')
        if store.verify(recipe['source_run']) or store.verify(recipe['parent_run']):
            raise ValueError('Input evidence failed verification')
        source = store.path(recipe['source_run']) / 'study.json'
        study = json.loads(source.read_bytes())
        identity = json.loads((store.path(recipe['parent_run']) / 'identity.json').read_bytes())
        for name in ('nca/contact_mass_generator.py', 'nca/massing_objective.py', 'nca/massing_targets.py'):
            if digest(REPO / name) != identity['source_hashes'][name]:
                raise ValueError('Frozen MD1 generator/evaluator changed: ' + name)
        save('input-study.json', study, 'unchanged_MG1_input')
        save('input-identity.json', {'source_run': recipe['source_run'], 'sha256': digest(source),
             'frozen_MD1_identity': identity}, 'source_identity')
        spec = MassingTargetSpec(**study['recipe']['spec'])
        for name in recipe['contexts']:
            context = next(c for c in study['contexts'] if c['case'] == name)
            contexts.append(context); save(name + '-context.json', context, 'fixed_context')
            scene = context['scene']; n = scene['grid_size']; domain = grid(context['domain_zyx'], n)
            fields = {k: grid(v, n) for k, v in context['masks'].items()}
            for request in recipe['volume_requests']:
                for seed in recipe['seeds']:
                    check_time(); case = f'{name}__v{int(request*100)}__s{seed}'
                    original = next(c for c in study['cases'] if c['case'] == case)
                    old = grid(original['occupied_zyx'], n)
                    if sha256(old.tobytes()).hexdigest() != original['field_sha256']:
                        raise ValueError('Original field hash differs')
                    before, _ = evaluate_targets(old, scene, fields, domain, spec)
                    if before != original['targets']:
                        raise AssertionError('Original evaluator changed')
                    field, route, generation = generate_contact_mass(scene, fields, domain, seed,
                        ContactGeneratorSpec(recipe['cube_m'], request, recipe['candidate_seconds_cap'], recipe['contact_weight']))
                    t = time.perf_counter(); targets, masks = evaluate_targets(field, scene, fields, domain, spec)
                    evaluation_seconds = time.perf_counter() - t
                    error = generation['target_error_voxels']
                    record = {'case': case, 'kind': 'generated', 'scene_case': name, 'request_fraction': request, 'seed': seed,
                              'occupied_zyx': np.argwhere(field).tolist(), 'route_zyx': np.argwhere(route).tolist(),
                              'bulk_zyx': np.argwhere(masks['bulk']).tolist(), 'field_sha256': sha256(field.tobytes()).hexdigest(),
                              'generation': generation, 'targets': targets, 'evaluation_seconds': evaluation_seconds,
                              'original': original, 'baseline_regression': before['contract_pass'] and not targets['contract_pass'],
                              'repaired': not before['contract_pass'] and targets['contract_pass'],
                              'request_met_with_cube_overshoot': 0 <= error < generation['cube_width_cells'] ** 3,
                              'added_voxels': int((field & ~old).sum()), 'removed_voxels': int((old & ~field).sum()), 'learned': False}
                    records.append(record); save(case + '.json', record, 'paired_candidate')
                    print(case, generation['status'], targets['contract_pass'], [k for k,v in targets['family_pass'].items() if not v], flush=True)
        check_time()
    except (Exception, KeyboardInterrupt) as exc:
        status = 'interrupted' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed'
        failure = traceback.format_exc(); store.event(run, 'error', failure)
    unexecuted = [case for case in tasks if case not in {c['case'] for c in records}]
    if status == 'completed' and unexecuted: status = 'failed'; failure = 'Incomplete planned matrix'
    open_cases = [c for c in records if c['scene_case'] != 'blocked_gap']
    regressions = [c['case'] for c in records if c['baseline_regression']]
    no_timeout = all(c['generation']['status'] != 'time_limit' for c in records)
    promoted = status == 'completed' and len(open_cases) == 36 and not regressions and no_timeout and all(
        c['targets']['contract_pass'] and c['request_met_with_cube_overshoot'] for c in open_cases)
    groups = summarize(records, contexts)
    metrics = {'generated': len(records), 'valid_generated': sum(c['targets']['contract_pass'] for c in records),
               'open_valid': sum(c['targets']['contract_pass'] for c in open_cases), 'open_total': len(open_cases),
               'baseline_regressions': regressions, 'repairs': [c['case'] for c in records if c['repaired']],
               'studio_promotion_gate': promoted, 'unexecuted': unexecuted,
               'generation_seconds': sum(c['generation']['wall_seconds'] for c in records),
               'evaluation_seconds': sum(c['evaluation_seconds'] for c in records),
               'wall_seconds': time.perf_counter() - start}
    result = {'run_id': run, 'version': recipe['version'], 'recipe': recipe, 'status': status, 'failure': failure,
              'contexts': contexts, 'cases': records, 'groups': groups, 'metrics': metrics, 'training': False}
    save('study.json', result, 'complete_or_partial_study')
    interpretation = 'Fixed procedural development comparison, no held-out generalization. All failures retained; MT1 and cost12 unchanged.'
    store.finish(run, status, metrics, interpretation)
    write_once(REPO / 'experiments/records' / (run + '.json'), {'run_id': run, 'status': status, 'config': recipe,
               'metrics': metrics, 'provenance': meta, 'artifact_location': d.relative_to(REPO).as_posix(),
               'drive_backup': 'pending', 'interpretation': interpretation})
    print(json.dumps({'run_id': run, 'status': status, **metrics}, indent=2), flush=True)
    if failure: print(failure)
    return 0 if status == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
