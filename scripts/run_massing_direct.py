"""MD1 gated four-case comparison. Unique evidence, bounded child processes, no UI publication."""
from pathlib import Path
import argparse
from hashlib import sha256
import json
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from nca.experiments import RunStore, write_once, provenance, snapshot_source, digest
from nca.massing_objective import make_context
from nca.massing_targets import MassingTargetSpec, evaluate_targets
from nca.massing_direct import MassingSession
from nca.contact_mass_generator import ContactGeneratorSpec, generate_contact_mass
from nca.recovery import tree_equal


def grid(coords, size):
    a = np.zeros((size,) * 3, bool)
    if coords:
        a[tuple(np.array(coords).T)] = True
    return a


def code_hashes():
    paths = list((REPO / 'nca').glob('*.py')) + [Path(__file__)]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(paths)}


def inputs(directory, case):
    study = json.loads((directory / 'input-study.json').read_bytes())
    recipe = json.loads((directory / 'recipe.json').read_bytes())
    identity = json.loads((directory / 'identity.json').read_bytes())
    if identity['source_hashes'] != code_hashes() or identity['study_sha256'] != digest(directory / 'input-study.json'):
        raise ValueError('Frozen source/input identity changed')
    if identity['recipe_sha256'] != digest(directory / 'recipe.json'):
        raise ValueError('Frozen recipe changed')
    if case not in recipe['cases']:
        raise ValueError('Case outside frozen pilot')
    original = next(x for x in study['cases'] if x['case'] == case)
    c = next(x for x in study['contexts'] if x['case'] == original['scene_case'])
    n = c['scene']['grid_size']
    domain = grid(c['domain_zyx'], n)
    fields = {k: grid(v, n) for k, v in c['masks'].items()}
    context = make_context(c['scene'], fields, domain, MassingTargetSpec(**study['recipe']['spec']))
    initial = grid(original['occupied_zyx'], n)
    if sha256(initial.tobytes()).hexdigest() != original['field_sha256']:
        raise ValueError('Original MG1 field hash mismatch')
    return recipe, {**identity, 'case': case}, original, context, initial


def worker(args):
    start = time.perf_counter()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    session = None; saved = set(); boundaries = []; updates = []; status = 'completed'; failure = None
    def check():
        if time.perf_counter() - start >= args.seconds:
            raise TimeoutError('Worker wall cap reached inside actual loop')
    def checkpoint(evaluate=True):
        step = session.completed
        if step in saved:
            return
        t = time.perf_counter()
        session.save(out / f'checkpoint-{step:03d}.pt'); saved.add(step)
        if evaluate:
            record, p = session.evaluate()
            write_once(out / f'boundary-{step:03d}.json', record)
            with (out / f'probabilities-{step:03d}.npy').open('xb') as stream:
                np.save(stream, p, allow_pickle=False)
            boundaries.append({'step': step, 'seconds': time.perf_counter() - t})
    try:
        recipe, identity, original, context, initial = inputs(Path(args.input), args.case)
        session = MassingSession(initial, context, recipe, identity)
        if args.resume:
            session.restore(args.resume)
        if session.completed > args.steps:
            raise ValueError('Resume cursor exceeds requested end')
        if not np.array_equal((session.model().detach().numpy() > .5), initial) and not args.resume:
            raise AssertionError('Iteration zero differs from MG1')
        checkpoint(); check()
        while session.completed < args.steps:
            check(); t = time.perf_counter(); record = session.step()
            record['seconds'] = time.perf_counter() - t
            updates.append(record); write_once(out / f'update-{session.completed:03d}.json', record)
            # Save completed state before checking elapsed time after an update.
            if session.completed in recipe['boundaries'] or session.completed == args.steps:
                checkpoint()
            check()
    except (Exception, KeyboardInterrupt) as exc:
        status = 'interrupted' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed'
        failure = traceback.format_exc()
        if session is not None:
            checkpoint(evaluate=False)
    result = {'case': args.case, 'status': status, 'failure': failure,
              'completed_updates': session.completed if session else 0,
              'wall_seconds': time.perf_counter() - start,
              'update_seconds': sum(x['seconds'] for x in updates),
              'boundary_seconds': sum(x['seconds'] for x in boundaries),
              'boundaries': boundaries, 'updates': len(updates), 'resume': args.resume}
    write_once(out / 'worker-result.json', result)
    print(json.dumps(result), flush=True)
    return 0 if status == 'completed' else 1


def main(args):
    recipe = json.loads((REPO / 'experiments/configs/MD1-direct.json').read_bytes())
    store = RunStore(REPO / '.local-artifacts/runs'); meta = provenance(REPO)
    run = store.create('MD1 gated direct mass comparison', 'direct_massing_pilot', recipe,
                       recipe['seed'], meta, args.parent_run or recipe['parent_run'])
    d = store.path(run); print('RUN_ID=' + run, flush=True)
    start = time.perf_counter(); status = 'completed'; failure = None; results = []; gates = {}; contexts = []
    def save(name, value):
        write_once(d / name, value)
    def launch(label, case, steps, seconds, resume=None):
        if seconds <= 0:
            raise TimeoutError('No remaining allowance for ' + label)
        command = [sys.executable, str(Path(__file__)), '--worker', '--input', str(d),
                   '--output', str(d / label), '--case', case, '--steps', str(steps), '--seconds', str(seconds)]
        if resume:
            command += ['--resume', str(resume)]
        t = time.perf_counter()
        with (d / (label + '.log')).open('xb') as log:
            completed = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=seconds)
        elapsed = time.perf_counter() - t
        result = json.loads((d / label / 'worker-result.json').read_bytes())
        result['external_wall_seconds'] = elapsed
        save(label + '-timing.json', result)
        print(label, result['status'], round(elapsed, 3), 's', flush=True)
        if completed.returncode or result['status'] != 'completed':
            raise RuntimeError(label + ' failed: ' + str(result['failure']))
        return result
    try:
        snapshot_source(REPO, d / 'source.zip')
        if store.verify(recipe['source_run']):
            raise ValueError('MG1 input archive failed verification')
        source = store.path(recipe['source_run']) / 'study.json'
        study = json.loads(source.read_bytes()); save('input-study.json', study); save('recipe.json', recipe)
        save('identity.json', {'source_hashes': code_hashes(), 'source_run': recipe['source_run'],
                              'original_study_sha256': digest(source), 'study_sha256': digest(d / 'input-study.json'),
                              'recipe_sha256': digest(d / 'recipe.json'), 'snapshot_sha256': digest(d / 'source.zip')})
        contexts = [c for c in study['contexts'] if c['case'] in ('aligned', 'partial_obstruction')]
        # Recovery uses the failing context; profiling uses four real updates per context.
        recovery_start = time.perf_counter()
        partial = launch('profile-partial', recipe['cases'][2], 4, recipe['profile_seconds_cap'])
        remaining = recipe['recovery_seconds_cap'] - (time.perf_counter() - recovery_start)
        launch('recovery-first', recipe['cases'][2], 2, min(remaining, recipe['profile_seconds_cap']))
        remaining = recipe['recovery_seconds_cap'] - (time.perf_counter() - recovery_start)
        launch('recovery-resumed', recipe['cases'][2], 4, min(remaining, recipe['profile_seconds_cap']),
               d / 'recovery-first/checkpoint-002.pt')
        a = torch.load(d / 'profile-partial/checkpoint-004.pt', weights_only=True)
        b = torch.load(d / 'recovery-resumed/checkpoint-004.pt', weights_only=True)
        recovery = {'whole_checkpoint_exact': tree_equal(a, b),
                    'evaluation_exact': json.loads((d / 'profile-partial/boundary-004.json').read_bytes()) == json.loads((d / 'recovery-resumed/boundary-004.json').read_bytes()),
                    'probabilities_exact': digest(d / 'profile-partial/probabilities-004.npy') == digest(d / 'recovery-resumed/probabilities-004.npy'),
                    'seconds': time.perf_counter() - recovery_start}
        save('recovery.json', recovery)
        if not all(recovery[k] for k in ('whole_checkpoint_exact', 'evaluation_exact', 'probabilities_exact')) or recovery['seconds'] > recipe['recovery_seconds_cap']:
            raise AssertionError('Fresh-process recovery gate failed')
        aligned = launch('profile-aligned', recipe['cases'][0], 4, recipe['profile_seconds_cap'])
        # External time includes process startup/context, two evaluations and four steps.
        # Add seven more four-step blocks and three more mean boundary costs, then double.
        estimates = {k: recipe['timing_safety_factor'] * (v['external_wall_seconds'] + 7 * v['update_seconds'] + 1.5 * v['boundary_seconds'])
                     for k, v in [('aligned', aligned), ('partial_obstruction', partial)]}
        total_estimate = 2 * sum(estimates.values()) + 4 * recipe['procedural']['max_seconds']
        gates = {'recovery': recovery, 'estimated_member_seconds': estimates, 'estimated_pilot_seconds': total_estimate,
                 'admitted': max(estimates.values()) < recipe['member_seconds_cap'] and total_estimate < recipe['pilot_seconds_cap']}
        save('admission.json', gates)
        if not gates['admitted']:
            raise TimeoutError('Timing estimate exceeds declared pilot allowance; pilot not started')
        pilot_start = time.perf_counter()
        for case in recipe['cases']:
            remaining = recipe['pilot_seconds_cap'] - (time.perf_counter() - pilot_start)
            if remaining <= 0:
                raise TimeoutError('Four-case pilot wall cap')
            _, _, original, context, initial = inputs(d, case)
            no_update, _ = evaluate_targets(initial, context.scene, context.masks, context.domain, context.spec)
            if no_update != original['targets']:
                raise AssertionError('No-update binary evaluation changed')
            spec = dict(recipe['procedural']); spec.pop('version')
            t = time.perf_counter()
            field, route, generation = generate_contact_mass(context.scene, context.masks, context.domain, original['seed'], ContactGeneratorSpec(**spec))
            proc_report, proc_masks = evaluate_targets(field, context.scene, context.masks, context.domain, context.spec)
            procedural = {'generation': generation, 'targets': proc_report, 'occupied_zyx': np.argwhere(field).tolist(),
                          'route_zyx': np.argwhere(route).tolist(), 'bulk_zyx': np.argwhere(proc_masks['bulk']).tolist(),
                          'field_sha256': sha256(field.tobytes()).hexdigest(),
                          'binary_request_error_fraction': float(field.sum() / context.domain.sum() - recipe['request_fraction']),
                          'generation_and_evaluation_seconds': time.perf_counter() - t, 'learned': False}
            # Preserve both controls even if the direct worker later fails.
            save(case + '-controls.json', {'case': case, 'original': original, 'contact_aware': procedural})
            remaining = recipe['pilot_seconds_cap'] - (time.perf_counter() - pilot_start)
            if remaining <= 0:
                raise TimeoutError('Pilot wall cap before direct worker')
            timing = launch(case, case, recipe['steps'], min(recipe['member_seconds_cap'], remaining))
            boundaries = [json.loads((d / case / f'boundary-{n:03d}.json').read_bytes()) for n in recipe['boundaries']]
            if boundaries[0]['field_sha256'] != original['field_sha256']:
                raise AssertionError('Iteration-zero identity gate')
            result = {'case': case, 'scene_case': original['scene_case'], 'original': original,
                      'contact_aware': procedural, 'direct': boundaries, 'timing': timing}
            results.append(result); save(case + '-comparison.json', result)
            print(case, 'original', no_update['contract_pass'], 'contact', proc_report['contract_pass'],
                  'direct-final', boundaries[-1]['targets']['contract_pass'], flush=True)
        gates['pilot_actual_seconds'] = time.perf_counter() - pilot_start
        if gates['pilot_actual_seconds'] > recipe['pilot_seconds_cap']:
            raise TimeoutError('Pilot total wall cap')
    except (Exception, KeyboardInterrupt) as exc:
        status = 'interrupted' if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt)) else 'failed'
        failure = traceback.format_exc(); store.event(run, 'failure', failure)
    primary = len(results) == 4 and all(r['direct'][-1]['targets']['contract_pass'] for r in results if r['scene_case'] == 'aligned') and any(r['direct'][-1]['targets']['contract_pass'] for r in results if r['scene_case'] == 'partial_obstruction')
    summary = {'version': recipe['version'], 'run_id': run, 'recipe': recipe, 'status': status, 'failure': failure,
               'gates': gates, 'contexts': contexts, 'cases': results, 'primary_success': primary,
               'unexecuted': [c for c in recipe['cases'] if c not in {r['case'] for r in results}],
               'wall_seconds': time.perf_counter() - start, 'training': False}
    save('study.json', summary)
    # Register every finished or partial artifact, including failed child logs/checkpoints.
    for path in sorted(d.rglob('*')):
        if path.is_file() and path.parts[len(d.parts)] not in ('events', 'artifacts') and path.name != 'run.json':
            store.attach(run, path, 'md1_evidence')
    metrics = {k: summary[k] for k in ('primary_success', 'unexecuted', 'wall_seconds', 'gates')}
    metrics['completed_cases'] = len(results)
    interpretation = 'Four development cases; direct per-instance optimization, not a trained generator. Final32 determines success, all boundaries retained.'
    store.finish(run, status, metrics, interpretation)
    write_once(REPO / 'experiments/records' / (run + '.json'), {'run_id': run, 'status': status, 'config': recipe,
               'metrics': metrics, 'provenance': meta, 'artifact_location': d.relative_to(REPO).as_posix(),
               'interpretation': interpretation, 'drive_backup': 'pending'})
    print(json.dumps({'run_id': run, 'status': status, **metrics}, indent=2), flush=True)
    if failure:
        print(failure)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run'); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--input'); parser.add_argument('--output'); parser.add_argument('--case')
    parser.add_argument('--steps', type=int); parser.add_argument('--seconds', type=float); parser.add_argument('--resume')
    args = parser.parse_args()
    raise SystemExit(worker(args) if args.worker else main(args))
