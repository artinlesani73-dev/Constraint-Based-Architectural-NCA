"""F1 exact-loop recovery, timing pilot and gated repeated-scene fitting."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.fitting import Session, make_metadata, cost_gate, PROPOSAL
from nca.experiments import provenance, snapshot_source, read_json, write_once
from nca.recovery import save_checkpoint
from scripts.run_sensitivity import STORE, records, record, arrays, verify_recovery as verify_training_recovery
from deploy.checkpoints import load_model_c
from scripts.diagnostic_inputs import load_inputs


def worker(args):
    d = STORE.path(args.run_id); protocol = read_json(d / 'protocol.json')
    meta = protocol['members'][args.branch]; proposal = protocol['proposal']
    tick = time.perf_counter(); session = Session(meta, args.resume)
    record(args.run_id, args.branch + '-setup', {'branch': args.branch,
        'seconds': time.perf_counter() - tick}, 'setup_record')
    if not session.completed <= args.stop_after <= meta['updates']:
        raise ValueError('Invalid stop boundary')

    def checkpoint():
        path = d / f'{args.branch}-u{session.completed:02d}.pt'
        save_checkpoint(path, session.model, session.optimizer, session.scheduler,
            session.generator, meta, session.completed)
        return STORE.attach(args.run_id, path, 'training_checkpoint')

    def score(ckpt):
        tick = time.perf_counter()
        for steps in proposal['evaluation']['horizons']:
            trace, fields = session.score(steps, proposal['evaluation']['firing_seed'])
            name = f'e-{args.branch}-u{session.completed:02d}-h{steps}'
            field = arrays(args.run_id, name, fields)
            record(args.run_id, name, {'branch': args.branch, 'trace': trace,
                'checkpoint': ckpt, 'fields': field}, 'evaluation_record')
        record(args.run_id, f't-{args.branch}-u{session.completed:02d}',
            {'branch': args.branch, 'update': session.completed, 'seconds': time.perf_counter() - tick}, 'evaluation_timing')

    if session.completed == 0:
        score(checkpoint())
    for _ in range(session.completed, args.stop_after):
        tick = time.perf_counter(); trace, fields = session.step(); u = session.completed
        ckpt = checkpoint()
        field = arrays(args.run_id, f'{args.branch}-u{u:02d}', fields)
        row = {'branch': args.branch, 'trace': trace, 'checkpoint': ckpt,
            'fields': field, 'seconds': time.perf_counter() - tick}
        record(args.run_id, f'{args.branch}-u{u:02d}', row, 'training_update')
        if u in proposal['evaluation']['boundaries'] or u == args.stop_after:
            score(ckpt)
        print(f'{args.branch} update={u} loss={trace["total_loss"]:.7g} seconds={row["seconds"]:.3f}', flush=True)


def launch(run, label, command, cap):
    d = STORE.path(run); log = d / (label + '.log'); tick = time.perf_counter(); timed_out = False
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *command],
            cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); code = process.returncode; timed_out = True
        except BaseException:
            process.kill(); process.wait(); raise
    STORE.attach(run, log, 'worker_log')
    record(run, 'process-' + label, {'label': label, 'seconds': time.perf_counter() - tick,
        'returncode': code, 'timed_out': timed_out, 'cap_seconds': cap}, 'process_record')
    print(f'{label}: exit={code}, seconds={time.perf_counter()-tick:.2f}', flush=True)
    if timed_out:
        raise TimeoutError(label + ' exceeded cap; completed records retained')
    if code:
        raise RuntimeError(label + ' failed; see ' + str(log))


def verify_recovery(run):
    checks = verify_training_recovery(run)
    d = STORE.path(run); evaluation = records(run, 'evaluation_record')
    def rows(branches):
        return sorted([r for r in evaluation if r['branch'] in branches],
            key=lambda r: (r['trace']['update'], r['trace']['steps']))
    whole = rows(['whole'])
    assert [(r['trace']['update'], r['trace']['steps']) for r in whole] == [(u,h) for u in (0,1,3) for h in (16,50)]
    for branch in ('resumed', 'repeat'):
        combined = rows(['prefix', branch])
        checks[branch + '_evaluation_traces'] = [r['trace'] for r in whole] == [r['trace'] for r in combined]
        checks[branch + '_evaluation_fields'] = len(whole) == len(combined)
        for a, b in zip(whole, combined):
            with np.load(d/a['fields']['path']) as x, np.load(d/b['fields']['path']) as y:
                checks[branch + '_evaluation_fields'] &= x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
    if not all(checks.values()):
        raise ValueError('Fitting recovery failed: ' + str(checks))
    return checks


def pilot_gate(run):
    if STORE.verify(run) or read_json(STORE.path(run)/'result.json')['status'] != 'completed':
        raise ValueError('Pilot is incomplete or corrupt')
    protocol = records(run, 'protocol')[0]
    if protocol['mode'] != 'pilot':
        raise ValueError('Expected timing pilot')
    rows = records(run, 'training_update')
    assert {(r['branch'],r['trace']['update']) for r in rows} == {(b,u) for b in protocol['members'] for u in (1,2)}
    return cost_gate([r['seconds'] for r in rows],
        [r['seconds'] for r in records(run,'evaluation_timing')],
        [r['seconds'] for r in records(run,'setup_record')])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['recovery','pilot','study'])
    p.add_argument('--recovery-run'); p.add_argument('--pilot-run'); p.add_argument('--parent-run')
    p.add_argument('--worker', action='store_true'); p.add_argument('--run-id'); p.add_argument('--branch')
    p.add_argument('--stop-after', type=int); p.add_argument('--resume')
    args = p.parse_args()
    if args.worker:
        worker(args); return 0
    if not args.mode:
        p.error('--mode required')
    proposal = read_json(REPO / PROPOSAL)
    cfg, _, checkpoint = load_model_c(); _, inputs = load_inputs(REPO)
    full = {f'{recipe}-r{i}': make_metadata(recipe, scene, inputs, cfg, checkpoint)
        for recipe in proposal['recipes'] for i,scene in enumerate(proposal['training_scenes'])}
    gate_meta = full['mass_3-r0']; checks = {}; admission = None
    if args.mode == 'recovery':
        members = {b: gate_meta for b in ('whole','prefix','resumed','repeat')}
    else:
        if not args.recovery_run:
            p.error('--recovery-run required')
        assert read_json(STORE.path(args.recovery_run)/'result.json')['status'] == 'completed'
        checks = verify_recovery(args.recovery_run)
        assert records(args.recovery_run,'protocol')[0]['members']['whole'] == gate_meta, 'Recovery source differs'
        members = full
        if args.mode == 'study':
            if not args.pilot_run:
                p.error('--pilot-run required')
            admission = pilot_gate(args.pilot_run)
            assert records(args.pilot_run,'protocol')[0]['members'] == full, 'Pilot source differs'
            if not admission['admitted']:
                raise ValueError('Timing pilot does not admit full study: ' + str(admission))
    protocol = {'protocol':'F1_v1','mode':args.mode,'proposal':proposal,'members':members,
        'recovery_gate':args.recovery_run,'pilot_gate':args.pilot_run,'admission':admission,
        'scope':'Local single-scene fitting diagnostic; no generalization or architectural certification'}
    origin = provenance(REPO)
    run = STORE.create('F1 '+args.mode, 'fitting_'+args.mode, protocol, 0, origin, parent_run=args.parent_run)
    print('RUN_ID='+run, flush=True); d = STORE.path(run); started = time.perf_counter()
    status, error = 'completed', None
    try:
        record(run,'protocol',protocol,'protocol')
        source=d/'source.zip'; snapshot_source(REPO,source); STORE.attach(run,source,'source_snapshot')
        if args.mode == 'recovery':
            for branch,stop in [('whole',3),('prefix',1),('resumed',3),('repeat',3)]:
                command=['--worker','--run-id',run,'--branch',branch,'--stop-after',str(stop)]
                if branch in ('resumed','repeat'):
                    row=next(r for r in records(run,'training_update') if r['branch']=='prefix')
                    command += ['--resume',str(d/row['checkpoint']['path'])]
                launch(run,branch,command,proposal['recovery_worker_cap_seconds'])
            checks=verify_recovery(run)
        else:
            stop=proposal['pilot_updates'] if args.mode=='pilot' else proposal['updates_per_member']
            for branch in members:
                cap=proposal['pilot_worker_cap_seconds'] if args.mode=='pilot' else min(proposal['member_cap_seconds'], proposal['study_cap_seconds']-(time.perf_counter()-started))
                if cap <= 0:
                    raise TimeoutError('Full study cap exhausted')
                launch(run,branch,['--worker','--run-id',run,'--branch',branch,'--stop-after',str(stop)],cap)
            expected_eval = 24 if args.mode=='pilot' else 56
            assert len(records(run,'training_update')) == 4*stop
            assert len(records(run,'evaluation_record')) == expected_eval
            checks.update(complete_training_matrix=True,complete_evaluation_matrix=True)
            if args.mode=='pilot':
                admission=cost_gate([r['seconds'] for r in records(run,'training_update')],
                    [r['seconds'] for r in records(run,'evaluation_timing')],
                    [r['seconds'] for r in records(run,'setup_record')])
                record(run,'admission',admission,'cost_admission')
    except (KeyboardInterrupt,TimeoutError):
        status,error='interrupted',traceback.format_exc()
    except Exception:
        status,error='failed',traceback.format_exc()
    summary={'run_id':run,'status':status,'error':error,'mode':args.mode,'checks':checks,
        'recorded_updates':len(records(run,'training_update')),
        'evaluation_cases':len(records(run,'evaluation_record')),'admission':admission,
        'seconds':time.perf_counter()-started,'provenance':origin,'artifact_location':f'.local-artifacts/runs/{run}'}
    record(run,'summary',summary,'summary')
    if error:
        STORE.event(run,'error','Attempt stopped; all evidence retained',traceback=error)
    STORE.finish(run,status,checks,protocol['scope'])
    write_once(REPO/'experiments/records'/(run+'.json'),summary)
    print(summary,flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
