"""F2 baseline parity, candidate recovery and access-only training comparison."""
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
from nca.access_training import Session, make_metadata, cost_gate, PROPOSAL, LEGACY, CANDIDATE, EXTRA_EVALUATION_KEYS
from nca.experiments import provenance, snapshot_source, read_json, write_once
from nca.recovery import save_checkpoint, tree_equal, metadata_hash
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


def parity_metadata_equal(current, historical):
    """Only source identity and the explicit version marker may differ."""
    if current.get('protocol') != 'F2_training_v1' or current.get('objective_version') != LEGACY:
        return False
    if historical.get('protocol') != 'F1_training_v1' or 'objective_version' in historical:
        return False
    excluded = {'protocol','proposal_sha256','code_sha256','objective_version'}
    return {k:v for k,v in current.items() if k not in excluded} == {k:v for k,v in historical.items() if k not in excluded}


def parity_current(run, candidate):
    protocol = records(run,'protocol')[0]
    expected = {b:dict(m,objective_version=LEGACY) for b,m in candidate.items()}
    return protocol['mode']=='parity' and protocol['members']==expected


def verify_parity(run, completed=True):
    d=STORE.path(run); assert not STORE.verify(run)
    if completed: assert read_json(d/'result.json')['status']=='completed'
    protocol=records(run,'protocol')[0]; assert protocol['mode']=='parity'
    source=protocol['proposal']['source_fitting_run']; old_dir=STORE.path(source)
    assert not STORE.verify(source) and read_json(old_dir/'result.json')['status']=='completed'
    old_protocol=records(source,'protocol')[0]
    assert old_protocol['protocol']=='F1_v1' and old_protocol['mode']=='study'
    assert protocol['members'].keys()==old_protocol['members'].keys()
    for b,m in protocol['members'].items():
        assert parity_metadata_equal(m,old_protocol['members'][b]), b
    counts={}
    for role in ('training_update','evaluation_record'):
        def key(r):
            return (r['branch'],r['trace']['update'],r['trace']['steps'])
        historical={key(r):r for r in records(source,role)}
        rows=records(run,role)
        boundaries=(1,2,3) if role=='training_update' else (0,1,3)
        horizons=(16,) if role=='training_update' else (16,50)
        expected={(b,u,h) for b in protocol['members'] for u in boundaries for h in horizons}
        assert len(rows)==len(expected) and {key(r) for r in rows}==expected
        for row in rows:
            old=historical[key(row)]
            trace={k:v for k,v in row['trace'].items() if k not in EXTRA_EVALUATION_KEYS}
            assert trace==old['trace'], key(row)
            with np.load(d/row['fields']['path'],allow_pickle=False) as a, np.load(old_dir/old['fields']['path'],allow_pickle=False) as b:
                assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files), key(row)
            a=torch.load(d/row['checkpoint']['path'],map_location='cpu',weights_only=True)
            b=torch.load(old_dir/old['checkpoint']['path'],map_location='cpu',weights_only=True)
            assert a['metadata']==protocol['members'][row['branch']]
            assert b['metadata']==old_protocol['members'][row['branch']]
            assert a['metadata_hash']==metadata_hash(a['metadata']) and b['metadata_hash']==metadata_hash(b['metadata'])
            assert tree_equal({k:v for k,v in a.items() if k not in ('metadata','metadata_hash')},
                {k:v for k,v in b.items() if k not in ('metadata','metadata_hash')}), key(row)
        counts[role+'_exact_F1_matches']=len(rows)
    return dict(baseline_parity=True,**counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['parity','recovery','pilot','study'])
    p.add_argument('--parity-run'); p.add_argument('--recovery-run'); p.add_argument('--pilot-run'); p.add_argument('--parent-run')
    p.add_argument('--worker', action='store_true'); p.add_argument('--run-id'); p.add_argument('--branch')
    p.add_argument('--stop-after', type=int); p.add_argument('--resume')
    args = p.parse_args()
    if args.worker:
        worker(args); return 0
    if not args.mode:
        p.error('--mode required')
    proposal = read_json(REPO / PROPOSAL)
    cfg, _, checkpoint = load_model_c(); _, inputs = load_inputs(REPO)
    full = {f'{recipe}-r{i}': make_metadata(recipe, scene, inputs, cfg, checkpoint, LEGACY if args.mode=='parity' else CANDIDATE)
        for recipe in proposal['recipes'] for i,scene in enumerate(proposal['training_scenes'])}
    gate_meta = full['mass_3-r0']; checks = {}; admission = None
    if args.mode == 'parity':
        members = full
    elif args.mode == 'recovery':
        if not args.parity_run: p.error('--parity-run required')
        checks = verify_parity(args.parity_run)
        assert parity_current(args.parity_run, full), 'Parity source/config differs'
        members = {b: gate_meta for b in ('whole','prefix','resumed','repeat')}
    else:
        if not args.parity_run: p.error('--parity-run required')
        checks = verify_parity(args.parity_run)
        assert parity_current(args.parity_run, full), 'Parity source/config differs'
        if not args.recovery_run:
            p.error('--recovery-run required')
        assert read_json(STORE.path(args.recovery_run)/'result.json')['status'] == 'completed'
        checks.update(verify_recovery(args.recovery_run))
        assert records(args.recovery_run,'protocol')[0]['members']['whole'] == gate_meta, 'Recovery source differs'
        members = full
        if args.mode == 'study':
            if not args.pilot_run:
                p.error('--pilot-run required')
            admission = pilot_gate(args.pilot_run)
            assert records(args.pilot_run,'protocol')[0]['members'] == full, 'Pilot source differs'
            if not admission['admitted']:
                raise ValueError('Timing pilot does not admit full study: ' + str(admission))
    protocol = {'protocol':'F2_v1','mode':args.mode,'proposal':proposal,'members':members,
        'parity_gate':args.parity_run,'recovery_gate':args.recovery_run,'pilot_gate':args.pilot_run,'admission':admission,
        'scope':'Access-only learning comparison against F1; no generalization or architectural certification'}
    origin = provenance(REPO)
    run = STORE.create('F2 '+args.mode, 'access_training_'+args.mode, protocol, 0, origin, parent_run=args.parent_run)
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
            stop={'parity':3,'pilot':2,'study':64}[args.mode]
            for branch in members:
                cap=proposal['parity_worker_cap_seconds'] if args.mode=='parity' else proposal['pilot_worker_cap_seconds'] if args.mode=='pilot' else min(proposal['member_cap_seconds'], proposal['study_cap_seconds']-(time.perf_counter()-started))
                if cap <= 0:
                    raise TimeoutError('Full study cap exhausted')
                launch(run,branch,['--worker','--run-id',run,'--branch',branch,'--stop-after',str(stop)],cap)
            expected_eval = 24 if args.mode in ('pilot','parity') else 56
            assert len(records(run,'training_update')) == 4*stop
            assert len(records(run,'evaluation_record')) == expected_eval
            checks.update(complete_training_matrix=True,complete_evaluation_matrix=True)
            if args.mode=='parity':
                checks.update(verify_parity(run, completed=False))
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
