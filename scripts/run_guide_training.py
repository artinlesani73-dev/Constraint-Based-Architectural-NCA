"""F5 original-architecture parity, conditioned recovery and constant16 training."""
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
from nca.guide_training import Session, make_metadata, next_horizon, cost_gate, PROPOSAL, CONSTANT, BASELINE, GUIDED
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
        reference=STORE.attach(args.run_id, path, 'training_checkpoint')
        record(args.run_id,f'cursor-{args.branch}-u{session.completed:02d}',
            {'branch':args.branch,'update':session.completed,'next_horizon':next_horizon(meta,session.completed),
             'schedule_version':meta['schedule_version'],'checkpoint':reference},'schedule_cursor')
        return reference

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


    if protocol['mode'] in ('pilot','study'):
        tick=time.perf_counter()
        boundary={(r['trace']['steps'],r['trace']['firing_seed']):r for r in records(args.run_id,'evaluation_record')
            if r['branch']==args.branch and r['trace']['update']==session.completed}
        for seed in proposal['final_evaluation']['firing_seeds']:
            for steps in proposal['final_evaluation']['horizons']:
                name=f'g-{args.branch}-u{session.completed:02d}-s{seed}-h{steps}'
                old=boundary.get((steps,seed))
                if old is not None:
                    row=dict(old,reused_boundary=True)
                else:
                    trace,fields=session.score(steps,seed)
                    field=arrays(args.run_id,name,fields)
                    row={'branch':args.branch,'trace':trace,'checkpoint':ckpt,'fields':field,'reused_boundary':False}
                record(args.run_id,name,row,'horizon_evaluation')
        record(args.run_id,'grid-timing-'+args.branch,{'branch':args.branch,'seconds':time.perf_counter()-tick},'grid_timing')


def launch(run, label, command, cap):
    if cap<=0:raise TimeoutError('Phase cap exhausted before worker')
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
    elapsed = time.perf_counter() - tick
    elapsed_cap_exceeded = elapsed > cap
    record(run, 'process-' + label, {'label': label, 'seconds': elapsed,
        'returncode': code, 'timed_out': timed_out, 'cap_seconds': cap,
        'elapsed_cap_exceeded': elapsed_cap_exceeded}, 'process_record')
    print(f'{label}: exit={code}, seconds={elapsed:.2f}', flush=True)
    # OS waits may exclude suspended time. Do not advance after an elapsed overrun,
    # even when wait() reports a successful process rather than a timeout.
    if timed_out or elapsed_cap_exceeded:
        raise TimeoutError(label + ' exceeded elapsed cap; completed records retained')
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
    if STORE.verify(run) or read_json(STORE.path(run)/'result.json')['status']!='completed':
        raise ValueError('Pilot is incomplete or corrupt')
    p=records(run,'protocol')[0];assert p['mode']=='pilot'
    rows=records(run,'training_update')
    assert len(rows)==8 and {(r['branch'],r['trace']['update']) for r in rows}=={(b,u) for b in p['members'] for u in (1,2)}
    assert len(records(run,'horizon_evaluation'))==72 and len(records(run,'grid_timing'))==4
    assert all(r['trace']['guide_gradient_norm_before_clip']>0 for r in rows), 'Guide branch received no gradient in pilot'
    return timing_admission(run)


def timing_admission(run):
    rows=records(run,'training_update')
    return cost_gate([r['seconds'] for r in rows if r['trace']['steps']==16],
        [r['seconds'] for r in records(run,'evaluation_timing')],
        [r['seconds'] for r in records(run,'grid_timing')],
        [r['seconds'] for r in records(run,'setup_record')])


def parity_metadata_equal(current,historical):
    if current.get('protocol')!='F5_training_v1' or current.get('schedule_version')!=CONSTANT or current.get('architecture_version')!=BASELINE:
        return False
    if current.get('horizon_schedule')!=[16]*64 or historical.get('protocol')!='F4_training_v1':
        return False
    excluded={'protocol','proposal_sha256','code_sha256','schedule_version','horizon_schedule','architecture_version'}
    return {k:v for k,v in current.items() if k not in excluded}=={k:v for k,v in historical.items() if k not in excluded}


def parity_current(run,candidate):
    p=records(run,'protocol')[0]
    expected={b:dict(m,architecture_version=BASELINE) for b,m in candidate.items()}
    return p['mode']=='parity' and p['members']==expected


def verify_parity(run,completed=True):
    d=STORE.path(run);assert not STORE.verify(run)
    if completed:assert read_json(d/'result.json')['status']=='completed'
    p=records(run,'protocol')[0];assert p['mode']=='parity'
    source=p['proposal']['source_raw_training_run'];old_d=STORE.path(source)
    assert not STORE.verify(source) and read_json(old_d/'result.json')['status']=='completed'
    old_p=records(source,'protocol')[0];assert p['members'].keys()==old_p['members'].keys()
    for b,m in p['members'].items():assert parity_metadata_equal(m,old_p['members'][b]),b
    counts={}
    for role in ('training_update','evaluation_record'):
        key=lambda r:(r['branch'],r['trace']['update'],r['trace']['steps'])
        previous={key(r):r for r in records(source,role)};rows=records(run,role)
        expected={(b,u,h) for b in p['members'] for u in ((1,2,3) if role=='training_update' else (0,1,3)) for h in ((16,) if role=='training_update' else (16,50))}
        assert len(rows)==len(expected) and {key(r) for r in rows}==expected
        for r in rows:
            old=previous[key(r)];assert r['trace']==old['trace'],key(r)
            with np.load(d/r['fields']['path']) as a,np.load(old_d/old['fields']['path']) as b:
                assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files),key(r)
            a=torch.load(d/r['checkpoint']['path'],map_location='cpu',weights_only=True)
            b=torch.load(old_d/old['checkpoint']['path'],map_location='cpu',weights_only=True)
            assert a['metadata']==p['members'][r['branch']] and b['metadata']==old_p['members'][r['branch']]
            assert a['metadata_hash']==metadata_hash(a['metadata']) and b['metadata_hash']==metadata_hash(b['metadata'])
            assert tree_equal({k:v for k,v in a.items() if k not in ('metadata','metadata_hash')},
                {k:v for k,v in b.items() if k not in ('metadata','metadata_hash')}),key(r)
        counts[role+'_exact_F4_matches']=len(rows)
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
    full = {f'{recipe}-r{i}': make_metadata(recipe, scene, inputs, cfg, checkpoint, BASELINE if args.mode=='parity' else GUIDED)
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
    protocol = {'protocol':'F5_v1','mode':args.mode,'proposal':proposal,'members':members,
        'parity_gate':args.parity_run,'recovery_gate':args.recovery_run,'pilot_gate':args.pilot_run,'admission':admission,
        'scope':'Persistent scaffold versus F4 raw access; constant16, equal updates/recurrent steps; no generalization claim'}
    origin = provenance(REPO)
    run = STORE.create('F5 '+args.mode, 'guide_training_'+args.mode, protocol, 0, origin, parent_run=args.parent_run)
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
                launch(run,branch,command,min(proposal['recovery_worker_cap_seconds'],proposal['phase_caps_seconds'][args.mode]-(time.perf_counter()-started)))
            checks=verify_recovery(run)
            assert [r['trace']['steps'] for r in sorted(records(run,'training_update'),key=lambda r:r['trace']['update']) if r['branch']=='whole']==[16,16,16]
        else:
            stop={'parity':3,'pilot':2,'study':64}[args.mode]
            for branch in members:
                cap=proposal['parity_worker_cap_seconds'] if args.mode=='parity' else proposal['pilot_worker_cap_seconds'] if args.mode=='pilot' else min(proposal['member_cap_seconds'], proposal['study_cap_seconds']-(time.perf_counter()-started))
                cap=min(cap,proposal['phase_caps_seconds'][args.mode]-(time.perf_counter()-started))
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
                admission=timing_admission(run)
                record(run,'admission',admission,'cost_admission')
        if time.perf_counter()-started>proposal['phase_caps_seconds'][args.mode]:
            raise TimeoutError('Phase elapsed cap exceeded')
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
