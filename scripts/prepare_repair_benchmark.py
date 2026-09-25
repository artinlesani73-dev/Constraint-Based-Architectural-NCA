"""NL0: archive frozen teachers, deterministic damage and nonlearned repair controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.experiments import RunStore, digest, provenance, snapshot_source, write_once
from nca.massing_targets import evaluate_targets
from nca.repair_benchmark import (VERSION, CHANNELS, condition, context_hash, field_hash,
    damage, closing_repair, repair_metrics, assert_split_integrity, load_example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run')
    args = parser.parse_args()
    recipe = json.loads((REPO/'experiments/configs/NL0-repair.json').read_bytes())
    store = RunStore(REPO/'.local-artifacts/runs')
    run = store.create('NL0 repair benchmark preparation', 'data_admission', recipe,
                       recipe['damage_seed'], provenance(REPO), args.parent_run or recipe['teacher_run'])
    d = store.path(run)
    print('RUN_ID='+run, flush=True)
    started = time.perf_counter()
    targets, examples, guards = [], [], []
    status, failure = 'completed', None

    def save(name, value, role='benchmark_record'):
        write_once(d/name, value)
        store.attach(run, d/name, role)

    def arrays(name, **values):
        p = d/name; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as stream:
            np.savez_compressed(stream, **values)
        store.attach(run, p, 'lossless_arrays')
        return digest(p)

    try:
        snapshot_source(REPO, d/'source.zip'); store.attach(run, d/'source.zip', 'exact_source')
        for path, expected in recipe['source_sha256'].items():
            if digest(REPO/path) != expected:
                raise ValueError('Frozen source changed: '+path)
        if store.verify(recipe['teacher_run']):
            raise ValueError('Teacher run integrity failed')
        source = store.path(recipe['teacher_run'])
        study = json.loads((source/'study.json').read_bytes())
        if digest(source/'study.json') != recipe['teacher_study_sha256']:
            raise ValueError('Teacher matrix manifest changed')
        indexed = {row['input_key']: row for row in study['cases']}
        for task in recipe['cases']:
            if time.perf_counter()-started > recipe['max_seconds']:
                raise TimeoutError('NL0 study boundary cap exceeded')
            key, split = task['case'], task['split']
            row = indexed[key]; entry = study['contexts'][key]
            context = json.loads((source/entry['json']).read_bytes())
            record = json.loads((source/row['json']).read_bytes())
            scene = context['scene']
            with np.load(source/entry['arrays'], allow_pickle=False) as p:
                domain = p['domain']; fields = {k:p[k] for k in ('permitted','existing','protected','support_boundary')}
            with np.load(source/row['arrays'], allow_pickle=False) as p:
                target = p['field']
            scores, _ = evaluate_targets(target, scene, fields, domain)
            if scores != record['targets']:
                raise ValueError('Archived MT1 score changed: '+key)
            request = record['generation']['spec']['target_fraction']
            path = 'targets/'+key
            save(path+'.json', {'scene':scene, 'source_case':row, 'generation':record['generation'], 'targets':scores})
            arrays(path+'.npz', target=target, domain=domain, **fields)
            if split == 'guard':
                if scores['contract_pass'] or record['generation']['status'] != 'no_cube_route':
                    raise ValueError('Blocked control changed')
                guards.append({'case':key, 'split':split, 'targets':scores, 'json':path+'.json', 'arrays':path+'.npz'})
                continue
            if not scores['contract_pass']:
                raise ValueError('Frozen teacher is not acceptable: '+key)
            target_row = {'case':key, 'split':split, 'site':task['site'],
                          'context_sha256':context_hash(scene,fields,domain), 'target_sha256':field_hash(target),
                          'json':path+'.json', 'arrays':path+'.npz'}
            targets.append(target_row)
            for kind in recipe['damage_kinds']:
                damaged, cut = damage(target, kind, key, recipe['damage_seed'])
                again, cut_again = damage(target, kind, key, recipe['damage_seed'])
                if not np.array_equal(damaged, again) or not np.array_equal(cut, cut_again):
                    raise ValueError('Damage replay mismatch')
                if kind != 'intact' and (np.array_equal(damaged,target) or not damaged.any()):
                    raise ValueError('Degenerate damage; retain failure without reroll')
                repaired = closing_repair(damaged, domain, fields['permitted'])
                metrics = {}
                for method, candidate in [('unchanged',damaged), ('closing3',repaired)]:
                    report, _ = evaluate_targets(candidate,scene,fields,domain)
                    metrics[method] = {'targets':report, **repair_metrics(candidate,target,damaged,domain,request)}
                name = 'examples/'+key+'__'+kind
                checksum = arrays(name+'.npz', target=target, damaged=damaged, cut_region=cut,
                                  removed=target & ~damaged, closing3=repaired,
                                  condition=condition(scene,fields,domain,request))
                example = {'case':key, 'split':split, 'site':task['site'], 'damage':kind,
                           'arrays':name+'.npz', 'arrays_sha256':checksum, 'metrics':metrics}
                # Audit the same public loader future training will use.
                loaded, label = load_example(d, example, split=split)
                if set(loaded) != {'occupancy','context'} or not np.array_equal(label,target):
                    raise ValueError('Loader label/input contract mismatch')
                if not np.array_equal(loaded['occupancy'],damaged) or loaded['context'].shape != (len(CHANNELS),)+target.shape:
                    raise ValueError('Loader input shape/content mismatch')
                save(name+'.json', example); examples.append(example)
            store.event(run, 'target_completed', 'Teacher and all frozen variants retained', case=key)
            print(f'{len(targets)}/54 teachers prepared: {key}', flush=True)
        integrity = assert_split_integrity(targets)
        if (len(targets),len(examples),len(guards)) != (54,162,9):
            raise ValueError('Frozen case count mismatch')
    except Exception:
        status, failure = 'failed', traceback.format_exc()
        integrity = None
        store.event(run, 'failure', 'Preparation stopped; partial evidence retained', traceback=failure)

    groups = defaultdict(list)
    for example in examples:
        for method, m in example['metrics'].items():
            groups[(example['split'],example['damage'],method)].append(m)
    aggregate = []
    for (split,kind,method), rows in sorted(groups.items()):
        aggregate.append({'split':split, 'damage':kind, 'method':method, 'count':len(rows),
            'all_nine_pass':sum(m['targets']['contract_pass'] for m in rows),
            'median_iou':float(np.median([m['iou'] for m in rows])),
            'total_removed_survivors':sum(m['surviving_cells_removed'] for m in rows),
            'total_false_additions':sum(m['false_positive_cells'] for m in rows)})
    metrics = {'admission_gate':status == 'completed', 'teachers':len(targets),
        'examples':len(examples), 'blocked_guards':len(guards), 'split_integrity':integrity,
        'wall_seconds':time.perf_counter()-started, 'learned_updates':0}
    result = {'version':VERSION, 'run_id':run, 'recipe':recipe, 'status':status,
        'metrics':metrics, 'targets':targets, 'examples':examples, 'guards':guards,
        'aggregate':aggregate, 'failure':failure}
    save('study.json',result)
    interpretation = 'Frozen data and nonlearned controls only. No trained NCA, held-out model performance or recovery certificate.'
    store.finish(run,status,metrics,interpretation)
    write_once(REPO/'experiments/records'/f'{run}.json', {'run_id':run,'status':status,
        'metrics':metrics,'interpretation':interpretation,'artifact_location':str(d),'drive_backup':'pending'})
    print(json.dumps({'run_id':run,'status':status,'metrics':metrics,'failure':failure},indent=2),flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
