"""Render L1 findings from verified immutable diagnostic records."""
from collections import defaultdict
from pathlib import Path
import sys
import math
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.experiments import RunStore, read_json, write_once, digest
from nca.losses import FAMILIES


def load(run_id):
    store = RunStore(REPO / '.local-artifacts/runs')
    if store.verify(run_id):
        raise ValueError('Artifact verification failed')
    directory = store.path(run_id)
    if read_json(directory / 'result.json')['status'] != 'completed':
        raise ValueError('Diagnostic did not complete')
    records = defaultdict(list)
    for path in sorted((directory / 'events').glob('*.json')):
        event = read_json(path)
        if event['kind'] == 'artifact' and event['details']['role'] in ('protocol', 'summary', 'context_record', 'model_record', 'historical_record'):
            records[event['details']['role']].append(read_json(directory / event['details']['path']))
    if any(len(records[role]) != expected for role, expected in
           [('protocol', 1), ('summary', 1), ('context_record', 72), ('model_record', 3), ('historical_record', 6)]):
        raise ValueError('Incomplete registered matrix')
    keys = [(r['scene_set'], r['scene_id'], r['envelope']) for r in records['context_record']]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate context')
    # Verify recorded gradient norms/cosines against the actual saved arrays.
    for row in records['model_record']:
        with np.load(directory / row['fields']['path'], allow_pickle=False) as data:
            vectors = {}
            for name in FAMILIES:
                occupancy = data['occupancy_' + name].astype(np.float64)
                vector = data['parameters_' + name].astype(np.float64)
                vectors[name] = vector
                if not np.isfinite(occupancy).all() or not np.isfinite(vector).all():
                    raise ValueError('Nonfinite saved gradient')
                if not math.isclose(float(np.linalg.norm(vector)), row['gradient_norms'][name]['parameter_l2'], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Parameter gradient norm mismatch')
                if not math.isclose(float(np.linalg.norm(occupancy)), row['gradient_norms'][name]['occupancy_l2'], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Occupancy gradient norm mismatch')
            for a in FAMILIES:
                for b in FAMILIES:
                    recorded = row['parameter_gradient_cosines'][a][b]
                    denominator = np.linalg.norm(vectors[a]) * np.linalg.norm(vectors[b])
                    if denominator == 0:
                        if recorded is not None: raise ValueError('Zero-norm cosine must be null')
                    elif not math.isclose(float(np.dot(vectors[a], vectors[b])/denominator), recorded, rel_tol=1e-9, abs_tol=1e-11):
                        raise ValueError('Gradient cosine mismatch')
    return records


def render(run_id):
    records = load(run_id)
    summary = records['summary'][0]
    lines = [f'# L1 loss diagnostic: {run_id}', '',
             f'Source commit `{summary["provenance"]["commit"]}`. All registered artifacts '
             'verified. Saved gradient arrays independently reproduce the recorded norms '
             'and pairwise cosines. Zero optimizer updates.', '',
             '## Objective context checks', '',
             '| Envelope | Cases | Satisfy capacity bounds | Valid context including route |',
             '|---|---:|---:|---:|']
    for name, row in summary['envelope_summary'].items():
        lines.append(f'| {name} | {row["cases"]} | {row["budget_compatible"]} | {row["valid_contexts"]} |')
    lines += ['', 'These are necessary checks, not proof that all objectives are jointly '
              'satisfied. The sealed reference remains infeasible. Radii are fixed graph '
              'distances (3/6 voxels = 2.4/4.8m); they were not adjusted per case. The '
              'permitted-space envelope is a control in which spill is redundant, not a '
              'recommended design envelope. No training envelope is selected.', '',
              '| Scene | Envelope | Guide voxels | Envelope capacity | Minimum mass | Route feasible | Context valid |',
              '|---|---|---:|---:|---:|---|---|']
    for row in records['context_record']:
        r = row['results']
        lines.append(f'| {row["scene_id"]} | {row["envelope"]} | {r["coverage_voxels"][0]:.0f} | '
                     f'{r["envelope_capacity"][0]:.0f} | {r["minimum_mass"][0]:.2f} | '
                     f'{r["route_feasible"][0]} | {r["context_valid"][0]} |')
    lines += ['', '## Gradients through the original checkpoint', '',
              'Four historical-training updates, seed 0, radius-six envelope, 64-hop '
              'proxies. These are local derivatives at three diagnostic states, not '
              'training results or a representative distribution of gradients.', '',
              '| Scene(s) | B | Term | Mean value | Occupancy gradient L2 | Parameter gradient L2 |',
              '|---|---:|---|---:|---:|---:|']
    for row in records['model_record']:
        for name in FAMILIES:
            norm = row['gradient_norms'][name]
            mean = sum(row['results']['terms'][name])/row['batch_size']
            lines.append(f'| {", ".join(row["scenes"])} | {row["batch_size"]} | {name} | '
                         f'{mean:.7g} | {norm["occupancy_l2"]:.7g} | {norm["parameter_l2"]:.7g} |')
    lines += ['', '### Parameter gradient alignment', '',
              'Selected pairs below; every pair is retained in the JSON model records. '
              'Negative cosine indicates local opposing derivatives, not proof of global '
              'infeasibility. Zero-norm pairs are undefined (null).', '',
              '| Scene(s) | Coverage / sparsity | Coverage / spill | Coverage / thickness | Coverage / access |',
              '|---|---:|---:|---:|---:|']
    def number(value):
        return 'null' if value is None else f'{value:.6f}'
    for row in records['model_record']:
        cosines = row['parameter_gradient_cosines']['coverage']
        lines.append('| ' + ', '.join(row['scenes']) + ' | ' + ' | '.join(number(cosines[k]) for k in ('sparsity', 'spill', 'thickness', 'access')) + ' |')
    lines += ['', '## Historical fine-tuner outcomes', '',
              '| Class | Batch | Outcome | Output requires gradients |', '|---|---:|---|---|']
    for row in records['historical_record']:
        lines.append(f'| {row["class"]} | {row["batch_size"]} | {row["status"]} | {row.get("requires_grad", "not reached")} |')
    lines += ['', 'Only these original class definitions were executed, without importing '
              'the trainer or starting optimization. Exact forward tracebacks and source '
              'hashes are retained. Historical defects are expected findings, not failures '
              'of the new run.', '', '## Limits and next decisions', '',
              'The new losses use separate per-scene reductions and explicit context '
              'validity. Empty material has an explicit nonempty flag; a zero thickness '
              'term on empty space is not architectural success. Legality/ground parameter '
              'gradients can be zero because the model projects those voxels away.', '',
              'The reach surrogate has a finite hop horizon and nonsmooth max/min ties. '
              'A valid finite gradient on a short diagnostic rollout does not establish '
              'useful learning on broken routes, other seeds or longer rollouts. The '
              'coverage guide still supplies a prescribed material connection; access '
              'through protected street void and elevated deck/headroom semantics remain '
              'unresolved. Full optimizer/recovery tests, regularizer integration, loss '
              'weight calibration, learned-value controls and held-out evaluation remain '
              'pending. No new weights, paid training, production defaults or cloud access.', '',
              f'Raw fields/gradients and source snapshot: `.local-artifacts/runs/{run_id}/`.', '']
    return '\n'.join(lines), records


def main():
    run_id = sys.argv[1]
    report, records = render(run_id)
    base = REPO / 'docs/next-phase/reports' / (run_id + '-L1')
    with base.with_suffix('.md').open('x', encoding='utf-8') as stream:
        stream.write(report)
    write_once(base.with_suffix('.details.json'), dict(records))
    write_once(base.with_suffix('.verification.json'), {'run_id': run_id, 'artifact_hashes_verified': True,
               'gradient_norms_and_cosines_recomputed': True, 'report_sha256': digest(base.with_suffix('.md'))})
    print(base.with_suffix('.md'))


if __name__ == '__main__':
    main()
