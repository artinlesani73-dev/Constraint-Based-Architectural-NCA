"""Post-hoc C1 check: can zero spill coexist with the notebook's volume floor?"""
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.experiments import RunStore, read_json, write_once
from scripts.report_corridor_comparison import load_verified


def analyze(run_id):
    protocol, summary, targets, _ = load_verified(run_id)
    directory = RunStore(REPO / '.local-artifacts/runs').path(run_id)
    sources = [e['details']['path'] for p in (directory / 'events').glob('*.json')
               for e in [read_json(p)] if e['kind'] == 'artifact' and e['details']['role'] == 'source_snapshot']
    if len(sources) != 1:
        raise ValueError('Expected one source snapshot')
    notebook = 'notebooks/model_c/NB02_AllConstraints_v3_1_C.ipynb'
    with ZipFile(directory / sources[0]) as archive:
        raw = archive.read(notebook)
    parsed = json.loads(raw)
    bounds, calls = [], []
    for index, cell in enumerate(parsed['cells']):
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'SparsityLossV31':
                init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
                defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],
                                    [ast.literal_eval(v) for v in init.args.defaults]))
                bounds.append((index, defaults))
            if isinstance(node, ast.ClassDef) and any(isinstance(method, ast.FunctionDef) and method.name == 'train_epoch' for method in node.body):
                initializer = next(method for method in node.body if isinstance(method, ast.FunctionDef) and method.name == '__init__')
                calls.extend((index, call) for call in ast.walk(initializer)
                             if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'SparsityLossV31')
    if len(bounds) != 1 or len(calls) != 1:
        raise ValueError('Ambiguous historical sparsity definition or constructor')
    index, defaults = bounds[0]
    call_index, call = calls[0]
    if call.args or any(k.arg != 'squared' for k in call.keywords):
        raise ValueError('Historical constructor overrides need explicit interpretation')
    floor = defaults['min_ratio']
    rows = []
    config = protocol['effective_model_config']
    for record in sorted(targets, key=lambda t: (t['scene_set'], t['scene_id'], t['version'])):
        if record['version'] != 'corridor_legal_v1':
            continue
        with np.load(directory / record['fields']['path'], allow_pickle=False) as arrays:
            # Notebook train_epoch: available = 1.0 - existing. This is not the
            # smaller permitted field used by the corrected target generator.
            available = float((1.0 - arrays['seed_state'][0, config['ch_existing']]).sum())
            mass = int((arrays['corridor_legal_v1'][0] > 0.5).sum())
        ratio = mass / available
        rows.append({'scene_set': record['scene_set'], 'scene_id': record['scene_id'],
                     'legal_target_voxels': mass, 'historical_available_voxels': available,
                     'max_zero_spill_mass_ratio': ratio, 'historical_min_ratio': floor,
                     'zero_spill_below_volume_floor': ratio < floor,
                     'minimum_extra_mass_to_meet_floor': max(0.0, floor * available - mass)})
    return {'run_id': run_id, 'analysis': 'C1_volume_compatibility_posthoc_v1',
            'pre_registered': False, 'source_snapshot': sources[0],
            'notebook': notebook, 'notebook_sha256': sha256(raw).hexdigest(),
            'definition_cell_zero_based': index, 'constructor_cell_zero_based': call_index,
            'historical_defaults': defaults, 'rows': rows,
            'interpretation': 'For occupancy in [0,1], filling the legal target gives the largest zero-spill mass. Where that mass is below the historical floor, zero spill and zero lower-volume penalty cannot coexist. These are competing soft objectives, not a proof that no feasible architectural design exists. No weights or thresholds were changed.'}


def main():
    run_id = sys.argv[1]
    result = analyze(run_id)
    destination = REPO / 'docs/next-phase/reports' / (run_id + '-C1.volume-audit.json')
    write_once(destination, result)
    print(json.dumps({'path': str(destination), 'conflicting_cases': sum(r['zero_spill_below_volume_floor'] for r in result['rows']), 'cases': len(result['rows'])}))


if __name__ == '__main__':
    main()
