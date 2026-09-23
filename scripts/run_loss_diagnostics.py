"""Run the local L1_v1 loss/gradient diagnostic; never perform optimizer updates."""
import argparse
import ast
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA
from nca.contract import load_reference_set
from nca.experiments import RunStore, provenance, snapshot_source, write_once, read_json, digest
from nca.losses import (LossSpec, LOSS_VERSION, context_from_scenes, material_envelope, loss_terms, FAMILIES)
from nca.rollout import historical_training, run_rollout
from scripts.report_corridor_comparison import load_verified

C1 = '20260923T000945Z_3cbdc3603a12'


def describe(result):
    return {name: value.detach().cpu().tolist() for name, value in result.items()
            if isinstance(value, torch.Tensor)} | {
            'terms': {name: term.detach().cpu().tolist() for name, term in result['terms'].items()},
            'applicable': {name: flag.cpu().tolist() for name, flag in result['applicable'].items()},
            'version': result['version'], 'reach_hops': result['reach_hops']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config, weights, checkpoint = load_model_c()
    spec = LossSpec()
    profile = historical_training(config).replace(name='loss-gradient-diagnostic', rng_source='explicit')
    protocol = {'protocol': 'L1_v1', 'loss_version': LOSS_VERSION, 'loss_spec': asdict(spec),
                'input_run': C1, 'checkpoint_sha256': digest(checkpoint), 'effective_model_config': config,
                'device': 'cpu', 'torch_threads': 2, 'deterministic_algorithms': True,
                'optimizer_updates': 0, 'expected_contexts': 72, 'expected_model_cases': 3,
                'expected_historical_checks': 6, 'model_steps': 4, 'seed': 0,
                'profile': profile.as_dict(), 'schedule_position': 60,
                'envelopes': ['scaffold', 'radius3', 'radius6', 'permitted-control'],
                'coverage': 'C1 legal centerline including endpoint regions',
                'limits': 'Software and gradient diagnosis only. Necessary context checks do not certify all objectives jointly feasible. No envelope selected, no optimizer, no quality claim.'}
    store = RunStore(REPO / '.local-artifacts/runs')
    origin = provenance(REPO)
    run = store.create('L1 shared loss mechanics', 'loss_diagnostic', protocol, 0, origin, parent_run=args.parent_run)
    directory = store.path(run)
    print(f'RUN_ID={run}', flush=True)
    contexts, model_cases, historical = [], [], []
    status, error = 'completed', None
    started = time.perf_counter()
    def record(name, data, role):
        path = directory / (name + '.json')
        write_once(path, data)
        return store.attach(run, path, role)
    def arrays(name, **data):
        path = directory / (name + '.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **{key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
                                          for key, value in data.items()})
        item = store.attach(run, path, 'fields')
        path.unlink()
        return item
    try:
        source = directory / 'source.zip'
        snapshot_source(REPO, source)
        store.attach(run, source, 'source_snapshot')
        source.unlink()
        record('protocol', protocol, 'protocol')
        c1_protocol, _, target_rows, case_rows = load_verified(C1)
        if c1_protocol['checkpoint_sha256'] != protocol['checkpoint_sha256'] or c1_protocol['effective_model_config'] != config:
            raise ValueError('Checkpoint/config differs from recorded C1')
        frozen = {name: load_reference_set(REPO / 'experiments/scenes' / name) for name in ('reference_v1', 'legacy_easy_v1')}
        for name in frozen:
            if digest(REPO / 'experiments/scenes' / name / 'manifest.json') != c1_protocol['scene_manifest_sha256'][name]:
                raise ValueError('Frozen scenes differ from C1')
        prepared = {}
        for target in sorted(target_rows, key=lambda r: (r['scene_set'], r['scene_id'])):
            if target['version'] != 'corridor_legal_v1':
                continue
            key = (target['scene_set'], target['scene_id'])
            scene = frozen[key[0]][key[1]]
            case = next(c for c in case_rows if (c['scene_set'], c['scene_id']) == key
                        and c['version'] == 'corridor_legal_v1' and c['base_profile'] == 'historical-training')
            with np.load(store.path(C1) / target['fields']['path'], allow_pickle=False) as data:
                seed = torch.from_numpy(data['seed_state'].copy())
                guide = torch.from_numpy(data['legal_centerline'].copy()) > .5
                scaffold = torch.from_numpy(data['corridor_legal_v1'].copy()) > .5
                permitted = torch.from_numpy(data['permitted'].copy())[None]
            with np.load(store.path(C1) / case['fields']['path'], allow_pickle=False) as data:
                final = torch.from_numpy(data['final_state'].copy())
            feasible = torch.tensor([target['router']['all_endpoints_connected']])
            envelopes = {'scaffold': scaffold, 'radius3': material_envelope(guide, permitted, 3),
                         'radius6': material_envelope(guide, permitted, 6), 'permitted-control': permitted}
            fields = arrays(f'input-{len(prepared):02d}', seed_state=seed, saved_final_state=final,
                            coverage=guide, **{'envelope_' + name: value for name, value in envelopes.items()})
            for name, envelope in envelopes.items():
                context = context_from_scenes(seed, config, [scene], guide, envelope, feasible)
                tick = time.perf_counter()
                with torch.no_grad():
                    result = loss_terms(final[:, config['ch_structure']], context, spec)
                row = {'scene_set': key[0], 'scene_id': key[1], 'envelope': name,
                       'input_fields': fields, 'c1_seed_fields': target['fields'], 'c1_final_fields': case['fields'],
                       'results': describe(result), 'wall_seconds': time.perf_counter() - tick}
                record(f'context-{len(contexts):03d}', row, 'context_record')
                contexts.append(row)
            prepared[key[1]] = (seed, guide, scaffold, envelopes['radius6'], scene, feasible)
            print(f'CONTEXTS {len(contexts)}/72 {key[1]}', flush=True)
        model = UrbanPavilionNCA(dict(config))
        model.load_state_dict(weights, strict=True)
        parameter_items = list(model.named_parameters())
        layouts = [{'name': name, 'shape': list(value.shape), 'elements': value.numel()} for name, value in parameter_items]
        for names in (('legacy-easy-seed-000',), ('ref-01-ground-pair',), ('legacy-easy-seed-000', 'ref-01-ground-pair')):
            inputs = [prepared[name] for name in names]
            seed = torch.cat([item[0] for item in inputs])
            guide = torch.cat([item[1] for item in inputs])
            scaffold = torch.cat([item[2] for item in inputs]).to(seed.dtype)
            envelope = torch.cat([item[3] for item in inputs])
            scenes = [item[4] for item in inputs]
            feasible = torch.cat([item[5] for item in inputs])
            context = context_from_scenes(seed, config, scenes, guide, envelope, feasible)
            generator = torch.Generator().manual_seed(0)
            tick = time.perf_counter()
            rollout = run_rollout(model, seed, profile, corridor_target=scaffold, steps=4,
                                  schedule_position=60, generator=generator)
            final = rollout.pop('state')
            occupancy = final[:, config['ch_structure']]
            result = loss_terms(occupancy, context, spec)
            gradients, norms, vectors = {}, {}, {}
            for name in FAMILIES:
                values = torch.autograd.grad(result['terms'][name].mean(),
                                            (occupancy, *[p for _, p in parameter_items]), retain_graph=True, allow_unused=True)
                if values[0] is None:
                    raise ValueError(f'{name} disconnected from occupancy')
                flat = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                                  for g, (_, p) in zip(values[1:], parameter_items)])
                if not torch.isfinite(values[0]).all() or not torch.isfinite(flat).all():
                    raise ValueError(f'Nonfinite {name} gradient')
                gradients['occupancy_' + name] = values[0]
                gradients['parameters_' + name] = flat
                vectors[name] = flat.double()
                norms[name] = {'occupancy_l2': float(values[0].double().norm()),
                               'parameter_l2': float(vectors[name].norm()),
                               'unused_parameters': [label for g, (label, _) in zip(values[1:], parameter_items) if g is None]}
            cosines = {}
            for a in FAMILIES:
                cosines[a] = {}
                for b in FAMILIES:
                    product = float(vectors[a].norm() * vectors[b].norm())
                    cosines[a][b] = float(torch.dot(vectors[a], vectors[b]) / product) if product else None
            ref = arrays(f'grad-{len(model_cases):02d}', seed_state=seed, final_state=final,
                         coverage=guide, envelope=envelope, scaffold=scaffold, **gradients)
            row = {'scenes': list(names), 'batch_size': len(names), 'results': describe(result), 'rollout': rollout,
                   'fields': ref, 'parameter_layout': layouts, 'gradient_norms': norms,
                   'parameter_gradient_cosines': cosines, 'wall_seconds': time.perf_counter() - tick}
            record(f'model-{len(model_cases):02d}', row, 'model_record')
            model_cases.append(row)
            print(f'MODEL_GRADIENTS {len(model_cases)}/3 B={len(names)}', flush=True)
            del result, final, occupancy, gradients, vectors, values, rollout
        unchanged = all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
        if not unchanged:
            raise ValueError('Checkpoint parameters/buffers changed during diagnostic')
        source_path = REPO / 'scripts/finetune_porosity.py'
        tree = ast.parse(source_path.read_text(encoding='utf-8-sig'))
        for name in ('ThicknessLoss', 'PorosityLoss', 'SurfaceAreaLoss'):
            definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional}
            exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source_path), 'exec'), namespace)
            for batch in (1, 2):
                p = torch.full((batch, 7, 7, 7), .7, requires_grad=True)
                row = {'class': name, 'batch_size': batch, 'input_shape': list(p.shape),
                       'source_sha256': digest(source_path), 'expected_historical_defect': True}
                try:
                    module = namespace[name]()
                    value = module(p, torch.ones_like(p)) if name == 'PorosityLoss' else module(p)
                    row.update(value=float(value.detach()), requires_grad=value.requires_grad,
                               status='disconnected_gradient' if not value.requires_grad else 'gradient_present')
                except Exception:
                    row.update(status='forward_error', error=traceback.format_exc())
                record(f'historical-{len(historical):02d}', row, 'historical_record')
                historical.append(row)
        if (len(contexts), len(model_cases), len(historical)) != (72, 3, 6):
            raise ValueError('Incomplete L1 matrix')
    except KeyboardInterrupt:
        status, error = 'interrupted', traceback.format_exc()
    except Exception:
        status, error = 'failed', traceback.format_exc()
    summary = {'run_id': run, 'protocol': 'L1_v1', 'status': status, 'error': error,
               'context_cases': len(contexts), 'model_cases': len(model_cases), 'historical_checks': len(historical),
               'envelope_summary': {name: {'cases': len(rows),
                   'valid_contexts': sum(r['results']['context_valid'][0] for r in rows),
                   'budget_compatible': sum(r['results']['budget_compatible'][0] for r in rows),
                   'route_feasible': sum(r['results']['route_feasible'][0] for r in rows)}
                   for name in protocol['envelopes'] for rows in [[r for r in contexts if r['envelope'] == name]]},
               'model_gradients': [{k: row[k] for k in ('scenes', 'batch_size', 'gradient_norms', 'parameter_gradient_cosines')} for row in model_cases],
               'historical_outcomes': historical, 'optimizer_updates': 0, 'wall_seconds': time.perf_counter() - started,
               'provenance': origin, 'artifact_location': f'.local-artifacts/runs/{run}',
               'drive_backup': 'not_uploaded', 'limits': protocol['limits']}
    record('summary', summary, 'summary')
    if error:
        store.event(run, 'error', 'Diagnostic stopped; prior evidence retained', traceback=error)
    store.finish(run, status, {'contexts': len(contexts), 'model_cases': len(model_cases), 'historical_checks': len(historical)}, protocol['limits'])
    write_once(REPO / 'experiments/records' / (run + '.json'), summary)
    print(json.dumps({'run_id': run, 'status': status, 'error': error}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
