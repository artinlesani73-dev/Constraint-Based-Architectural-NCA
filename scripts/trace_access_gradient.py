"""Post-hoc exact replay locating L1's zero access parameter gradient."""
from pathlib import Path
import sys
import traceback
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.experiments import RunStore, provenance, snapshot_source, read_json, write_once, digest
from deploy.checkpoints import load_model_c
from deploy.model_utils import UrbanPavilionNCA, LocalLegalityLoss


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    parent = sys.argv[1]
    store = RunStore(REPO / '.local-artifacts/runs')
    if store.verify(parent): raise ValueError('Parent artifacts invalid')
    directory = store.path(parent)
    if read_json(directory / 'result.json')['status'] != 'completed': raise ValueError('Parent incomplete')
    parent_config = read_json(directory / 'run.json')['config']
    rows = [read_json(directory / event['details']['path'])
            for path in sorted((directory / 'events').glob('*.json')) for event in [read_json(path)]
            if event['kind'] == 'artifact' and event['details']['role'] == 'model_record']
    row, = [r for r in rows if r['scenes'] == ['ref-01-ground-pair']]
    config, weights, checkpoint = load_model_c()
    if digest(checkpoint) != parent_config['checkpoint_sha256']: raise ValueError('Checkpoint differs')
    run = store.create('L1 post-hoc access clamp replay', 'gradient_attribution',
                       {'input_run': parent, 'profile': row['rollout']['profile'], 'optimizer_updates': 0,
                        'post_hoc': True, 'checkpoint_sha256': digest(checkpoint)}, 0, provenance(REPO), parent_run=parent)
    out = store.path(run)
    status, details = 'completed', {}
    try:
        source = out / 'source.zip'; snapshot_source(REPO, source)
        store.attach(run, source, 'source_snapshot'); source.unlink()
        with np.load(directory / row['fields']['path'], allow_pickle=False) as arrays:
            seed = torch.from_numpy(arrays['seed_state'].copy())
            scaffold = torch.from_numpy(arrays['scaffold'].copy())
            expected = torch.from_numpy(arrays['final_state'].copy())
            gradient = torch.from_numpy(arrays['occupancy_access'].copy())
        profile = row['rollout']['profile']
        assert profile['firing'] == 'delta_mask' and profile['noise_std'] == 0
        assert row['rollout']['applied']['seed_mask_strength'] == 0
        config = dict(config, fire_rate=profile['fire_rate'], update_scale=profile['update_scale'])
        model = UrbanPavilionNCA(config); model.load_state_dict(weights); model.train()
        state = seed.clone()
        state[:, config['ch_structure']] = torch.clamp(state[:, config['ch_structure']] + profile['corridor_seed_scale'] * scaffold, 0, 1)
        generator = torch.Generator().manual_seed(0)
        captured = {}
        hook = model.update_net.register_forward_hook(lambda module, inputs, output: captured.update(delta=output.detach()))
        with torch.no_grad():
            for step in range(row['rollout']['steps_run']):
                duplicate = torch.Generator(); duplicate.set_state(generator.get_state())
                firing = (torch.rand(state.shape[0], 1, *state.shape[-3:], generator=duplicate) < config['fire_rate']).float()
                before = state
                state = model._step(state, generator=generator)
                raw = before[:, config['n_frozen']:] + config['update_scale'] * captured['delta'] * firing
        hook.remove()
        if not torch.equal(state, expected): raise ValueError('Replay differs from saved L1 state')
        legality = LocalLegalityLoss(config).compute_legality_field(seed)
        cells = []
        for coordinate in torch.nonzero(gradient, as_tuple=False):
            index = tuple(int(x) for x in coordinate)
            b, z, y, x = index
            cells.append({'bzyx': list(index), 'occupancy_gradient': float(gradient[index]),
                          'raw_structure_before_clamp': float(raw[b, 0, z, y, x]),
                          'final_occupancy': float(state[b, config['ch_structure'], z, y, x]),
                          'legality': float(legality[index]), 'fired': bool(firing[b, 0, z, y, x])})
        details = {'input_run': parent, 'input_fields': row['fields'], 'post_hoc': True,
                   'full_state_bitwise_equal': True, 'active_access_derivative_cells': cells,
                   'all_active_cells_strictly_below_clamp': bool(cells) and all(c['raw_structure_before_clamp'] < 0 and c['legality'] == 1 for c in cells),
                   'access_parameter_gradient_l2': row['gradient_norms']['access']['parameter_l2'],
                   'interpretation': 'Exact replay attributes the zero final-step derivative at these cells to the lower hard clamp, if raw values are strictly negative. This does not justify a particular replacement parameterization or establish behavior at longer horizons.'}
        path = out / 'raw.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, raw_grown=raw.numpy(), final_state=state.numpy(), occupancy_gradient=gradient.numpy(), legality=legality.numpy())
        details['fields'] = store.attach(run, path, 'attribution_fields'); path.unlink()
    except Exception:
        status = 'failed'; details['error'] = traceback.format_exc()
    write_once(out / 'analysis.json', details)
    store.attach(run, out / 'analysis.json', 'analysis')
    store.finish(run, status, {'optimizer_updates': 0}, 'Post-hoc local derivative attribution; no training')
    summary = {'run_id': run, 'status': status, **details}
    write_once(REPO / 'experiments/records' / (run + '.json'), summary)
    print(summary)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
