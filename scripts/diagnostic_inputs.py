"""Read verified frozen C1 inputs for subsequent local diagnostics."""
from pathlib import Path
import numpy as np
import torch
from nca.experiments import RunStore, digest
from nca.contract import load_reference_set
from scripts.report_corridor_comparison import load_verified

C1_RUN = '20260923T000945Z_3cbdc3603a12'


def load_inputs(repo):
    protocol, _, targets, _ = load_verified(C1_RUN)
    directory = RunStore(repo / '.local-artifacts/runs').path(C1_RUN)
    sets = {name: load_reference_set(repo / 'experiments/scenes' / name) for name in protocol['scene_manifest_sha256']}
    for name in sets:
        if digest(repo / 'experiments/scenes' / name / 'manifest.json') != protocol['scene_manifest_sha256'][name]:
            raise ValueError('Frozen scenes differ from C1')
    result = {}
    for target in targets:
        if target['version'] != 'corridor_legal_v1':continue
        with np.load(directory / target['fields']['path'],allow_pickle=False) as data:
            seed=torch.from_numpy(data['seed_state'].copy())
            guide=torch.from_numpy(data['legal_centerline'].copy())>.5
            scaffold=torch.from_numpy(data['corridor_legal_v1'].copy())
            permitted=torch.from_numpy(data['permitted'].copy())[None]
        result[target['scene_id']]={'seed':seed,'guide':guide,'scaffold':scaffold,'permitted':permitted,
            'scene':sets[target['scene_set']][target['scene_id']], 'scene_set':target['scene_set'],
            'feasible':torch.tensor([target['router']['all_endpoints_connected']]),
            'source_fields':target['fields'],'scene_hash':target['scene_hash']}
    return protocol,result
