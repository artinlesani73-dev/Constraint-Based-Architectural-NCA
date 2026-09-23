"""Exact CPU checkpoint boundaries for local optimizer-recovery verification.

Save only between completed updates. Restore into fresh model/optimizer/scheduler
instances with identical metadata; GPU/Colab recovery is not yet certified.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch

SCHEMA = 'cpu_training_checkpoint_v1'


def metadata_hash(metadata):
    return sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _cpu(model, generator):
    if any(p.device.type != 'cpu' for p in model.parameters()) or generator.device.type != 'cpu':
        raise ValueError('This recovery implementation is certified only for CPU')


def save_checkpoint(path, model, optimizer, scheduler, generator, metadata, completed_updates):
    _cpu(model, generator)
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 0:
        raise ValueError('Invalid completed update count')
    np_state = np.random.get_state()
    payload = {'schema': SCHEMA, 'metadata': metadata, 'metadata_hash': metadata_hash(metadata),
               'completed_updates': completed_updates, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'optimizer_class': optimizer.__class__.__module__ + '.' + optimizer.__class__.__name__,
               'scheduler_class': scheduler.__class__.__module__ + '.' + scheduler.__class__.__name__,
               'parameter_names': [name for name, _ in model.named_parameters()],
               'module_training': model.training, 'python_rng': random.getstate(),
               'numpy_rng': {'name': np_state[0], 'keys': torch.from_numpy(np_state[1].astype(np.int64)),
                             'position': np_state[2], 'has_gauss': np_state[3], 'gauss': np_state[4]},
               'torch_rng': torch.get_rng_state(), 'firing_rng': generator.get_state()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + uuid.uuid4().hex + '.pt.tmp')
    try:
        with temp.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            raise
        except OSError:
            with temp.open('rb') as source, path.open('xb') as destination:
                shutil.copyfileobj(source, destination)
                destination.flush(); os.fsync(destination.fileno())
    finally:
        temp.unlink(missing_ok=True)
    return path


def restore_checkpoint(path, model, optimizer, scheduler, generator, expected_metadata):
    _cpu(model, generator)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != SCHEMA or payload['metadata_hash'] != metadata_hash(payload['metadata']):
        raise ValueError('Unsupported checkpoint or corrupt metadata')
    if payload['metadata_hash'] != metadata_hash(expected_metadata):
        raise ValueError('Checkpoint configuration/scenes/source metadata do not match')
    if payload['parameter_names'] != [name for name, _ in model.named_parameters()]:
        raise ValueError('Parameter ordering differs')
    for key, instance in (('optimizer_class', optimizer), ('scheduler_class', scheduler)):
        if payload[key] != instance.__class__.__module__ + '.' + instance.__class__.__name__:
            raise ValueError('Optimizer/scheduler class differs')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    model.train(payload['module_training'])
    random.setstate(payload['python_rng'])
    n = payload['numpy_rng']
    np.random.set_state((n['name'], n['keys'].numpy().astype(np.uint32), n['position'], n['has_gauss'], n['gauss']))
    torch.set_rng_state(payload['torch_rng'])
    generator.set_state(payload['firing_rng'])
    return payload['completed_updates']


def tree_equal(a, b):
    """Exact recursive comparison for full optimizer/RNG/checkpoint evidence."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a == b
