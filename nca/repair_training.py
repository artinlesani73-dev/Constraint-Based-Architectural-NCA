"""NR1 versioned CPU repair NCA and completed-update recovery; no production model."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import os
import platform
import random
import uuid
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nca.experiments import digest, write_once, read_json
from nca.recovery import metadata_hash

VERSION = 'volume_repair_nca_cpu_v1'


def perceive(x):
    """Identity, central X/Y/Z differences in cell units; zero exterior padding."""
    p = F.pad(x, (1,1,1,1,1,1))
    dx = (p[:,:,1:-1,1:-1,2:] - p[:,:,1:-1,1:-1,:-2]) * .5
    dy = (p[:,:,1:-1,2:,1:-1] - p[:,:,1:-1,:-2,1:-1]) * .5
    dz = (p[:,:,2:,1:-1,1:-1] - p[:,:,:-2,1:-1,1:-1]) * .5
    return torch.cat((x,dx,dy,dz), dim=1)


class RepairNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv3d(60,64,1)
        self.last = nn.Conv3d(64,8,1)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, state, static_features, allowed, generator):
        features = torch.cat((perceive(state),static_features), dim=1)
        delta = self.last(F.relu(self.first(features)))
        fire = torch.rand(state.shape[0],1,*state.shape[2:], generator=generator,
                          device=state.device) < .5
        updated = state + delta*fire
        # Logits stay raw; only hidden state is reset outside the legal domain.
        return torch.cat((updated[:,:1], updated[:,1:]*allowed), dim=1)

    def rollout(self, occupancy, static_features, allowed, generator, steps=16):
        state = torch.cat((occupancy*4-2, torch.zeros_like(occupancy).expand(-1,7,-1,-1,-1)),dim=1)
        for _ in range(steps):
            state = self(state,static_features,allowed,generator)
        return state


def balanced_loss(logits, target, allowed):
    if logits.shape != target.shape or target.shape != allowed.shape:
        raise ValueError('Loss shape mismatch')
    positive = allowed & target.bool()
    negative = allowed & ~target.bool()
    if not positive.any() or not negative.any():
        raise ValueError('Both target classes must exist inside legal domain')
    return .5*F.softplus(-logits[positive]).mean() + .5*F.softplus(logits[negative]).mean()


def rng_state(generator):
    n = np.random.get_state()
    return {'python':random.getstate(), 'numpy':(n[0],n[1].astype(np.int64).tolist(),n[2],n[3],n[4]),
            'torch':torch.get_rng_state(), 'firing':generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch']); generator.set_state(state['firing'])


def save_payload(path, payload, *, fault=None):
    """Manifest is the commit marker. Failed payload/partial files remain for diagnosis.

    Atomic exclusive hard-link publication requires a local filesystem supporting
    hard links. No FUSE/Drive fallback or cloud-recovery claim is made.
    """
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    receipt = path.with_suffix('.json')
    if path.exists() or receipt.exists():
        raise FileExistsError(path)
    temp = path.with_name('.'+uuid.uuid4().hex+'.partial.pt')
    with temp.open('xb') as stream:
        torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    if fault == 'before_publish':
        raise OSError('Injected interruption before payload publication')
    os.link(temp,path)
    temp.unlink()  # Successful identical payload is now retained at the final path.
    if fault == 'before_manifest':
        raise OSError('Injected interruption before commit manifest')
    write_once(receipt, {'version':VERSION,'sha256':digest(path),'bytes':path.stat().st_size,
                        'completed':payload['completed'], 'identity_sha256':metadata_hash(payload['identity'])})
    return path


def read_payload(path, identity):
    path = Path(path); receipt = read_json(path.with_suffix('.json'))
    if receipt['version'] != VERSION or receipt['identity_sha256'] != metadata_hash(identity):
        raise ValueError('Checkpoint identity differs')
    if path.stat().st_size != receipt['bytes'] or digest(path) != receipt['sha256']:
        raise ValueError('Checkpoint payload corrupt')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload['version'] != VERSION or payload['identity'] != identity or payload['completed'] != receipt['completed']:
        raise ValueError('Checkpoint payload identity/cursor differs')
    if type(payload['completed']) is not int or payload['completed'] < 0:
        raise ValueError('Invalid completed cursor')
    if payload['next_sample_cursor'] != payload['completed'] or len(payload['trace']) != payload['completed']:
        raise ValueError('Checkpoint trace/cursor differs')
    if any(row['update'] != i+1 or row['sample_cursor'] != i for i,row in enumerate(payload['trace'])):
        raise ValueError('Checkpoint trace ordering differs')
    return payload


def latest_verified(directory, identity):
    rejected = []
    for receipt in sorted(Path(directory).glob('checkpoint-*.json'),reverse=True):
        path = receipt.with_suffix('.pt')
        try:
            read_payload(path,identity)
            return path,rejected
        except (ValueError, OSError, KeyError, EOFError) as error:
            rejected.append({'path':str(path),'reason':str(error)})
    raise ValueError('No verified completed checkpoint: '+str(rejected))


class RepairSession:
    def __init__(self, inputs, target, identity, seed=1201):
        # CPU only until an independently checked GPU implementation is admitted.
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.identity = deepcopy(identity)
        self.model = RepairNCA().float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
                                          weight_decay=0,amsgrad=False,foreach=False,fused=False)
        self.generator = torch.Generator(device='cpu').manual_seed(seed+1)
        self.occupancy = torch.tensor(inputs['occupancy'],dtype=torch.float32)[None,None]
        self.context = torch.tensor(inputs['context'],dtype=torch.float32)[None]
        self.target = torch.tensor(target,dtype=torch.float32)[None,None]
        if self.context.ndim != 5 or self.context.shape[:2] != (1,7) or self.context.shape[2:] != self.occupancy.shape[2:]:
            raise ValueError('One occupancy field and seven aligned context channels required')
        for tensor in (self.occupancy,self.target,self.context):
            if not torch.isfinite(tensor).all():raise ValueError('Nonfinite input')
        if self.target.shape != self.occupancy.shape:
            raise ValueError('Target shape mismatch')
        for tensor in (self.occupancy,self.target,self.context[:,:6]):
            if not ((tensor==0)|(tensor==1)).all():raise ValueError('Binary geometry required')
        request = self.context[:,6:7]
        if not (request==request.flatten()[0]).all() or not 0<float(request.flatten()[0])<=1:
            raise ValueError('Constant request fraction required')
        self.allowed = (self.context[:,:1]>0) & (self.context[:,1:2]>0)
        if (self.occupancy.bool() & ~self.allowed).any() or (self.target.bool() & ~self.allowed).any():
            raise ValueError('Repair examples must lie in permitted domain')
        self.static_features = perceive(self.context).detach()
        self.identity = {'experiment':self.identity,'version':VERSION,
            'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),
                       'numpy':str(np.__version__),'device':'cpu','dtype':'float32','threads':2,
                       'deterministic_algorithms':True,'seed':seed,'firing_seed':seed+1},
            'tensor_hashes':{name:sha256(str(tuple(t.shape)).encode()+t.numpy().tobytes()).hexdigest()
                            for name,t in [('occupancy',self.occupancy),('context',self.context),('target',self.target)]}}
        self._immutable = [(x,x._version) for x in (self.occupancy,self.context,self.target,self.allowed,self.static_features)]
        self.completed = 0; self.trace = []
        balanced_loss(self.occupancy,self.target,self.allowed)

    def check_inputs(self):
        if any(t._version != version for t,version in self._immutable):
            raise ValueError('Session context/target/input was mutated')

    def step(self):
        self.check_inputs(); self.optimizer.zero_grad(set_to_none=True)
        state = self.model.rollout(self.occupancy,self.static_features,self.allowed,self.generator)
        loss = balanced_loss(state[:,:1],self.target,self.allowed)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise FloatingPointError('Missing/nonfinite parameter gradient')
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        evidence = {'update':self.completed+1,'sample_cursor':self.completed,'loss':float(loss.detach()),
                    'pre_clip_gradient_norm':float(norm),'clipped':bool(norm>1)}
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError('Nonfinite model after update')
        self.completed += 1; self.trace.append(evidence)
        return evidence,state.detach().numpy()[0].copy()

    def evaluate(self, steps=16, firing_seed=2101):
        self.check_inputs()
        generator = torch.Generator(device='cpu').manual_seed(firing_seed)
        with torch.no_grad():
            state = self.model.rollout(self.occupancy,self.static_features,self.allowed,generator,steps)
            probability = torch.sigmoid(state[:,:1])*self.allowed
            loss = balanced_loss(state[:,:1],self.target,self.allowed)
        return {'loss':float(loss),'completed':self.completed,'steps':steps,'firing_seed':firing_seed}, \
            state.numpy()[0].copy(),probability.numpy()[0,0].copy()

    def payload(self):
        return {'version':VERSION,'identity':deepcopy(self.identity),'completed':self.completed,
                'next_sample_cursor':self.completed,'trace':deepcopy(self.trace),
                'model':deepcopy(self.model.state_dict()),'optimizer':deepcopy(self.optimizer.state_dict()),
                'optimizer_class':'torch.optim.adam.Adam','parameter_names':[n for n,_ in self.model.named_parameters()],
                'module_training':self.model.training,'rng':rng_state(self.generator)}

    def save(self,path):
        self.check_inputs(); return save_payload(path,self.payload())

    def restore(self,path):
        self.check_inputs(); p = read_payload(path,self.identity)
        if p['parameter_names'] != [n for n,_ in self.model.named_parameters()] or p['optimizer_class'] != 'torch.optim.adam.Adam':
            raise ValueError('Model/optimizer identity differs')
        self.model.load_state_dict(p['model'],strict=True); self.optimizer.load_state_dict(p['optimizer'])
        self.model.train(p['module_training']); self.completed=p['completed']; self.trace=p['trace']
        restore_rng(p['rng'],self.generator)
        return self.completed
