"""NR2 portable preflight: same NR1 math, explicit sampler and device-bound recovery."""
from copy import deepcopy
from pathlib import Path
import os
import platform
import random
import subprocess
import uuid
import numpy as np
import torch
from nca.experiments import digest, read_json, write_once
from nca.recovery import metadata_hash
from nca.repair_benchmark import load_example
from nca.repair_training import RepairNCA, perceive, balanced_loss, rng_state, restore_rng

VERSION = 'volume_repair_portable_preflight_v1'


def cpu_tree(value):
    if isinstance(value, torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value, dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):return tuple(cpu_tree(v) for v in value)
    return deepcopy(value)


class TrainingOrder:
    """Shuffle all training rows without replacement; separate CPU RNG."""
    def __init__(self, size, seed):
        if type(size) is not int or size<1:raise ValueError('Nonempty training set required')
        self.size=size;self.generator=torch.Generator(device='cpu').manual_seed(seed)
        self.order=torch.randperm(size,generator=self.generator).tolist()
        self.position=0;self.epoch=0;self.consumed=0

    def next(self):
        if self.position==self.size:
            self.order=torch.randperm(self.size,generator=self.generator).tolist()
            self.position=0;self.epoch+=1
        index=self.order[self.position];self.position+=1;self.consumed+=1
        return index

    def state(self):
        return {'size':self.size,'order':self.order.copy(),'position':self.position,'epoch':self.epoch,
                'consumed':self.consumed,'rng':self.generator.get_state()}

    def restore(self, state):
        if state['size']!=self.size or sorted(state['order'])!=list(range(self.size)):
            raise ValueError('Sampler dataset/permutation differs')
        if not 0<=state['position']<=self.size or state['epoch']<0 or state['consumed']!=state['epoch']*self.size+state['position']:
            raise ValueError('Sampler cursor inconsistent')
        self.order=state['order'].copy();self.position=state['position'];self.epoch=state['epoch'];self.consumed=state['consumed']
        self.generator.set_state(state['rng'])


def runtime(device):
    result={'host':platform.node(),'python':platform.python_version(),'torch':str(torch.__version__),'numpy':str(np.__version__),
            'device':str(device),'dtype':'float32','threads':2,'deterministic':True,
            'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'torch_build_configuration':torch.__config__.show(),
            'tf32_matmul':False,'tf32_cudnn':False,'cudnn_benchmark':False}
    if device.type=='cuda':
        p=torch.cuda.get_device_properties(device)
        result.update(gpu_name=p.name,gpu_capability=list(torch.cuda.get_device_capability(device)),
                      gpu_memory_bytes=p.total_memory,gpu_count=torch.cuda.device_count(),
                      driver_and_gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,uuid',
                          '--format=csv,noheader'],text=True,timeout=10).strip())
    return result


class PortableSession:
    def __init__(self, root, rows, identity, *, device='cpu', seed=1201):
        self.root=Path(root);self.rows=deepcopy(rows)
        if not rows or any(x['split']!='train' for x in rows):raise ValueError('TRAIN rows only')
        if len({x['arrays'] for x in rows})!=len(rows):raise ValueError('Duplicate training row')
        self.device=torch.device(device)
        if str(self.device) not in ('cpu','cuda:0'):raise ValueError('CPU or one explicit cuda:0 device only')
        if self.device.type=='cuda':
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Set CUBLAS_WORKSPACE_CONFIG before launching Python')
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactly one visible CUDA GPU required')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if self.device.type=='cuda':torch.cuda.manual_seed_all(seed)
        self.model=RepairNCA().float().to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
                                         weight_decay=0,amsgrad=False,foreach=False,fused=False)
        self.firing=torch.Generator(device=self.device).manual_seed(seed+1)
        self.sampler=TrainingOrder(len(rows),seed+2)
        self.identity={'version':VERSION,'experiment':deepcopy(identity),'runtime':runtime(self.device),'seed':seed,
            'ordered_training_rows':[{'arrays':x['arrays'],'sha256':x['arrays_sha256']} for x in rows]}
        # Validate bytes even for rows not sampled during this short pilot.
        for row in self.rows:load_example(self.root,row)
        self.completed=0;self.trace=[]

    def tensors(self,index):
        inputs,target=load_example(self.root,self.rows[index])
        occupancy=torch.as_tensor(inputs['occupancy'],device=self.device,dtype=torch.float32)[None,None]
        context=torch.as_tensor(inputs['context'],device=self.device,dtype=torch.float32)[None]
        label=torch.as_tensor(target,device=self.device,dtype=torch.float32)[None,None]
        if context.shape!=(1,7,*occupancy.shape[2:]) or label.shape!=occupancy.shape:raise ValueError('Example shapes differ')
        for t in (occupancy,context,label):
            if not torch.isfinite(t).all():raise ValueError('Nonfinite example')
        for t in (occupancy,label,context[:,:6]):
            if not ((t==0)|(t==1)).all():raise ValueError('Binary geometry required')
        allowed=(context[:,:1]>0)&(context[:,1:2]>0)
        if (label.bool()&~allowed).any() or (occupancy.bool()&~allowed).any():raise ValueError('Geometry outside domain')
        return occupancy,perceive(context).detach(),allowed,label

    def synchronize(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    def step(self):
        index=self.sampler.next();occupancy,features,allowed,target=self.tensors(index)
        self.optimizer.zero_grad(set_to_none=True)
        state=self.model.rollout(occupancy,features,allowed,self.firing,16)
        loss=balanced_loss(state[:,:1],target,allowed)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('Invalid gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step();self.synchronize()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite weights')
        self.completed+=1
        row={'update':self.completed,'row_index':index,'loss':float(loss.detach()),'pre_clip_gradient_norm':float(norm)}
        self.trace.append(row)
        return row,state.detach().cpu().numpy()[0].copy()

    def evaluate(self,index=0):
        occupancy,features,allowed,target=self.tensors(index)
        generator=torch.Generator(device=self.device).manual_seed(2101)
        with torch.no_grad():
            state=self.model.rollout(occupancy,features,allowed,generator,16)
            probability=torch.sigmoid(state[:,:1])*allowed
        self.synchronize()
        return {'state':state.cpu().numpy()[0].copy(),'probability':probability.cpu().numpy()[0,0].copy()}

    def payload(self):
        self.synchronize()
        return {'version':VERSION,'identity':deepcopy(self.identity),'completed':self.completed,
                'model':cpu_tree(self.model.state_dict()),'optimizer':cpu_tree(self.optimizer.state_dict()),
                'optimizer_class':'torch.optim.adam.Adam','parameter_names':[n for n,_ in self.model.named_parameters()],
                'module_training':self.model.training,'trace':deepcopy(self.trace),'sampler':self.sampler.state(),
                'rng':rng_state(self.firing),'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else []}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        receipt=path.with_suffix('.json')
        if path.exists() or receipt.exists():raise FileExistsError(path)
        temp=path.with_name('.'+uuid.uuid4().hex+'.partial.pt')
        with temp.open('xb') as f:torch.save(self.payload(),f);f.flush();os.fsync(f.fileno())
        os.link(temp,path);temp.unlink()
        write_once(receipt,{'version':VERSION,'sha256':digest(path),'bytes':path.stat().st_size,
                            'completed':self.completed,'identity_sha256':metadata_hash(self.identity)})

    def restore(self,path):
        p=read_portable(path,self.identity)
        if p['parameter_names']!=[n for n,_ in self.model.named_parameters()] or p['optimizer_class']!='torch.optim.adam.Adam':
            raise ValueError('Model/optimizer identity differs')
        self.model.load_state_dict(p['model'],strict=True);self.optimizer.load_state_dict(p['optimizer'])
        self.model.train(p['module_training']);self.completed=p['completed'];self.trace=p['trace']
        self.sampler.restore(p['sampler']);restore_rng(p['rng'],self.firing)
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(p['cuda_rng'])


def read_portable(path,identity):
    path=Path(path);receipt=read_json(path.with_suffix('.json'))
    if receipt['version']!=VERSION or receipt['identity_sha256']!=metadata_hash(identity):raise ValueError('Runtime/source/data identity differs')
    if digest(path)!=receipt['sha256'] or path.stat().st_size!=receipt['bytes']:raise ValueError('Checkpoint bytes corrupt')
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p['version']!=VERSION or p['identity']!=identity or p['completed']!=receipt['completed']:raise ValueError('Checkpoint metadata differs')
    if p['completed']!=len(p['trace']) or p['completed']!=p['sampler']['consumed']:raise ValueError('Checkpoint cursor differs')
    if any(x['update']!=i+1 for i,x in enumerate(p['trace'])):raise ValueError('Trace order differs')
    if len(p['cuda_rng'])!=(1 if identity['runtime']['device']=='cuda:0' else 0):raise ValueError('GPU RNG topology differs')
    return p
