"""Opt-in persistent scaffold conditioning. Historical model/rollout untouched."""
import copy
import hashlib
import json
import torch
import torch.nn.functional as F
from deploy.model_utils import UrbanPavilionNCA, Perceive3D, LocalLegalityLoss
from nca.interventions import experimental_rollout

BASELINE = 'original_model_c_v1'
GUIDED = 'persistent_scaffold_v1'

def fingerprint(t):
    return (tuple(t.shape),str(t.dtype),str(t.device),hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest())

class GuideContext:
    """Owned static cache, bound to exact scene, seed, scaffold and config.

    Never stores trainable projections or autograd graphs. Tensor integrity is
    checked on each rollout; callers receive a clone, not mutable cached storage.
    """
    def __init__(self,scene,seed,scaffold,config):
        if not isinstance(scene,str) or not scene:raise ValueError('Scene identity required')
        if seed.ndim!=5 or seed.shape[1]!=config['n_channels'] or seed.dtype!=torch.float32:
            raise ValueError('Expected float32 B,C,D,H,W state')
        if scaffold.shape!=(seed.shape[0],*seed.shape[-3:]) or scaffold.dtype!=seed.dtype or scaffold.device!=seed.device:
            raise ValueError('Scaffold shape/device/dtype mismatch')
        if seed.requires_grad or scaffold.requires_grad:raise ValueError('Context must be static')
        if not torch.isfinite(seed).all() or not torch.isfinite(scaffold).all() or ((scaffold<0)|(scaffold>1)).any():
            raise ValueError('Invalid context values')
        self.scene=scene
        self.config=json.dumps(config,sort_keys=True)
        self.seed_key=fingerprint(seed);self.scaffold_key=fingerprint(scaffold)
        with torch.no_grad():self._features=Perceive3D(1).to(seed.device)(scaffold[:,None]).detach().clone()
        self._feature_key=fingerprint(self._features)

    def features(self,scene,seed,scaffold,config):
        if (scene!=self.scene or json.dumps(config,sort_keys=True)!=self.config or
            fingerprint(seed)!=self.seed_key or fingerprint(scaffold)!=self.scaffold_key or
            fingerprint(self._features)!=self._feature_key):raise ValueError('Stale, mutated or mismatched guide context')
        if seed.requires_grad or scaffold.requires_grad:raise ValueError('Context must be static')
        return self._features.clone()

class GuidedNCA(UrbanPavilionNCA):
    def __init__(self,config):
        super().__init__(copy.deepcopy(config))
        if config['n_channels']!=8 or config['n_grown']!=4 or config['hidden_dim']!=96:
            raise ValueError('This migration is specific to original Model C')
        # torch.zeros consumes no RNG. Keep original backbone state_dict names.
        self.guide_weight=torch.nn.Parameter(torch.zeros(96,4,1,1,1))

    def _step(self,state,*,generator=None):
        raise ValueError('Conditioned model requires guided_rollout with a bound context')

    def load_original(self,weights):
        if 'guide_weight' in weights:raise ValueError('Migration requires an original checkpoint')
        result=self.load_state_dict(weights,strict=False)
        if result.missing_keys!=['guide_weight'] or result.unexpected_keys:
            raise ValueError('Incomplete original checkpoint migration')

def guided_rollout(model,seed,scaffold,arm,steps,generator,context=None,scene=None):
    if not isinstance(model,GuidedNCA):
        if context is not None:raise ValueError('Original model cannot consume guide context')
        return experimental_rollout(model,seed,scaffold,arm,steps,generator)
    if arm!='hard_preclamp' or isinstance(steps,bool) or not isinstance(steps,int) or steps<1 or generator is None:
        raise ValueError('Invalid conditioned rollout')
    if context is None:raise ValueError('Bound guide context required')
    cfg=model.config
    if cfg['ch_structure']!=cfg['n_frozen']:raise ValueError('Invalid material channel')
    features=context.features(scene,seed,scaffold,cfg)
    if model.guide_weight.device!=seed.device or model.guide_weight.dtype!=seed.dtype:
        raise ValueError('Model/context device or dtype mismatch')
    # Project once per rollout, retaining autograd through every recurrent use.
    guide=F.conv3d(features,model.guide_weight)
    state=seed.clone()
    state[:,cfg['ch_structure']]=(state[:,cfg['ch_structure']]+.15*scaffold).clamp(0,1)
    legality=LocalLegalityLoss(cfg).compute_legality_field(seed)
    available=1-seed[:,cfg['ch_existing']]
    for _ in range(steps):
        x=model.update_net[0](model.perceive(state))+guide
        for layer in list(model.update_net.children())[1:]:x=layer(x)
        delta=x
        mask=(torch.rand(seed.shape[0],1,*seed.shape[-3:],device=seed.device,generator=generator)<cfg['fire_rate']).float()
        candidate=state[:,cfg['n_frozen']:]+cfg['update_scale']*delta*mask
        raw=candidate[:,0]
        material=raw.clamp(0,1)*available*legality
        state=torch.cat((state[:,:cfg['n_frozen']],material[:,None],candidate[:,1:].clamp(0,1)),dim=1)
    return {'state':state,'raw_material':raw,'version':GUIDED,'arm':arm,'steps':steps,'beta':20.0}
