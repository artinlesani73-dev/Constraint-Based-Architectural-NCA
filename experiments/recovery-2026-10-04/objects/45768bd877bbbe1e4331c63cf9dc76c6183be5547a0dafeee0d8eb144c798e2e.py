from pathlib import Path
import json,zipfile,hashlib,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OLD=BASE/'G4-Block-Training-2026-10-03-v2';OUT=BASE/'G5-Destination-Guidance-2026-10-04';OUT.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(OLD/'NCA-G4-Block-Package.zip') as z:original={k:z.read(k) for k in z.namelist() if k!='manifest.json'}
payload=dict(original);payload['nca/destination_cue.py']=Path(__file__).with_name('destination_cue.py').read_bytes()
# Preserve G4 module. Derive G5 rollout with a narrowly scoped2channel input extension.
s=payload['nca/block_generation.py'].decode()
s=s.replace('G4: learned cube proposals','G5: destination-conditioned cube proposals').replace("VERSION='block_generation_v1'","VERSION='destination_conditioned_block_generation_v1'")
s=s.replace('from torch.nn import functional as F','from torch.nn import functional as F\nfrom torch import nn\nfrom nca.destination_cue import destination_cue')
s=s.replace('class BlockNCA(BudgetNCA):','''class GuidedNCA(BudgetNCA):
    def __init__(self):
        super().__init__()
        old=self.first;self.first=nn.Conv3d(63,64,1)
        with torch.no_grad():
            self.first.weight[:,:61].copy_(old.weight);self.first.weight[:,61:].zero_();self.first.bias.copy_(old.bias)

    def import_base(self,base):
        # Parent copies identically initialized60core channels; all3added inputs zero.
        super().import_base(base)
''')
needle='        valid=full_origins(legal)'
assert needle in s;s=s.replace(needle,needle+'''
        interfaces=static_features[0,5].detach().cpu().numpy()
        if not np.isin(interfaces,[0,1]).all():raise ValueError('Binary interface channel required')
        cue,_=destination_cue(legal,interfaces.astype(bool))
        cue_tensor=torch.tensor(cue,device=occupancy.device)[None]
''')
needle='static_features,remaining.expand_as(occupancy)),1)';assert needle in s
s=s.replace(needle,'static_features,remaining.expand_as(occupancy),cue_tensor),1)')
payload['nca/guided_generation.py']=s.encode()
s=payload['nca/generation_training.py'].decode().replace('G4 cube-proposal','G5 destination-conditioned cube-proposal').replace('from nca.block_generation import BlockNCA,COUNT_COLUMNS','from nca.guided_generation import GuidedNCA,COUNT_COLUMNS').replace("VERSION='seed_generation_training_v4_blocks'","VERSION='seed_generation_training_v5_destination'").replace('model=BlockNCA()','model=GuidedNCA()').replace('architecture="61-64-8"','architecture="63-64-8",destination_cue="opposite_x_interface_cube_distance_v1",cue_channels=["normalized_cube_graph_distance","reachable_origin"],cue_normalization="maximum finite context distance,minimum1",cue_weights_initialization="zero"')
payload['nca/generation_training.py']=s.encode()
s=payload['scripts/colab_generation.py'].decode().replace('G4','G5').replace('from nca.block_generation import device_probe','from nca.destination_cue import device_probe');payload['scripts/colab_generation.py']=s.encode()
protocol='''# G5 destination-guidance pilot

One focused hypothesis: G4 exhausts its budget before reaching the opposite
interface because useful destination information is not readily available to
early local proposals. Add two static context-only input channels:normalized
shortest legal cube-origin distance to the farXinterface,and a reachable flag.
This tests feature availability; it does not establish the cause of G4failure.
No new constraint category. Same nine families and overall building-volume brief.

Destination is maximumX plane of legal interface-union voxels,opposite the
existing minimumX seed convention. Legal origins fit an entire3cube inside the
allowed domain. Goal origins cover at least one destination voxel. Multi-source
6-neighbour BFS on origins computes shortest graph distances; normalize by the
maximum finite distance in that context,minimum1. Invalid/unreachable origins
have distance0,reachable0. Valid goal has distance0,reachable1. Pad one zero voxel
around origin arrays to align their values with cube centres. No derivatives of
these two channels are added. Context-only cache has32entries;cached values are
immutable and independent of training/RNG. No teacher,route,target mass,current
occupancy or development labels enter feature construction. This is scoped to
the existing oppositeXinterface convention,not arbitrary-interface support.

Fresh seed1201,63->64->8 model. Copy the same freshly initialized60core inputs as
G4; budget and two cue weights start at zero. No trained G4weights imported.
Existing27TRAIN dataset bytes,teacher cube stages,sampler,start schedule,losses,
origin firing,hard threshold,overlap admission,volume band and64step training
remain unchanged. All previously declared hybrid CPU/GPU limitations remain.
Global context preprocessing is additional nonlocal information,not a pure
local NCA. Cube support,budget obedience and saturation stability remain enforced.

One Tesla T4 job:256updates,batch1,64steps,float32,Adam0.001,gradient clip1,
maximum600controlledseconds. Includes12admission reference probes,union backward,
cue device-copy check,exact full-payload recovery at updates2and3,every completed
update checkpoint/start/state/trace,and a final seed-only diagnostic. Setup,
upload/export/download/idle extra. ExpectedPython3.13.15,Torch2.11.0+cu130,
NumPy2.1.3,CUDA13.0,cuDNN92700. Stop on runtime mismatch,probe/recovery failure,
nonfinite values or reserved GPU memory>80%. No retry or automatic extension.
Exact recovery is completed-update and same-runtime only. Export FULL ZIP+receipt.

Frozen review unchanged:final256checkpoint,CPUfloat32,firing2101,nine reused
G1development requests,single scene-defined seed,64primary and128stability steps.
Require9/9all-nine at both horizons,median absolute requested-fraction error
<=0.02,max<=0.04,and each mass change<=5%. Report all cases/families,IoUdiagnostic,
cap steps and seven-column block accounting;no postprocessing,rerolls,threshold
or checkpoint search. Existing G4run is comparison;no extra control GPU job.
Development reuse is explicit;reserved labels remain unopened. Passing these
pilot gates does not authorize deployment. MG7 remains live.

Keep APPROVED_G5_JOB=False until this exact single job has an explicit compute
allowance. No Drive operation,paid retry,extra seed,push or publication authorized.
'''
payload['PROTOCOL.md']=protocol.encode()
for k,v in payload.items():
    if k.endswith('.py'):compile(v,k,'exec')
assert payload['data.json']==original['data.json']
for row in json.loads(payload['data.json'])['rows']:assert payload[row['arrays']]==original[row['arrays']]
manifest={'version':'g5_destination_package_v1','files':{k:sha(v) for k,v in payload.items()}}
archive=OUT/'NCA-G5-Destination-Package.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for k,v in payload.items():z.writestr(k,v)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(payload)+1
    assert all(sha(z.read(k))==v for k,v in manifest['files'].items());z.extractall(OUT/'package')
receipt=json.loads((OLD/'package-receipt.json').read_text());nb=json.loads((OLD/'NCA-G4-Block.ipynb').read_text())
for c in nb['cells']:
    text=''.join(c['source']).replace(receipt['sha256'],sha(archive.read_bytes())).replace('G4','G5').replace('nca-g4-','nca-g5-')
    if c['cell_type']=='markdown':text=protocol
    else:compile(text,'notebook','exec')
    c['source']=text.splitlines(True)
(OUT/'NCA-G5-Destination.ipynb').write_text(json.dumps(nb,indent=2));(OUT/'PROTOCOL.md').write_text(protocol)
(OUT/'package-receipt.json').write_text(json.dumps(dict(archive=archive.name,sha256=sha(archive.read_bytes()),payloads=len(payload),manifest_sha256=sha((OUT/'package/manifest.json').read_bytes())),indent=2))
(OUT/'change-audit.json').write_text(json.dumps(dict(changed_members=[k for k in payload if k in original and payload[k]!=original[k]],added_members=[k for k in payload if k not in original],dataset_bytes_unchanged=True,parent_manifest_sha256=receipt['manifest_sha256']),indent=2))
shutil.copyfile(__file__,OUT/'build-package.py');print(OUT)
