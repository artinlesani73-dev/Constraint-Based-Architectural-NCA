from pathlib import Path
from collections import deque
import sys,json,hashlib,shutil,zipfile
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G5-Final-Review-2026-10-04';OLD=BASE/'G4-Final-Review-2026-10-04';PREP=BASE/'G1-Preparation-2026-10-03'
sys.path.insert(0,str(OUT/'source'))
from nca.contract import entrance_masks
from nca.massing_cases import target_context
from nca.generation_data import seed_inputs
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
r=json.loads((OUT/'result.json').read_text());old=json.loads((OLD/'result.json').read_text());imports=json.loads((OUT/'imports.json').read_text());pair=json.loads((OUT/'pairing-with-g4.json').read_text())
split=json.loads((PREP/'split-manifest.json').read_text());scenes={x['id']:x['scene'] for x in split['entries']};config=json.loads((PREP/'environment.json').read_text())['config']
rows={x['id']:x for x in json.loads((PREP/'dataset.json').read_text())['rows'] if x['split']=='development'}

def graph_distance(field,domain,region):
    distance=np.full(field.shape,-1,np.int16);distance[field]=0;queue=deque(map(tuple,np.argwhere(field)))
    while queue:
        p=queue.popleft()
        if region[p]:return int(distance[p])
        for axis in range(3):
            for sign in [-1,1]:
                n=list(p);n[axis]+=sign;n=tuple(n)
                if all(0<=n[i]<field.shape[i] for i in range(3)) and domain[n] and distance[n]<0:distance[n]=distance[p]+1;queue.append(n)
    return None
cases=[]
for rec in r['observations']:
    if rec['steps']!=64:continue
    case=rec['case'];scene=scenes[case.rsplit('-v',1)[0]];_,domain,_=target_context(scene,config);endpoints=entrance_masks(scene)
    with np.load(OUT/f'observations/{case}-64.npz',allow_pickle=False) as a:field=a['field'].copy();counts=a['admission_counts'].copy()
    with np.load(OUT/f'observations/{case}-128.npz',allow_pickle=False) as a:field128=a['field'].copy()
    with np.load(OLD/f'observations/{case}-64.npz',allow_pickle=False) as a:previous=a['field'].copy()
    with np.load(PREP/rows[case]['arrays'].replace('\\','/'),allow_pickle=False) as a:seed=seed_inputs(a['context'])['occupancy'].astype(bool)
    assert field[seed].all();old_rec=next(x for x in old['observations'] if x['case']==case and x['steps']==64)
    events=np.flatnonzero(counts[:,6]>0)
    cases.append(dict(case=case,request=rec['request'],occupied=int(field.sum()),ceiling=rec['budget']['ceiling'],first_ceiling_step=rec['first_ceiling_step'],last_growth_step=int(events[-1]+1) if len(events) else None,unused_capacity=rec['unused_capacity'],same_field64_128=bool(np.array_equal(field,field128)),direct_contact_voxels={k:int((field&mask).sum()) for k,mask in endpoints.items()},legal_voxel_moves_to_interface={k:graph_distance(field,domain,mask) for k,mask in endpoints.items()},third_bulk_fractions=[x['fraction'] for x in rec['score']['thirds']],failed_families=[k for k,v in rec['score']['family_pass'].items() if not v],g4_failed_families=[k for k,v in old_rec['score']['family_pass'].items() if not v],g4_iou=old_rec['teacher_iou'],g5_iou=rec['teacher_iou'],changed_voxels_from_g4=int((field^previous).sum()),seed_retained=True))
save('allocation-diagnostics.json',dict(method='Read-only diagnostics of frozen saved outputs; no additional inference or training.',interface_caveat='Frozen interface-hit flags indicate reachability from E_east. These supplemental direct-contact counts distinguish physical west contact from east-rooted reachability.',distance_caveat='Legal6-neighbour voxel distance,not feasible whole-cube admission cost.',cases=cases))
curve={}
for label,folder in [('G4',OLD),('G5',OUT)]:
    traces=[json.loads((folder/f'import/worker/update-{i:04d}.json').read_text()) for i in range(1,257)]
    curve[label]={}
    for name,selection in [('first32',traces[:32]),('last32',traces[-32:])]:
        curve[label][name]={}
        for start in ['seed','cube_teacher_stage']:
            selected=[x for x in selection if x['start']['kind']==start]
            curve[label][name][start]={k:float(np.mean([x[k] for x in selected])) for k in ['loss','frontier_loss','volume_loss','band_loss','pre_clip_gradient_norm']}
save('training-trace-summary.json',dict(windows=curve,caution='On-policy rollout losses,not a shared-state quality test. Same rows/starts but generated trajectories differ. No checkpoint selection based on these windows.'))
comparison=dict(g4=old['summary'],g5=r['summary'],g4_gates=old['gates'],g5_gates=r['gates'],g5_accepted=r['accepted'],same_nine_reused_development_cases=True,final_checkpoint=256,horizons=[64,128],firing_seed=2101,evaluation_device='cpu',dtype='float32',pairing=pair)
save('comparison-with-g4.json',comparison)
summary=r['summary']['64'];stats=dict(all_fields_identical64_128=all(x['same_field64_128'] for x in cases),capacity_reached=sum(x['unused_capacity']==0 for x in cases),first_cap_step_range=[min(x['first_ceiling_step'] for x in cases if x['first_ceiling_step'] is not None),max(x['first_ceiling_step'] for x in cases if x['first_ceiling_step'] is not None)],direct_west_contacts=sum(x['direct_contact_voxels']['E_west']>0 for x in cases),direct_east_contacts=sum(x['direct_contact_voxels']['E_east']>0 for x in cases),median_fraction_error_percentage_points=100*summary['median_absolute_fraction_error'],maximum_fraction_error_percentage_points=100*summary['max_absolute_fraction_error'],median_iou_change=summary['median_teacher_iou']-old['summary']['64']['median_teacher_iou'],bulk_fraction_range=[min(x['score']['bulk_fraction'] for x in r['observations']),max(x['score']['bulk_fraction'] for x in r['observations'])])
save('diagnostic-summary.json',stats)
# Inspect the actual packaged objective, without changing it or tuning to held-out cases.
source=OUT/'source/nca/guided_generation.py';text=source.read_text();start=text.index('def block_loss(');end=text.index('\nclass GuidedNCA',start)
save('objective-audit.json',dict(source=str(source),source_sha256=sha(source.read_bytes()),function='block_loss',active_terms=['Eligible fired origin positive/negative teacher BCE','After seed phase:0.25 local3cube mean-volume error','After seed phase:1.0 global volume-band error'],explicit_interface_connection_loss=False,explicit_third_coverage_loss=False,hard_births_detached=True,interpretation='Access/coverage are currently supervised indirectly through teacher geometry. Their absence as explicit loss terms is a code finding; it does not prove adding such a term alone will solve the task.',implementation=text[start:end]))
table='\n'.join('| '+x['case'].replace('g1-offset_interfaces-','')+' | '+str(x['occupied'])+' | '+str(x['first_ceiling_step'])+' | '+str(x['legal_voxel_moves_to_interface']['E_east'])+' | '+(', '.join(x['failed_families']) or 'PASS')+' |' for x in cases)
report=f'''# G5 final review — 2026-10-04

G5 completed correctly but did not improve the frozen pilot result. It passes
all nine families in **1 of 9** reused development cases, versus **2 of 9** for
G4. Keep G4 as the stronger cube-growth research reference and MG7 as the live
model. G5 is preserved as a rejected candidate, not discarded or promoted.

## Verified execution

Run `20261004T074101Z_d3a90a77c991` completed all 256 updates on Tesla T4 in
{imports['result']['wall_seconds']:.3f} controlled seconds; worker time was
{imports['result']['worker']['wall_seconds']:.3f} seconds. Peak reserved memory was
{imports['result']['worker']['peak_reserved']/1048576:.0f} MiB. All 1,035 payload hashes,
unique archive membership and the exact G5 package identity were verified.
The 12 admission reference checks, union backward check and cue device-copy check
passed. Recovery reproduced the full payload and state exactly at updates 2 and 3.

All 256 saved starts and their hashes match the frozen schedule: 128 seed starts
and 128 cube-stage starts. Sampler order, 16,384 step accounts, final optimizer
counts and each retained field's legality, connectivity, seed retention and
cube support were checked. This is not a replay of every training rollout.
The original full ZIP, receipt and every checkpoint remain preserved.

## Matched comparison

G4 and G5 used identical dataset hashes, all 256 row/start choices, core initial
parameters, final firing-generator state and recorded software versions. G5
adds 128 first-layer parameters for its two context channels, initialized to
zero. Those parameters acquired nonzero weights. This confirms participation
in training, not useful guidance. One seed cannot establish a general causal
effect; the input-shape change may also affect numerical kernels.

Evaluation used final checkpoint 256 only, CPU float32, firing seed 2101 and
the same nine development requests at 64 and 128 steps. There was no clipping,
threshold search, checkpoint selection, extra seed or reserved evaluation.

| Measure | G4 at 64 steps | G5 at 64 steps | G5 at 128 steps |
|---|---:|---:|---:|
| All nine families pass | 2/9 | 1/9 | 1/9 |
| Access | 2/9 | 1/9 | 1/9 |
| Coverage | 6/9 | 3/9 | 3/9 |
| Thickness | 9/9 | 9/9 | 9/9 |
| Each of the other six families | 9/9 | 9/9 | 9/9 |
| Median volume error, percentage points | 0.152 | {stats['median_fraction_error_percentage_points']:.3f} | {stats['median_fraction_error_percentage_points']:.3f} |
| Maximum volume error, percentage points | 0.166 | {stats['maximum_fraction_error_percentage_points']:.3f} | {stats['maximum_fraction_error_percentage_points']:.3f} |
| Median teacher IoU | {old['summary']['64']['median_teacher_iou']:.4f} | {summary['median_teacher_iou']:.4f} | {r['summary']['128']['median_teacher_iou']:.4f} |

Both candidates pass the volume-error and stability gates and fail all-nine
validity at both horizons. All G5 fields are identical between 64 and 128 steps.
The one valid case is `g1-offset_interfaces-y4-v32`; G5 loses G4's passing
`g1-offset_interfaces-y2-v32` case. Every 16% and 24% request now fails coverage.
Improved teacher overlap therefore did not translate into better task success.

Thickness is imposed by complete-cube additions. Budget control and stability
at full capacity are also enforced by the transition; they are not evidence
of independently learned self-stabilization or general architectural quality.

## Saved-output diagnosis

All nine outputs retain physical west-interface contact. Only one reaches the
east interface. All nine exhaust capacity by steps
{stats['first_cap_step_range'][0]}–{stats['first_cap_step_range'][1]}. Further steps cannot
redistribute already occupied volume because this transition only adds cubes.

| Case | Occupied voxels | First cap step | Legal voxel moves to east | Failed families |
|---|---:|---:|---:|---|
{table}

Distances are supplemental legal-voxel graph diagnostics, not whole-cube repair
costs. The frozen evaluator starts reachability at E_east; missing east contact
makes both reachability flags false even when west is physically touched.
Direct contact is recorded separately without altering the frozen score.
The pre-admission all-offer candidate remains a diagnostic, not an unguarded
rollout or a cap-removal experiment.

## What this means for the next step

The distance-input intervention alone is insufficient in this run. Avoid another
paid feature variant or a longer run before examining the training signal.
Code inspection confirms that the active objective contains teacher-origin BCE,
local mean-volume error and global volume-band error. It has no explicit loss
for interface connection or coverage of the fixed site thirds: those outcomes
are encouraged only indirectly through the teacher shapes. This mismatch is
an actionable hypothesis, not proof of the sole failure cause.

Next, audit early growth decisions and objective gradients on TRAIN contexts
using G4 as the reference. Check whether useful progress toward the opposite
interface competes with rapid lateral filling under the current loss. Design
one task-aligned access or allocation objective only after this local audit,
with explicit gradient and failure checks. Preserve the nine families, current
cube representation and budget, and do not encode a teacher route at inference.
No G6 implementation, training configuration or paid allowance is frozen here.

The development set has been reused and must be described that way. Keep
reserved labels unopened for later generalization review; do not tune new
coefficients by repeatedly re-scoring these nine cases. No deployment follows
from a future development-only pass without its planned generalization review.

## Preservation

All observations, source fingerprints, paired checks, decisions and continuation
instructions are saved locally. Earlier G4/G5 preparation and experiment files
are unchanged. Repository synchronization is pending; the checkout's older
RESUME is stale, so use this folder's RESUME.json. The verified archive is on
the same disk, not an off-device backup. No Drive operation, additional paid
job, retry, push, publication or live-model replacement was performed.
'''
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('RESUME.json',dict(status='G5 completed;1of9valid vs G4two;candidate rejected',run='20261004T074101Z_d3a90a77c991',previous=str(BASE/'G5-Destination-Guidance-2026-10-04/RESUME.json'),review=str(OUT/'REVIEW.md'),result=str(OUT/'result.json'),next='Before another paid feature trial,audit early G4cube-growth decisions and objective gradients on TRAIN contexts. Active objective has no explicit interface-connection or site-third-coverage term. Determine whether current losses favor lateral fill over useful destination progress;then propose one task-aligned objective with tested gradients. No new package/protocol frozen. Keep ninefamilies,cube representation,budget and reserved holdout;do not tune on reused development labels.',research_reference='G4 for cube-growth studies;neither G4 nor G5 accepted',paid_run_authorized=False,reserved_targets_opened=False,repository_sync_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save('project-record.json',dict(date='2026-10-04',event='G5 final256 reviewed against G4',accepted=False,results=r['summary'],gates=r['gates'],pairing=pair,diagnosis=stats,changes='Added read-only G5evaluation,pairing,allocation diagnostics and objective audit;no training/model changes.',decision='Reject G5promotion;retain G4research reference. Audit task-objective alignment on TRAIN before another GPU request.',limitations=['One seeded intervention','Reused development set','Cue contribution not isolated from changed network input shape','No full replay of all256GPUupdates','No reserved evaluation','Enforced thickness/budget/stability','Repository sync and off-device backup pending']))
shutil.copyfile(__file__,OUT/'summarize-review.py');shutil.copyfile(Path(__file__).with_name('make_review_g5.py'),OUT/'build-review-script.py')
save('source-fingerprints.json',{p.relative_to(OUT/'source').as_posix():sha(p.read_bytes()) for p in sorted((OUT/'source').rglob('*.py'))})
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',{'files':files});files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
record=dict(archive=str(archive),payloads=len(files),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),verified=True,off_device_backup=False)
archive.with_suffix('.receipt.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(stats=stats,archive=record,curve=curve),indent=2))
