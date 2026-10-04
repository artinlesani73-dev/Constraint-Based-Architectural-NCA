from pathlib import Path
from collections import deque
import sys,json,hashlib,shutil,zipfile
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G4-Final-Review-2026-10-04';PREP=BASE/'G1-Preparation-2026-10-03'
sys.path.insert(0,str(OUT/'source'))
from nca.contract import entrance_masks
from nca.massing_cases import target_context
from nca.generation_data import seed_inputs
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
r=json.loads((OUT/'result.json').read_text());old=json.loads((BASE/'G3-Final-Review-2026-10-03/result.json').read_text());imports=json.loads((OUT/'imports.json').read_text())
split=json.loads((PREP/'split-manifest.json').read_text());scenes={r['id']:r['scene'] for r in split['entries']};config=json.loads((PREP/'environment.json').read_text())['config']
dataset=json.loads((PREP/'dataset.json').read_text());rows={row['id']:row for row in dataset['rows'] if row['split']=='development'}

def distance_to_region(field,domain,region):
    # Minimum number of6-neighbour legal voxel moves from current mass to interface.
    dist=np.full(field.shape,-1,np.int16);dist[field]=0;queue=deque(map(tuple,np.argwhere(field)))
    while queue:
        p=queue.popleft()
        if region[p]:return int(dist[p])
        for axis in range(3):
            for sign in [-1,1]:
                n=list(p);n[axis]+=sign;n=tuple(n)
                if all(0<=n[i]<field.shape[i] for i in range(3)) and domain[n] and dist[n]<0:dist[n]=dist[p]+1;queue.append(n)
    return None
diagnostics=[]
for rec in r['observations']:
    if rec['steps']!=64:continue
    case=rec['case'];scene=scenes[case.rsplit('-v',1)[0]];_,domain,_=target_context(scene,config);endpoints=entrance_masks(scene)
    with np.load(OUT/f'observations/{case}-64.npz',allow_pickle=False) as a:field=a['field'].copy();counts=a['admission_counts'].copy()
    with np.load(OUT/f'observations/{case}-128.npz',allow_pickle=False) as a:field128=a['field'].copy()
    with np.load(PREP/rows[case]['arrays'].replace('\\','/'),allow_pickle=False) as a:seed=seed_inputs(a['context'])['occupancy'].astype(bool)
    assert field[seed].all() and np.array_equal(field,field128)
    diagnostic=dict(case=case,direct_interface_occupied_cells={k:int((field&mask).sum()) for k,mask in endpoints.items()},legal_voxel_moves_to_interface={k:distance_to_region(field,domain,mask) for k,mask in endpoints.items()},seed_retained=True,field_identical64_128=True,ceiling=rec['budget']['ceiling'],occupied=int(field.sum()),first_ceiling_step=rec['first_ceiling_step'],last_growth_step=int(np.flatnonzero(counts[:,6]>0)[-1]+1),unused_capacity=rec['unused_capacity'],third_bulk_fractions=[x['fraction'] for x in rec['score']['thirds']],failed_families=[k for k,v in rec['score']['family_pass'].items() if not v])
    diagnostics.append(diagnostic)
save('allocation-diagnostics.json',dict(method='Read-only analysis of frozen saved64/128 outputs; no additional inference,training or selection.',interface_metric_caveat='Existing raw_interface_hits report reachability from lexically first endpoint E_east; if E_east is untouched,both reachability flags are false. Direct physical contact is reported separately here; no metric was changed.',distance_caveat='Distance is legal voxel graph distance,not feasible whole-cube additions or a structural/access guarantee.',cases=diagnostics))
comparison=dict(g3=old['summary'],g4=r['summary'],g3_gates=old['gates'],g4_gates=r['gates'],g4_accepted=r['accepted'],same_nine_reused_development_cases=True,final_checkpoint_only=256,firing_seed=2101,device='cpu',dtype='float32',horizons=[64,128],caution='Integrated architectural change,not a single-factor ablation. Full-cube support,budget cap and saturation stability are imposed by the transition. No reserved generalization evidence.')
save('comparison-with-g3.json',comparison)
aggregate=dict(all_final_fields_equal_64_128=True,all_direct_west_contacts=all(x['direct_interface_occupied_cells']['E_west']>0 for x in diagnostics),east_contacts=sum(x['direct_interface_occupied_cells']['E_east']>0 for x in diagnostics),capacity_reached=sum(x['unused_capacity']==0 for x in diagnostics),ceiling_step_range=[min(x['first_ceiling_step'] for x in diagnostics if x['first_ceiling_step'] is not None),max(x['first_ceiling_step'] for x in diagnostics if x['first_ceiling_step'] is not None)],last_growth_step_range=[min(x['last_growth_step'] for x in diagnostics),max(x['last_growth_step'] for x in diagnostics)],bulk_fraction_range=[min(x['score']['bulk_fraction'] for x in r['observations']),max(x['score']['bulk_fraction'] for x in r['observations'])],budget_rejected_block_events64=sum(x['budget_rejected_events'] for x in r['observations'] if x['steps']==64),median_volume_error_percentage_points=100*r['summary']['64']['median_absolute_fraction_error'],max_volume_error_percentage_points=100*r['summary']['64']['max_absolute_fraction_error'])
save('diagnostic-summary.json',aggregate)
table='\n'.join('| '+x['case'].replace('g1-offset_interfaces-','')+' | '+str(x['occupied'])+' | '+str(x['first_ceiling_step'])+' | '+str(x['legal_voxel_moves_to_interface']['E_east'])+' | '+(', '.join(x['failed_families']) or 'PASS')+' |' for x in diagnostics)
report=f'''# G4 final review — 2026-10-04

G4 completed successfully, but does not meet the frozen pilot acceptance gates.
Two of nine reused development requests pass all nine families at both64 and128
steps. The next research issue is spatial allocation before capacity is consumed.
MG7 remains the live model. No replacement or new paid run was performed.

## Verified run

User-supplied run20261004T065608Z_176639ac1bc5 completed256updates on Tesla T4.
All1035evidence payload hashes,unique archive membership and exact G4-v2 package
identity verified. Controlled time {imports['result']['wall_seconds']:.3f}s;
worker {imports['result']['worker']['wall_seconds']:.3f}s;
peak reserved GPU memory {imports['result']['worker']['peak_reserved']/1048576:.0f}MiB.
The12reference/device cases and differentiable-union backward check passed.
Full-payload and state recovery matched exactly at updates2 and3.

All256saved start fields and hashes match the frozen seed/cube-stage schedule:
128seed starts,128cube teacher stages. Sampler order,all16384step-count accounts,
final optimizer step counts and each retained output's legality,connectivity,
seed retention and complete-cube support verified. Count verification is not
a replay of every training rollout. Full original ZIP and receipt are preserved.
No assistant-launched paid job or automatic retry occurred during this review.

## Frozen comparison

Same nine development requests,final checkpoint256only,CPUfloat32,firing2101.
No postprocessing,threshold search,checkpoint selection or reserved evaluation.

| Check | G3 at64 | G4 at64 | G4 at128 |
|---|---:|---:|---:|
| All nine families pass |1/9|2/9|2/9|
| Access |5/9|2/9|2/9|
| Coverage |7/9|6/9|6/9|
| Thickness |1/9|9/9|9/9|
| Each of facade,ground,legality,sparsity,spill,support |9/9|9/9|9/9|
| Median volume-fraction error,percentage points |4.452|{aggregate['median_volume_error_percentage_points']:.3f}|{aggregate['median_volume_error_percentage_points']:.3f}|
| Maximum volume-fraction error,percentage points |8.255|{aggregate['max_volume_error_percentage_points']:.3f}|{aggregate['max_volume_error_percentage_points']:.3f}|
| Median teacher IoU |0.5354|{r['summary']['64']['median_teacher_iou']:.4f}|{r['summary']['128']['median_teacher_iou']:.4f}|

All nine G4fields are identical between64 and128steps; G3 failed the5%stability
gate in all nine. G4passes3of5gates:median/max volume error and stability. It
fails all-nine validity at both horizons. Both passing cases request32%volume
atYoffset variants2and4; the other32%case still fails access.

This is a partial improvement with regressions in access,coverage and teacher
overlap. Complete-cube growth enforces100%bulk support here. The hard cap tightly
bounds mass and explains saturation stability; these are not proof that the
network learned thickness or a self-stabilizing dynamical rule independently.
The integrated change also alters training stages,proposal semantics,firing RNG
and the volume surrogate,so its effects cannot be attributed to one component.

## Why the remaining cases fail

Every output physically touches the west interface and retains the seed. Only
two reach the east interface. All nine exhaust capacity:the first exact cap
hit occurs between steps{aggregate['ceiling_step_range'][0]}and{aggregate['ceiling_step_range'][1]}.
Because additions are irreversible,remaining proposals cannot move already
allocated volume toward the missing interface. Doubling the rollout does not
repair this. All16%requests also underfill the farXthird under the unchanged
coverage metric. Saved geometry supports this diagnosis; it does not identify
which loss or input feature is solely responsible.

| Case | Occupied voxels | First cap step | Legal voxel moves to east | Failure |
|---|---:|---:|---:|---|
{table}

Existing evaluator interface-hit flags measure reachability from alphabetically
first endpoint E_east. If east is untouched,both reported flags are false,even
though west contact exists. allocation-diagnostics.json adds direct contact
counts to avoid misreading that result; the frozen score itself is unchanged.
Graph distances above are diagnostic voxel moves,not whole-cube repair costs.
The all-offer pre-admission candidate is not an unguarded rollout; its seed-step
form can contain multiple cubes. Do not claim it isolates the cap's causal effect.

## Next bounded development step

Retain G4's cube representation and current budget for now. Before another paid
run,design a teacher-independent cue that lets proposals account for the distant
interface early,using the existing access family. A candidate is a legal
cube-origin distance field to the opposite interface,derived from context only.
Audit representability and finite values on TRAIN geometries and check whether
the cue distinguishes useful advance from lateral expansion. Do not encode a
teacher route,hardcode held-out cases,or introduce a new constraint category.

This is a proposed direction,not an implemented or frozen G5protocol. Freeze
one focused change and its effective configuration before training. Keep the
same acceptance thresholds and disclose repeated development-set reuse. Do not
increase grid size,run duration,or add a second seed merely to search for a pass.
Reserved cases remain available for later generalization review.

All evidence,decisions and resumption instructions are local in this folder.
Repository synchronization remains pending:the project checkout's older RESUME
is stale. This archive is on the same disk,not an off-device backup. No Drive,
push,publication or live-model promotion was performed.
'''
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('RESUME.json',dict(status='G4 complete;2of9valid at64/128;not accepted;volumetric thickness and budget fixed,spatial allocation remains',run='20261004T065608Z_176639ac1bc5',previous=str(BASE/'G4-Block-Training-2026-10-03-v2/RESUME.json'),review=str(OUT/'REVIEW.md'),result=str(OUT/'result.json'),next='Design one teacher-independent destination-guidance change on TRAIN contexts for early whole-cube growth. Candidate:legal cube-origin distance to opposite interface. Not yet implemented/frozen. Preserve G4 cap,9families,metrics and128stepstability; avoid further tuning on the nine reused development labels. Prepare tested/versioned next package only after local feasibility checks. Ask separate explicit compute allowance before a new paid run.',paid_run_authorized=False,reserved_targets_opened=False,repository_sync_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save('project-record.json',dict(event='G4 returned training evidence reviewed',date='2026-10-04',run='20261004T065608Z_176639ac1bc5',verified=True,accepted=False,changes='No model/training changes in this review. Added G4seven-column review and saved-output allocation diagnostics.',results=r['summary'],gates=r['gates'],findings=aggregate,decision='Preserve cube/budget improvements; do not promote. Next investigate destination-aware spatial allocation on TRAIN before more paid compute.',limitations=['Single trained seed','Reused development set','Thickness/budget/stability largely enforced','No reserved generalization','No exact replay of all256GPUupdates','Repository synchronization/off-device backup pending']))
shutil.copyfile(__file__,OUT/'summarize-review.py');shutil.copyfile(Path(__file__).with_name('make_review_g4.py'),OUT/'build-review-script.py')
save('source-fingerprints.json',{p.relative_to(OUT/'source').as_posix():sha(p.read_bytes()) for p in sorted((OUT/'source').rglob('*.py'))})
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',{'files':files});files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
archive.with_suffix('.receipt.json').write_text(json.dumps(dict(archive=str(archive),payloads=len(files),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),verified=True,off_device_backup=False),indent=2))
print(json.dumps(dict(aggregate=aggregate,cases=diagnostics,archive=str(archive)),indent=2))
