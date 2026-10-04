from pathlib import Path
import json,hashlib,zipfile,shutil
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G6-Final-Review-2026-10-04';OLD=BASE/'G4-Final-Review-2026-10-04';PREP=BASE/'G1-Preparation-2026-10-03'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
r=json.loads((OUT/'result.json').read_text());old=json.loads((OLD/'result.json').read_text());imports=json.loads((OUT/'imports.json').read_text());pair=json.loads((OUT/'pairing-with-g4.json').read_text())
assert r['accepted'] and all(r['gates'].values())
cases=[]
for rec in r['observations']:
    if rec['steps']!=64:continue
    case=rec['case']
    with np.load(OUT/f'observations/{case}-64.npz',allow_pickle=False) as a:f64=a['field'].copy();counts=a['admission_counts'].copy();caps=a['step_ceilings'].copy()
    with np.load(OUT/f'observations/{case}-128.npz',allow_pickle=False) as a:f128=a['field'].copy()
    assert np.array_equal(f64,f128)
    voxels=np.argwhere(f64);span=voxels.max(0)-voxels.min(0)+1
    cases.append(dict(case=case,request=rec['request'],occupied=int(f64.sum()),quota=rec['quota'],global_ceiling=rec['budget']['ceiling'],first_global_cap_step=rec['first_ceiling_step'],effective_allowance_hit_steps=rec['effective_allowance_hit_steps'],allowance_rejected_events=rec['allowance_rejected_events'],field_identical64_128=True,volume_fraction=rec['score']['volume_fraction'],volume_error_percentage_points=100*rec['absolute_fraction_error'],bulk_fraction=rec['score']['bulk_fraction'],third_bulk_fractions=[x['fraction'] for x in rec['score']['thirds']],facade_contact_fraction=rec['score']['facade_contact_fraction'],teacher_iou=rec['teacher_iou'],bounding_span_zyx_cells=span.tolist(),all_nine_pass=True))
training={}
for label,folder in [('G4',OLD),('G6',OUT)]:
    capped=[];first_hits=[];seed_capped=[];stage_capped=[]
    for i in range(1,257):
        t=json.loads((folder/f'import/worker/update-{i:04d}.json').read_text());cs=np.asarray(t['admission_counts']);C=t['budget'][2]
        flags=(cs[:,0]==C).tolist();capped.extend(flags)
        (seed_capped if t['start']['kind']=='seed' else stage_capped).extend(flags)
        hits=np.flatnonzero(cs[:,0]+cs[:,6]==C)
        if len(hits):first_hits.append(int(hits[0]+1))
    training[label]=dict(total_steps=len(capped),steps_starting_at_global_cap=sum(capped),fraction_at_global_cap=float(np.mean(capped)),seed_fraction_at_global_cap=float(np.mean(seed_capped)),teacher_stage_fraction_at_global_cap=float(np.mean(stage_capped)),updates_reaching_global_cap=len(first_hits),median_first_global_cap_step=float(np.median(first_hits)) if first_hits else None)
save('training-capacity-comparison.json',training)
stats=dict(all_fields_identical64_128=True,global_cap_step_range=[min(x['first_global_cap_step'] for x in cases),max(x['first_global_cap_step'] for x in cases)],quota_range=[min(x['quota'] for x in cases),max(x['quota'] for x in cases)],median_error_percentage_points=100*r['summary']['64']['median_absolute_fraction_error'],max_error_percentage_points=100*r['summary']['64']['max_absolute_fraction_error'],minimum_third_fraction=min(min(x['third_bulk_fractions']) for x in cases),maximum_facade_fraction=max(x['facade_contact_fraction'] for x in cases),bulk_fraction_range=[min(x['bulk_fraction'] for x in cases),max(x['bulk_fraction'] for x in cases)])
save('case-diagnostics.json',dict(cases=cases,summary=stats,interpretation='Frozen final-checkpoint development review;additional read-only diagnostics of saved fields. No new inference.'))
save('comparison-with-g4.json',dict(g4=old['summary'],g6=r['summary'],g4_gates=old['gates'],g6_gates=r['gates'],g6_pilot_gate_passed=True,paired_evidence=pair,training_capacity=training,scope='One seed and nine reused development requests;not reserved generalization or deployment approval.'))
checkpoint=OUT/'import/worker/checkpoint-0256.pt';checkpoint_receipt=json.loads(checkpoint.with_suffix('.json').read_text());assert sha(checkpoint.read_bytes())==checkpoint_receipt['sha256']
candidate=dict(status='qualified_research_candidate_after_development_gate',run='20261004T081739Z_95ddbf523738',checkpoint_update=256,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_receipt['sha256'],checkpoint_bytes=checkpoint.stat().st_size,package_manifest_sha256=imports['identity']['experiment']['manifest_sha256'],model_class='nca.paced_generation.PacedNCA',contract='massing_targets_v1',metric_source_sha256=sha((OUT/'source/nca/massing_targets.py').read_bytes()),configuration_source_sha256=sha((PREP/'environment.json').read_bytes()),firing_seed=2101,threshold=.5,quota_rule='max(9,ceil((C-27)/63));unchanged across64/128',evaluation_device='cpu',evaluation_dtype='float32',horizons=[64,128],development_case_ids=[x['case'] for x in cases],reserved_evaluated=False,live_promoted=False)
save('candidate-freeze.json',candidate)
split=json.loads((PREP/'split-manifest.json').read_text());reserved=[x['id'] for x in split['entries'] if x['split']=='reserved'];assert len(reserved)==4
plan=dict(status='planned_not_executed',candidate=candidate['checkpoint_sha256'],split_manifest_sha256=sha((PREP/'split-manifest.json').read_bytes()),reserved_scene_ids=reserved,requests=[.16,.24,.32],expected_cases=12,horizons=[64,128],firing_seed=2101,device='cpu',dtype='float32',threshold=.5,quota_rule=candidate['quota_rule'],input_rule='Reconstruct the existing seven-channel scene context from saved environment config;scene-defined seed only;no teacher geometry or route as input.',teachers_required=False,gates={'all_nine_at64':12,'all_nine_at128':12,'median_absolute_fraction_error_max':.02,'max_absolute_fraction_error_max':.04,'per_case_relative_mass_change_max':.05},failure_rule='Report every scene/request and retain infeasible/unsupported/failing cases. Do not drop cases,retune,reroll,change quota or select checkpoints after inspection.',remaining_work=['Verify existing context construction matches archived TRAIN fixtures byte-for-byte before reserved generation','Generate and evaluate all12reserved requests once at frozen settings','Inspect saved geometry visually and document physical interpretation','Decide separately whether the candidate is ready for research UI integration'],paid_training_required=False,executed=False)
save('next-generalization-protocol.json',plan)
rows='\n'.join('| '+x['case'].replace('g1-offset_interfaces-','')+' | '+str(x['occupied'])+' | '+str(x['quota'])+' | '+str(x['first_global_cap_step'])+' | '+f"{x['facade_contact_fraction']:.3f}"+' | PASS |' for x in cases)
report=f'''# G6 final review — 2026-10-04

**G6 passes all five frozen development gates.** All nine reused development
requests pass all nine constraint families at both64 and128steps. The fields
are identical at the two horizons. Freeze checkpoint256 as the qualified
research candidate; generalization and visual review remain before deployment.
MG7 stays live.

## Verified run

Run `20261004T081739Z_95ddbf523738` completed256updates on Tesla T4 in
{imports['result']['wall_seconds']:.3f}controlledseconds, with
{imports['result']['worker']['wall_seconds']:.3f}workerseconds and
{imports['result']['worker']['peak_reserved']/1048576:.0f}MiB peak reserved GPU memory.
All1,035evidence payload hashes,unique archive membership and exact G6package
identity verified. The12original admission probes,8paced probes and union
backward check passed. Full-payload/state recovery matched exactly at updates2
and3. All256saved starts,16,384step ceilings and count records were verified.
This does not mean every GPU training rollout was independently replayed.

G4 and G6 share all initial parameters,dataset hashes,row/start schedules and
final firing-generator state. Runtime versions also match. No new network
parameters or G5distance channels were introduced. This is one seeded pacing
intervention; broader causal or multi-seed reliability is not established.

## Frozen results

Final checkpoint256only,CPUfloat32,firing2101,same9development requests,single
scene-defined seed,threshold0.5. No postprocessing,threshold/horizon/quota search,
checkpoint selection or reserved evaluation. The per-step allowance is fixed
from the64step training schedule and is unchanged in128step review.

| Measure | G4 at64 | G6 at64 | G6 at128 |
|---|---:|---:|---:|
| All nine families pass | 2/9 | **9/9** | **9/9** |
| Access | 2/9 | 9/9 | 9/9 |
| Coverage | 6/9 | 9/9 | 9/9 |
| Facade | 9/9 | 9/9 | 9/9 |
| Each other family | 9/9 | 9/9 | 9/9 |
| Median volume error,percentage points | 0.152 | {stats['median_error_percentage_points']:.3f} | {stats['median_error_percentage_points']:.3f} |
| Maximum volume error,percentage points | 0.166 | {stats['max_error_percentage_points']:.3f} | {stats['max_error_percentage_points']:.3f} |
| Median teacher IoU | {old['summary']['64']['median_teacher_iou']:.4f} | {r['summary']['64']['median_teacher_iou']:.4f} | {r['summary']['128']['median_teacher_iou']:.4f} |

The five gates are:all-nine validity at64,median error<=2percentage points,
maximum error<=4points,all-nine validity at128,and each mass change<=5%.
All pass. Every field has100%cube-supported mass. Lowest site-third bulk
fraction is{stats['minimum_third_fraction']:.4f} against0.08minimum;highest facade
contact fraction is{stats['maximum_facade_fraction']:.4f} against0.15maximum.

| Case | Occupied voxels | Per-step allowance | First global-cap step | Facade fraction | Both horizons |
|---|---:|---:|---:|---:|---|
{rows}

## What changed

G4 exhausted volume early. G6 spreads admissions across the growth sequence.
These development outputs reach the global cap at steps
{stats['global_cap_step_range'][0]}–{stats['global_cap_step_range'][1]}, after connecting and
distributing mass successfully. The retained training fraction starting at
global capacity fell from{100*training['G4']['fraction_at_global_cap']:.1f}% inG4 to
{100*training['G6']['fraction_at_global_cap']:.1f}% inG6. This is consistent with the timing
hypothesis,although it does not isolate each effect through learned trajectories.

The prior TRAIN-only pacing diagnostic used fixed G4weights and exposed facade
and stability failures. This result comes from a separate fresh model trained
under pacing. Do not combine its development numbers with that earlier27case
TRAIN diagnostic as if they were one evaluation set.

Column3 of G6admission counts records rejection by the effective per-step
allowance,not solely by the global cap. Quota and every effective ceiling are
saved separately. G4/G6raw rejection counts are therefore not directly
comparable. The all-offer pre-admission candidate remains a diagnostic,not a
no-guard rollout or a causal cap-removal experiment.

## Limits of this milestone

Complete-cube geometry enforces thickness. Budget and saturation enforce size
limits and eventual fixed occupancy; this is not independent evidence of a
learned self-stabilizing dynamical system. Access,coverage and facade passes
are observed under this hybrid transition. The contract concerns overall
building volume and geometric support,not interior layout or structural safety.

The nine development requests have been reused during development. This result
is a pilot gate,not untouched generalization evidence or final deployment
approval. Only one training seed has been evaluated. No visual inspection of
G6geometry is claimed by this numerical review.

## Next phase — frozen generalization and visual review

Keep the exact candidate recorded in candidate-freeze.json. Next evaluate the
four previously reserved scene variants,each at16%,24% and32%requested volume:
12requests,at64 and128steps,with the same seed,threshold,quota and metrics.
No teacher geometry is needed for inference or the nine-family evaluation.
First verify reconstruction of the existing input channels against archived
TRAIN fixtures. Report every case; do not tune or discard failures after seeing
the results. next-generalization-protocol.json records this plan before access.
That evaluation has not been run during this review.

Then inspect saved volumes visually before deciding on research-interface
integration. No additional training or paid GPU job is needed for this next
local evaluation. MG7 remains the live model until that separate decision.

## Preservation

Original evidenceZIP and receipt,checkpoint,source snapshot,all18observations,
pairing checks,training-capacity comparison,candidate hashes and continuation
instructions are saved and archived locally. Repository synchronization remains
pending;the checkout's older RESUME is stale. Use this folder's RESUME.json.
The milestone archive is on the same disk,not an off-device backup. No Drive,
paid retry,push,publication or live-model replacement occurred.
'''
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('RESUME.json',dict(status='G6passes all5frozen development gates;qualified research candidate,frozen before reserved evaluation',run=candidate['run'],checkpoint_sha256=candidate['checkpoint_sha256'],previous=str(BASE/'G6-Paced-Growth-2026-10-04/RESUME.json'),candidate=str(OUT/'candidate-freeze.json'),review=str(OUT/'REVIEW.md'),next='Execute the saved next-generalization-protocol.json locally:verify7channel context reconstruction on archivedTRAIN fixtures first,then exactly4reserved scenes x3requests at64/128 with fixedfinal256model,CPUfloat32,firing2101,threshold0.5 and samequota. Preserve everyfailure,no tuning/drop/reroll. No teacher input or new training needed. Visually inspect saved volumes,then separately decide research UI integration. Reserved evaluation and visual review not yet performed.',reserved_evaluated=False,visual_review_completed=False,paid_run_authorized=False,repository_sync_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save('project-record.json',dict(event='G6first complete development gate in generation series',date='2026-10-04',run=candidate['run'],development_gate_passed=True,deployment_approved=False,results=r['summary'],gates=r['gates'],candidate=candidate,training_capacity=training,decision='Freeze final256candidate;advance to untouched reserved evaluation and visual review before any live integration.',limitations=['Single seed','Reused development','Enforced thickness/budget/stability','No reserved evaluation yet','No visual review yet','Repository sync and off-device backup pending'],changes='Read-only evidence review,new local records and frozen next-phase plan;no model/training/live modifications.'))
shutil.copyfile(__file__,OUT/'summarize-review.py');shutil.copyfile(Path(__file__).with_name('make_review_g6.py'),OUT/'build-review-script.py')
save('source-fingerprints.json',{p.relative_to(OUT/'source').as_posix():sha(p.read_bytes()) for p in sorted((OUT/'source').rglob('*.py'))})
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',{'files':files});files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
ar=dict(archive=str(archive),payloads=len(files),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),verified=True,off_device_backup=False)
archive.with_suffix('.receipt.json').write_text(json.dumps(ar,indent=2));print(json.dumps(dict(stats=stats,training=training,candidate=candidate['checkpoint_sha256'],archive=ar),indent=2))
