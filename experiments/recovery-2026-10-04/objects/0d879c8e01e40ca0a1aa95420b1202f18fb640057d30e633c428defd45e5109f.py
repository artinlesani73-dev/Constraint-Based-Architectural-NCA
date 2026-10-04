from pathlib import Path
import json,hashlib,zipfile,shutil,sys
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G6-Reserved-Review-2026-10-04';OLD=BASE/'G6-Final-Review-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
r=json.loads((OUT/'result.json').read_text());split=json.loads((OUT/'split-manifest.json').read_text());scenes={e['id']:e for e in split['entries']}
fingerprints=json.loads((OLD/'source-fingerprints.json').read_text())
assert all(sha((OUT/'source'/k).read_bytes())==v for k,v in fingerprints.items())
save('source-verification.json',dict(passed=True,files=len(fingerprints),fingerprints=fingerprints))
sys.path.insert(0,str(OUT/'source'));sys.dont_write_bytecode=True
from nca.contract import entrance_masks
diagnostics=[]
for st in r['stability']:
 case=st['case'];row=next(o for o in r['observations'] if o['case']==case and o['steps']==128)
 with np.load(OUT/f'observations/{case}-64.npz') as a,np.load(OUT/f'observations/{case}-128.npz') as b:
  assert np.array_equal(a['births'],b['births'][:64])
  field=b['field'];coords=np.argwhere(field)
  endpoints=entrance_masks(scenes[row['scene']]['scene'])
  nearest={k:int(np.abs(coords[:,None,:]-np.argwhere(m)[None,:,:]).sum(2).min()) for k,m in endpoints.items()}
  diagnostics.append(dict(case=case,prefix64_equal=True,nearest_occupied_to_interface_center_manhattan_cells=nearest))
z_pairs=[dict(scene=e['id'],z=[a['z'] for a in e['scene']['entrances']]) for e in split['entries'] if e['split']=='train']
assert all(p['z']==[8,8] for p in z_pairs)
save('diagnostics.json',dict(saved_fields_only=True,no_new_inference=True,training_elevation_pairs=z_pairs,observations=diagnostics,all_context_necessary_checks_pass=all(o['score']['context_necessary_checks_pass'] for o in r['observations'])))
cases=[]
for st in r['stability']:
 a,b=[next(o for o in r['observations'] if o['case']==st['case'] and o['steps']==s) for s in [64,128]]
 cases.append(f"| {a['case']} | {'Pass' if a['score']['contract_pass'] else 'Access fail'} | {'Pass' if b['score']['contract_pass'] else 'Access fail'} | {100*a['absolute_fraction_error']:.3f} | {100*st['relative_mass_change']:.3f}% |")
max_change=max(s['relative_mass_change'] for s in r['stability'])
extents=np.array([o['extent_zyx_cells'] for o in r['observations'] if o['steps']==64])*.8
report=f'''# G6 reserved-scene review — 2026-10-04

G6 passes **10 of 12 cases at both 64 and 128 steps**. The frozen gate requires
12/12 at both horizons, so this candidate does **not** qualify for unrestricted
research generation or live promotion. MG7 remains unchanged.

## Evaluation integrity

The final update-256 checkpoint was frozen before first use of these four scene
variants. Checkpoint SHA256: `{json.loads((OUT/'candidate-freeze.json').read_text())['checkpoint_sha256']}`.
The exact original protocol is copied in next-generalization-protocol.json; its
historical planned status is deliberately retained. execution.json and result.json
record its execution here. All 27 archived TRAIN contexts match reconstructed
seven-channel inputs byte-for-byte, with matching seeds and context hashes.
All frozen Python source fingerprints match. Every 128-step trajectory has an
identical first 64 birth masks to its independently executed 64-step trajectory.

All 12 scene/request pairs were evaluated once per prescribed horizon. CPU
float32, deterministic operations, two threads, firing seed 2101, proposal
threshold 0.5 and fixed quota max(9,ceil((C-27)/63)) were used. No teacher labels,
route input, new training, altered thresholds, retries, dropped failures or
postprocessing were used. No execution errors occurred. All necessary context
feasibility checks pass, which does not prove that every requested design is feasible.

These cases are synthetic relatives of prior scenes, not external architectural
validation. Their first-use results are now consumed evidence: future tuning
against them makes them a regression set, not untouched generalization data.

## Results

| Measure | 64 steps | 128 steps |
|---|---:|---:|
| All nine families | 10/12 | 10/12 |
| Access | 10/12 | 10/12 |
| Each of the other eight families | 12/12 | 12/12 |
| Median absolute volume error, percentage points | {100*r['summary']['64']['median_absolute_fraction_error']:.3f} | {100*r['summary']['128']['median_absolute_fraction_error']:.3f} |
| Maximum absolute volume error, percentage points | {100*r['summary']['64']['max_absolute_fraction_error']:.3f} | {100*r['summary']['128']['max_absolute_fraction_error']:.3f} |

All mass changes are below the frozen 5% tolerance; the largest is
{100*max_change:.3f}%. Nine fields are identical across horizons and three grow
slightly. All fields have 100% cube-supported mass. The two all-nine gates fail;
the error and stability gates pass. Do not pool these 12 cases with the nine
reused development cases to inflate a generalization claim.

| Case | 64 | 128 | 64-step error (pp) | Mass change |
|---|---|---|---:|---:|
{chr(10).join(cases)}

## Failure interpretation

Both failures are `g1-unequal_building_heights-1`, at 16% and 24% volume. Both
have eight occupied voxels touching the west interface and zero touching the
east interface. They reach across the site's X span but remain below the higher
east connection. The 16% output adds 17 cells after step 64 and reaches its cap
at step 66 without making that contact. The 24% output reaches its cap at step
64 and stays unchanged. More iterations alone cannot repair the saturated
add-only field. The 32% request passes on the same scene.

The frozen access metric starts its flood fill at the alphabetically first
interface, E_east. Therefore both reachability flags are false when east is
untouched. Supplemental direct-contact counts distinguish this from losing
west contact. The metric was not changed.

Every original TRAIN scene has west/east connection origins at z=8/8. Training
varies horizontal position, gap and obstruction but has no vertical connection
offset. That is a verified distribution gap. It supports testing broader
vertical training diversity; it does not prove that this alone will solve the
failure. Data, objective and finite local information propagation may all matter.
The prior destination-cue experiment G5 failed and should not be repeated without
a new mechanism and evidence.

## Visual review

All twelve 64-step geometries were inspected in isometric, front and plan views;
all twelve final 128-step plates were also inspected. The fields are chunky,
stepped building masses with open exterior space around them. They are no longer
one-voxel paths or flat platforms. At 64 steps their vertical bounding extents
range from {extents[:,0].min():.1f} to {extents[:,0].max():.1f} m; these bounds do not
imply uniform thickness. The low-volume failing case is shallower and spreads
sideways, while the 24% failing case grows tall near the west side yet misses the
east connection. Passing geometry still has coarse terraces, irregular additions
and limited demonstrated design diversity. Numerical validity is not architectural
quality, interior circulation, habitability or structural certification.

Final plates are in visuals-final-64/ and visuals-final-128/. The initial plates
in visuals/ had wireframes overlapping captions; they are retained as draft
history, with their rendering script, and superseded by the final layout. The
images use exposed voxel faces and orthographic occupancy projections; they are
not interior sections. Gray wireframes are context, not generated mass.

Thickness, budget and saturation stability are partly enforced by the hybrid
cube-admission algorithm. Do not describe these as independently learned NCA
self-organization. No diversity claim is possible from one firing seed.

## Decision and next implementation

Keep G6 as a frozen research reference. A read-only gallery can show all saved
results with failure labels; unrestricted interactive use and MG7 replacement
are deferred because the admission gate failed.

Next prepare one **training-distribution intervention**: add scene-defined
vertical connection offsets and building-height diversity, keeping the G6 model,
losses, pacing rule, 32-cube resolution and nine families fixed. Use TRAIN-only
teacher construction; freeze the new scene split and compute budget before any
training. Include the original TRAIN cases to limit forgetting. Do not train on
these twelve reserved outputs or use them to select a checkpoint. Freeze fresh
unseen combinations before the next run; label the present twelve as regression.
Check label feasibility and recovery in one consolidated local preparation pass,
then request one explicit Colab allowance. No next training package or paid job
has been launched by this review. Larger grids and deployment polish remain
planned after this specific reliability gap is addressed.

## Preservation and continuation

Saved all 24 fields, hidden states, proposals, birth masks, admission counts,
step ceilings, 12 contexts, exact checkpoint, source/config/split snapshots,
protocol, runtime, metrics and visual plates. The manifest and verified ZIP
preserve this milestone on the same disk; that is not an off-device backup.
No Drive operation was performed. Repository synchronization remains pending;
use this folder's RESUME.json rather than the checkout's older D098 resume.
Original G6 development evidence and the original report remain untouched.
'''
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('RESUME.json',dict(status='G6 reserved first-use review complete;10/12 passes at both horizons;gate failed',previous=str(OLD/'RESUME.json'),result=str(OUT/'result.json'),review=str(OUT/'REVIEW.md'),reserved_evaluated=True,visual_review_completed=True,live_model='MG7 unchanged',next='Prepare a single vertical-context training-distribution intervention with G6 model/loss/pacing unchanged. Freeze TRAIN/development/fresh unseen scene groups and label-feasibility report;present one concrete bounded Colab package and ask for its compute allowance. These12 cases are now exposed regression evidence. Do not tune,reroll or overwrite G6 first-use results.',paid_run_authorized=False,repository_sync_pending=True,drive_operations=0,off_device_backup_pending=True))
save('project-record.json',dict(event='G6 first reserved evaluation and visual review',date='2026-10-04',results=r['summary'],gates=r['gates'],candidate_promoted=False,decision='Retain G6 as reference;prepare vertical training diversity before unrestricted integration',changes=['New frozen local CPU evaluation of12scene/request pairs at2horizons','Saved24outputs and12contexts','Verified27TRAIN context reconstructions and24source/result accounts','Rendered and inspected geometry;retained initial layout drafts','Saved next-step decision and durable resume'],limitations=['Single training and firing seed','Synthetic relatives only','Two access failures','Generalization gate failed','Repository sync and off-device backup pending']))
shutil.copyfile(__file__,OUT/'finalization-script.py')
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files)
 assert all(sha(z.read(k))==v for k,v in files.items())
receipt=dict(archive=str(archive),payloads=len(files),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),verified=True,off_device_backup=False)
with archive.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(dict(archive=receipt,diagnostics=diagnostics),indent=2))
