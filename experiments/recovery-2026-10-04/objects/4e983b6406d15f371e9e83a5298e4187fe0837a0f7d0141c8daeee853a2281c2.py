from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G7-Final-Review-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
r=json.loads((OUT/'result.json').read_text());execution=json.loads((OUT/'execution.json').read_text());audit=json.loads((OUT/'result-audit.json').read_text())
table=[]
for cohort in ['regression','fresh_reserved']:
 for step in [64,128]:
  s=r['summary'][cohort][str(step)]
  table.append(f"| {cohort} | {step} | {s['valid']}/{s['expected']} | {100*s['median_absolute_fraction_error']:.3f} | {100*s['max_absolute_fraction_error']:.3f} |")
details=[]
for s in r['stability']:
 a,b=[next(o for o in r['observations'] if o['case']==s['case'] and o['steps']==step) for step in [64,128]]
 details.append(f"| {s['case']} | {'Pass' if a['score']['contract_pass'] else 'Access fail'} | {'Pass' if b['score']['contract_pass'] else 'Access fail'} | {100*a['absolute_fraction_error']:.3f} | {100*s['relative_mass_change']:.2f}% |")
timing=[]
for o in r['observations']:
 with np.load(OUT/f"observations/{o['case']}-{o['steps']}.npz") as a:
  counts=a['admission_counts'];growth=np.flatnonzero(counts[:,6]>0)
  timing.append(dict(case=o['case'],steps=o['steps'],first_growth_step=int(growth[0]+1) if len(growth) else None,zero_growth_before_cap=int(((counts[:,6]==0)&(counts[:,0]<o['budget']['ceiling'])).sum()),unused_capacity=o['unused_capacity'],cap_step=o['first_ceiling_step']))
save('growth-timing-observations.json',dict(saved_counts_only=True,observations=timing,interpretation='Descriptive timing only;no new rollout,optimization or causal attribution.'))
stable={c:sum(s['relative_mass_change']<=.05 for s in r['stability'] if s['cohort']==c) for c in ['regression','fresh_reserved']}
maxchange=max(s['relative_mass_change'] for s in r['stability'])
report=f'''# G7 review — vertical training diversity — 2026-10-04

**The GPU run is valid, but G7 fails the frozen acceptance gates. Do not promote it.**
It fixes one G6 connection failure, yet size accuracy at64 steps and64-to128
stability regress substantially. MG7 remains live; G6 remains the prior research
reference and is itself not generally qualified for deployment.

## Verified evidence

Run20261004T103602Z_1da578905a51 completed256 updates in
{execution['run_result']['wall_seconds']:.3f} controlled seconds
({execution['run_result']['worker']['wall_seconds']:.3f} worker seconds), within600.
Peak GPU reservation was{execution['run_result']['worker']['peak_reserved']/1024**2:.0f}MiB.
Verified original ZIP SHA256, all1035 payload hashes, package manifest and
runtime/identity, device probes and both full-payload/state recovery checks.
Verified256 row selections and teacher starts, all16384 admission accounts and
temporary caps, finite saved training states and final legal connected cube unions.
The exact original evidence ZIP retains every checkpoint and training array;
the review uses only the frozen final256 checkpoint.

Checkpoint SHA256: `{execution['checkpoint_sha256']}`.
All45 reconstructed TRAIN contexts match package input bytes and seeds. Initial
parameters and paced model/loss source match G6 exactly. Data distribution and
per-row exposure differ. All33 independently executed128-step rollouts match
their64-step birth-mask prefixes exactly; all saved hidden states are finite.
These checks establish engineering integrity, not model quality.

## Frozen results, separate cohorts

| Cohort | Steps | All nine families | Median volume error (pp) | Maximum volume error (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

The21 regression requests contain9 reused development requests and12 consumed
G6 reserved requests. The12 fresh reserved requests are first-use G7 synthetic
variations. They are now exposed evidence and must not be called untouched in
future tuning. No teachers, routes or labels were used for these inference calls.
No checkpoint, threshold, seed, quota or horizon was selected after inspection.

Access is the only failing family; each of the other eight passes in every case
at both horizons. However, all-nine validity is only one part of the acceptance
gate. The separate requested-volume and stability requirements matter: at64,
both cohorts exceed the median2pp and maximum4pp error limits. At128 those size
errors pass, but only{stable['regression']}/21 regression and{stable['fresh_reserved']}/12
fresh cases stay within5% mass change. Maximum mass growth is{100*maxchange:.2f}%.
The visual plates' PASS label denotes **all nine families only**, not complete
acceptance. See this report and result.json for the separate size/stability gates.

## Comparison with G6

On the same21 legacy requests, G6 passed19/21 at both horizons. G7 passes19/21
at64 and20/21 at128. `g1-unequal_building_heights-1-v24` now connects at both
horizons. Its16% request still misses the east interface. A new64-step miss
appears in `g1-offset_interfaces-y4-v16`, which connects by128.
G6 satisfied size/stability limits on these legacy cohorts; G7 does not.
No G6 result on the new G7 reserved scenes is claimed.

Fresh16% requests0 and1 miss the east interface even at128. Fresh16% request2
misses at64 but connects by128; request3 passes both. All fresh24%/32% requests
pass the nine-family conjunction, but their substantial later growth still fails
stability. Three access misses remain at128 across the two cohorts, and all
three have exhausted the volume cap. More monotone growth steps cannot fix
their occupancy. Other failures are delayed growth rather than saturation.

## Visual interpretation

Inspected all33 outputs at both horizons through six overview sheets of the
22 full scene plates. They remain genuine three-dimensional voxel masses with
coarse terraces, protrusions and open exterior space, rather than single-cell
paths. Higher-volume outputs grow noticeably wider and taller after64; the
extra mass is visible, not merely a numerical artifact. The low-volume failing
upward scenes stop below their east connection; the downward case stops above
it. Whole-cube support does not make these finished architectural designs.
Projection views show occupied extents, not interior sections or habitable rooms.

## What this experiment establishes

Broader training data alone, at the same256-update budget, is insufficient for
the frozen target. This does not prove the data change was wrong. Increasing
the dataset from27 to45 lowers per-example exposure to5-6 visits and changes
the sequence of gradients. One seed cannot disentangle diversity, exposure and
optimization. Do not infer that512 updates, a new loss or a larger grid will
necessarily solve this result.

The saved trajectories expose two different issues: delayed filling within64
steps, and allocation that exhausts the cap without completing a connection.
Extending the review horizon would conceal the first problem while leaving the
second. The prescribed64/128 results and thresholds remain unchanged.
Thickness, global budgets and eventual saturation stability are partly enforced
by the hybrid admission algorithm; these are not independently learned behavior.

## Decision and next step

Keep this run as a failed acceptance result with useful partial gains. Do not
replace G6 or MG7, change the acceptance horizon, or launch another paid job.
Next do one consolidated local TRAIN-only diagnosis using existing G6/G7 weights
and training traces: examine seed-start delays, eligible proposal scores and
unused per-step allowance, and compare near-connection allocation before cap.
Separate reduced training exposure from the known lack of temporal ordering in
teacher membership labels before choosing one intervention. Avoid tuning on the
now-consumed reserved cases. Prepare a concrete new frozen proposal only after
that diagnosis; any paid allowance requires explicit approval.

## Individual case ledger

Errors are absolute requested fraction differences in percentage points.

| Case | Nine families at64 | Nine families at128 | Error at64 (pp) | Mass change |
|---|---|---|---:|---:|
{chr(10).join(details)}

## Preservation

Original ZIP+receipt, final checkpoint, all66 reviewed output arrays,33 contexts,
source/config/split snapshots, package provenance, exact metrics, timing counts,
comparison audit and visual plates are retained. Local verified milestone ZIP
is a same-disk archive, not an off-device backup. No Drive, new training, push,
publication or live changes occurred. Repository synchronization remains pending;
resume from this folder's RESUME.json instead of the checkout's stale D098 file.
'''
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('RESUME.json',dict(status='G7 final256 evaluated;acceptance failed;all evidence preserved',run='20261004T103602Z_1da578905a51',previous=str(BASE/'G7-Vertical-Training-2026-10-04-v2/RESUME.json'),checkpoint_sha256=execution['checkpoint_sha256'],result=str(OUT/'result.json'),review=str(OUT/'REVIEW.md'),next='One consolidated TRAIN-only diagnosis of G6/G7 start latency, proposal availability, per-step quota use and connection allocation using existing weights/traces;separate delayed growth from cap-exhausted misses and reduced exposure. Freeze one justified next change before requesting another paid job. Do not retune on exposed fresh G7 reserved scenes or select intermediate checkpoints.',fresh_reserved_consumed=True,visual_review_completed=True,paid_retry_authorized=False,repository_sync_pending=True,off_device_backup_pending=True,drive_operations=0,live_model='MG7 unchanged',research_reference='G6 retained;G7 not promoted'))
save('project-record.json',dict(event='G7 returned GPU evidence review',run='20261004T103602Z_1da578905a51',integrity_passed=True,quality_accepted=False,summary=r['summary'],gates=r['gates'],stable_cases=stable,maximum_mass_growth=maxchange,decision='Do not promote;diagnose TRAIN dynamics locally before choosing another intervention',changes='Read-only returned run review,66 frozen observations,visual inspection,new durable local records;no training/model/live modifications',limitations=['Single seed','Synthetic relatives','Unequal per-row exposure versusG6','Access,size-at64 andstability failures','Repository sync andoff-device backup pending']))
for script in ['make_review_g7.py','make_render_g7.py','finalize_review_g7.py']:shutil.copyfile(Path(__file__).with_name(script),OUT/script)
files={f.relative_to(OUT).as_posix():sha(f.read_bytes()) for f in sorted(OUT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
save('milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
receipt=dict(archive=str(archive),sha256=sha(archive.read_bytes()),payloads=len(files),bytes=archive.stat().st_size,verified=True,off_device_backup=False)
with archive.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
