from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G11-R2-Packing-2026-10-04';OLD=BASE/'G11-R1-Prototype-2026-10-04-v2';D=BASE/'G11-Scheduling-Diagnosis-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
# Check exact cached baseline observations and contexts; no assumed equivalence.
verified=[]
for p in sorted((OUT/'cases').iterdir()):
 for name in ['context.npz','G10-64.npz','G10-128.npz','G10-trajectory.npz','raw-terminal.npz']:
  with np.load(p/name) as a,np.load(OLD/'cases'/p.name/name) as b:
   assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
 verified.append(p.name)
assert len(verified)==45
r=json.loads((OUT/'result.json').read_text());a=json.loads((OUT/'audit.json').read_text())['summary'];oldr=json.loads((OLD/'result.json').read_text());olda=json.loads((OLD/'audit.json').read_text())['summary']
diag=json.loads((D/'result.json').read_text())
gate=all(x['valid']==45 and x['median_error']<=.02 and x['max_error']<=.04 for x in r['summary']['G11-R2'].values()) and a['G11-R2']['stable']==45
table=[]
for label,rr in [('G10',r),('G11-R1',oldr),('G11-R2',r)]:
 for step,s in rr['summary'][label].items():
  table.append(f"| {label} | {step} | {s['valid']}/45 | {s['median_error']*100:.3f} | {s['max_error']*100:.3f} |")
result=dict(train_gates_pass=gate,live_promoted=False,cached_baseline_array_equivalence_cases=verified,diagnosis_replays=8,paired_stability=dict(G10=a['G10']['stable'],R1=olda['G11-R1']['stable'],R2=a['G11-R2']['stable']))
(OUT/'decision.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
report=f"""# G11-R2 scheduling review — 2026-10-04

## Outcome

R2 is rejected: stability falls from37/45 to21/45 and maximum early volume
error rises from4.344 to6.262 percentage points. All45 still pass nine families,
but the complete TRAIN acceptance gates fail. Preserve R1 as the reference.
R2 is an experimental ordering variant, not a trained model or live replacement.

## Diagnosis before choosing the change

Replayed R1's eight unstable TRAIN cases for64 steps with additional observation
only. All eight birth sequences match their saved originals exactly.
They waste {sum(x['unused'] for x in diag)} voxel-step allowance in total across
{sum(x['steps'] for x in diag)} non-seed steps with unused capacity.
Only {sum(x['new_offer_fits'] for x in diag)} of these steps have a newly exposed
frontier offer that would fit; {sum(x['frozen_offer_fits'] for x in diag)} have a
remaining previously eligible offer that fits after the complete admission pass.
{sum(x['unoffered_fits'] for x in diag)} steps have a fitting but unoffered cube.
These categories can overlap. Reasons are sequential first-failure counts,
not an independent causal decomposition. Unused allowance is not itself proof
that another ordering could use all of it.

This made intra-step packing a reasonable single hypothesis, not an established
fix. All diagnostic traces and the instrumented code are retained.

## The one tested change

R1 visits reserved cubes lexicographically, then learned proposals by score.
R2 keeps the same two priority groups but repeatedly picks the cube requiring
the fewest new voxels, recalculating after each accepted overlap. Original
ordering breaks ties. Reserved cubes still bypass score/firing explicitly;
learned proposals still need score>0.5 and firing. Frozen eligibility, witness,
checkpoint, seed, quota K, total ceiling C, horizons and evaluator are unchanged.

This remains an explicitly global hybrid. It changes execution trajectories,
including later logits, even though weights do not change.

## Complete paired TRAIN results

One fixed45-case pass; no held-out scenes, optimizer update, threshold tuning,
paid run or second scheduling variant. G10 baseline arrays were reused from
the preceding run and checked for exact array equality in all45 cases.
R1 comparisons use its archived metrics. R2 was newly inferred to128, with64
as the recorded prefix; independent horizon replays were not performed.

| Model | Steps | Nine families | Median volume error (pp) | Maximum error (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Stability passes: G10 {a['G10']['stable']}/45;
R1 {olda['G11-R1']['stable']}/45; R2 {a['G11-R2']['stable']}/45.
R2 maximum growth64–128: {a['G11-R2']['max_growth']*100:.2f}%.
R2 completed witnesses by64: {a['witness_complete64']}/45.
R2 planner-born median fraction of newly occupied voxels at128:
{a['procedural_share128']['median']*100:.1f}%.
This is recorded admission attribution, not a causal measure of learned ability.

R2 median128-step CPU rollout: {a['G11-R2']['median_seconds128']:.3f}s;
R1 {olda['G11-R1']['median_seconds128']:.3f}s, excluding witness planning.
These single-run timings are descriptive.

## Interpretation and next step

Smallest-first does not necessarily maximize filled volume per step. Its early
choices change overlap geometry and subsequent neural proposals, so local
small-delta preference can worsen the longer trajectory. Do not mistake the
packing heuristic for an optimal solver.

Reject R2 as the selected schedule and retain R1.
The next design to assess is an explicit cumulative allowance ledger: unused
capacity from earlier steps remains available later, while the total planned
allowance and hard final volume cap stay fixed. Proposed non-seed envelope:
min(C, 27 + (t-1)*K) at step t, with the original single-cube seed phase.
This CHANGES the per-step rule from min(C,current_mass+K); it must be labelled
and tested as a new version, never presented as the unchanged quota.
The proposed envelope was checked arithmetically against all45 saved R1
trajectories: each fits within it, and its64-step ceiling equals C in every case.
This proves neither usable proposals nor success under the changed trajectory.
Next freeze one cumulative-ledger local experiment with R1 ordering restored,
including late-seed and cap boundary checks. Keep64/128 horizons and all
acceptance thresholds. No extra training or fresh reserved cases yet.

Do not consume new reserved scenes until the selected method passes the
complete TRAIN gates. decision.json records this failed experiment.

## Verification and preservation

All5,760 R2 birth/provenance accounts passed monotonicity, legality, per-step
and global ceilings, witness-union reservation and saved-horizon agreement.
Both output horizons were checked for full-cube depth and connectivity.
The G10 checkpoint hash is unchanged. All eight raw geometry comparison sheets
were inspected; passing family labels do not imply stability or habitability.

This archive includes sources, fixed checkpoint, contexts, witnesses, raw
and hybrid observations, provenance, metrics and the complete eight-case
diagnosis. Earlier failed and successful attempts remain intact. The source
renderer is included. Same-disk archives are not off-device backups.
Repository synchronization is pending. No Drive operation, push, publication
or live-model change occurred. Continue from RESUME.json in this directory.
"""
(OUT/'REVIEW.md').write_text(report,encoding='utf-8')
shutil.copyfile(OUT/'RESUME.json',OUT/'RESUME-started.json')
(OUT/'RESUME.json').write_text(json.dumps(dict(status='R2 completed; see decision.json for gates',previous=str(OLD/'RESUME.json'),train_gates_pass=gate,next='Reject R2 if gates fail; cumulative allowance envelope already checked on45 R1 traces. Freeze one versioned local ledger experiment using R1 ordering, including late-seed/cap boundary checks. This changes per-step admission, not total cap/horizons. No paid training or reserved evaluation yet.' if not gate else 'Freeze and assess on new disjoint reserved set; no paid training or promotion',live_model='MG7 unchanged',repository_sync_pending=True,off_device_backup_pending=True),indent=2),encoding='utf-8')
shutil.copytree(D,OUT/'scheduling-diagnosis')
for name in ['make_g11_r2.py','prepare_g11_r2_review.py','render_g10.py',Path(__file__).name]:shutil.copyfile(Path(__file__).with_name(name),OUT/name)
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
(OUT/'milestone-manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8');files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(OUT/n,n)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files)
 assert all(sha(z.read(n))==s for n,s in files.items())
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False)
archive.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(decision=result,receipt=receipt),indent=2))

