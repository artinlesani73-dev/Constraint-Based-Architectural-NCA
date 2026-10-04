from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G11-R1-Prototype-2026-10-04-v2')
r=json.loads((OUT/'result.json').read_text());a=json.loads((OUT/'audit.json').read_text())['summary']
table=[]
for label,steps in r['summary'].items():
 for step,v in steps.items():
  table.append(f"| {label} | {step} | {v['valid']}/{v['evaluated']} | {v['median_error']*100:.3f} | {v['max_error']*100:.3f} |")
ws=json.loads((OUT/'witnesses.json').read_text())
changes=[]
for step in [64,128]:
 old={o['case']:o for o in r['observations'] if o['model']=='G10' and o['steps']==step}
 new=[o for o in r['observations'] if o['model']=='G11-R1' and o['steps']==step]
 gains=[o['case'] for o in new if o['score']['contract_pass'] and not old[o['case']]['score']['contract_pass']]
 losses=[o['case'] for o in new if not o['score']['contract_pass'] and old[o['case']]['score']['contract_pass']]
 changes.append(dict(steps=step,gains=gains,losses=losses))
p=a['procedural_share128']
report=f"""# G11-R1 local prototype review — 2026-10-04

G11-R1 repairs all four TRAIN family failures but is NOT accepted: eight cases
fail stability and the largest 64-step volume error is 4.344 percentage points,
above the unchanged 4-point maximum. G10 itself has three TRAIN stability
failures here; its earlier 69-case evaluation was a different set.

## Implemented change

A separate inference adapter reserves a context-derived connected full-cube
witness meeting the existing nine families. Fixed G10 proposal weights grow
around it. Candidate unions must leave enough capacity to complete the witness
and must preserve its final facade ratio. Existing legality, connected cube
growth, quota and total volume ceiling remain in force.

This is explicitly a hybrid planner/NCA. Witness cubes receive priority and
bypass learned scores and stochastic firing. Learned cubes still use score >0.5
and firing probability 0.5. The same random stream continues to update hidden
state. No optimizer update or new paid training occurred.

The witness covers connection and minimum site coverage; it does not prescribe
the full requested mass. Its remaining completion cost is counted by exact voxel
union, including overlaps. Planning uses site geometry, not teacher shapes.
The route and coverage code are deterministic and versioned.

## Frozen local comparison

All 45 existing TRAIN inputs were used, with fixed G10 final checkpoint427,
CPU float32, two threads, firing seed2101 and horizons64/128. Each model was
run once to128;64 is its recorded prefix. These are not independent horizon
replays. No held-out scene was evaluated or selected for this prototype.

{r['certified_witnesses']}/45 witnesses pass the nine-family certificate and cap
before inference. Witness planning median time: {np.median([w['seconds'] for w in ws]):.3f}s.
All 45 paired cases were evaluated if all certificates succeeded; certificate
failures are preserved and must not be omitted from an overall success claim.

| Model | Steps | Nine families | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Stability (<=5% growth64–128): G10 {a['G10']['stable']}/45;
G11-R1 {a['G11-R1']['stable']}/45.
Maximum growth: G10 {a['G10']['max_growth']*100:.2f}%;
G11-R1 {a['G11-R1']['max_growth']*100:.2f}%.
Reserved witness fully present by64 in {a['witness_complete64']}/45 cases.

Median128-step rollout time, excluding planning:
G10 {a['G10']['median_seconds128']:.3f}s;
G11-R1 {a['G11-R1']['median_seconds128']:.3f}s.
These single-run local CPU timings are descriptive, not a repeated latency
benchmark or GPU performance estimate.

## Attribution and limitations

Planner-born share of newly occupied voxels at128:
minimum {p['min']*100:.1f}%, median {p['median']*100:.1f}%,
maximum {p['max']*100:.1f}%.
The seed is excluded from the denominator. A cube admitted through the planner
is counted as procedural even if the network might also have proposed it.
These shares record the executed decision path; they are not causal estimates
of how many voxels would be impossible without planning.

The method reserves a particular route and can bias morphology. Passing these
synthetic TRAIN cases does not establish generalization, diversity, architectural
quality or mechanical safety. The witness deliberately builds in some metric
requirements; passing them is not evidence the NCA learned those requirements.
No strict completion deadline is guaranteed for unseen geometry.

## Audit and preservation

Replayed all 5,760 hybrid birth/provenance accounts: no deletion, no illegal
births, unchanged per-step/global ceilings, exact witness-union reservation,
and agreement with saved64/128 output fields. Checkpoint hash is unchanged.
Full-cube depth/connectivity were checked on both output horizons for both models.
Saved four boundary checks cover valid certification, cap shortage, a severed
critical bridge and blocked-plane invalidation; they test certificate predicates,
not completeness of search. Boundary checks ran alongside the paired evaluation.

One first attempt failed before model inference because a NumPy Boolean was
not JSON serializable. Its source and partial output are preserved separately
and copied into this archive. The v2 attempt converts that scalar to bool;
the model/algorithm did not change as a result.

All eight comparison sheets were visually inspected. They show raw exposed
voxel surfaces with existing-context wireframes. They do not show interior
sections. Individual outcomes and failed families remain in case JSON records.

## Decision and next step

Retain G11-R1 as a local experimental hybrid baseline. Do not replace MG7.
Inspect the numerical gates and planner share together; do not describe this
as a newly trained model or automatic G10 improvement.

Next inspect scheduling on the saved TRAIN traces before consuming reserved
cases: distinguish unused per-step capacity, whole-cube packing and proposals
blocked by the witness/facade guards. Current trace records aggregate rejection
only; instrument exact reasons in one bounded diagnostic if necessary. All
witnesses finish by64, so remaining late volume is surrounding mass, not an
unfinished connection. Do not assume planner-first priority alone is causal.
Select one scheduling change with explicit before/after semantics. Keep quotas,
horizons and acceptance thresholds unchanged; no blind sweep or paid training.
Only after TRAIN timing and volume gates pass should a new, disjoint reserved
set be frozen for paired raw/hybrid assessment. No live preview admission yet.

Raw states, birth provenance, per-step accounts, witness paths, contexts, source,
checkpoint and metrics are archived. Original G10 evidence is untouched.
Repository synchronization remains pending; use this RESUME.json.
The verified archive is a same-disk copy, not an off-device backup.
No Drive action, push, publication or paid run occurred.
"""
(OUT/'REVIEW.md').write_text(report,encoding='utf-8')
(OUT/'paired-changes.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
# Preserve in-progress resume rather than overwrite its history.
shutil.copyfile(OUT/'RESUME.json',OUT/'RESUME-started.json')
(OUT/'RESUME.json').write_text(json.dumps(dict(status='G11-R1 implemented; TRAIN families45/45 but timing/volume gates fail; not accepted',previous=str(OUT.parent/'G11-Allocation-Design-2026-10-04/RESUME.json'),next='Diagnose unused quota and exact rejection reasons on TRAIN; all witnesses finish by64. Select one bounded scheduling correction without changing quota/horizon/thresholds. Do not consume reserved scenes or launch paid training yet.',summary=r['summary'],audit_summary=a,live_model='MG7 unchanged',paid_run_authorized=False,repository_sync_pending=True,off_device_backup_pending=True),indent=2),encoding='utf-8')
(OUT/'project-record.json').write_text(json.dumps(dict(event='G11-R1 fixed-weight hybrid prototype',summary=r['summary'],paired=changes,model_training=False,heldout_evaluation=False,promotion=False),indent=2),encoding='utf-8')
shutil.copytree(OUT.parent/'G11-R1-Prototype-2026-10-04',OUT/'failed-first-attempt')
shutil.copyfile(Path(__file__).with_name('render_g10.py'),OUT/'render_g10.py')
shutil.copyfile(__file__,OUT/'finalize.py')
sha=lambda b:hashlib.sha256(b).hexdigest()
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
(OUT/'milestone-manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')
files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(OUT/n,n)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files)
 assert all(sha(z.read(n))==s for n,s in files.items())
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False)
archive.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))

