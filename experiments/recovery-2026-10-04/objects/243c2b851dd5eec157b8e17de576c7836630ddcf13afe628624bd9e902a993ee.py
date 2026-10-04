from pathlib import Path
import json,hashlib,zipfile,shutil
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Independent-Review-2026-10-04')
r=json.loads((OUT/'result.json').read_text());a=json.loads((OUT/'audit.json').read_text());v=json.loads((OUT/'visual-selection.json').read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
table=[]
for key,steps in r['summary'].items():
 for step,s in steps.items():
  table.append(f"| {key} | {step} | {s['valid']}/{s['expected']} | {s['evaluated']} | {s['median_error']*100:.3f} | {s['max_error']*100:.3f} |")
accepted=r['accepted']
nextstep=('Prepare a separate, clearly labelled local hybrid preview with G10 raw comparison, provenance and honest failure display. Keep MG7 live until the user reviews the preview. Before any general release, extend assessment to more seeds and geometry families and benchmark larger-grid costs separately.' if accepted else 'Keep R3 experimental; inspect the recorded individual failures without tuning on the new reserved set. Preserve all outputs and select a bounded next design before any new experiment. No promotion.')
headline='R3 passes all frozen gates on69 regression cases and12 new cases.' if accepted else 'R3 fails at least one frozen evaluation gate; no promotion.'
report=f"""# R3 independent review — 2026-10-04

{headline}

## What was frozen

R3's adapter and G10 final427 checkpoint were frozen by SHA-256 before inference.
The route helper was extracted from the earlier geometry audit and reproduced
all45 saved TRAIN routes exactly before use on new scenes. Planner, admission
ordering, cumulative allowance, score/firing, nine families and thresholds were
unchanged throughout evaluation. There were no training updates or retuning.

The69 previous cases are regression evidence. The12 new cases comprise four
new geometries, each at16%,24%,32% requested volume. Geometry hashes use the six
physical context channels, excluding scene names and requested volume.
They were disjoint from all38 previously known unique TRAIN/evaluation contexts
and from one another before inference. This is a small related synthetic set,
not broad out-of-distribution or real-building validation. It is now consumed;
future work must treat these cases as exposed regression evidence.

Same32-cubed domain resolution,0.8m voxels, CPU float32, deterministic two-thread
execution, firing seed2101. No independent training/firing seeds were added.
R3 and new-case G10 run to128 with64 recorded as a prefix, not a separate run.
Old G10 results were reused only after original hashes and exact context checks.

## Numerical results

| Cohort/model | Steps | All nine | Evaluated outputs | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Required: all nine families on every case; median error<=2 percentage points,
maximum<=4; and mass growth64–128<=5% for every case.
Certificate failures count in the denominator and fail admission; no geometry
was replaced, omitted or procedurally rescued after seeing results.

Stability:
{chr(10).join('- '+k+': '+str(s['stable'])+'/'+str(s['count'])+'; maximum growth '+(f"{s['max_growth']*100:.3f}%" if s['max_growth'] is not None else 'unavailable') for k,s in a['stats'].items())}

Certificate failures: {len(a['certificate_failures'])}.
R3's overall frozen numerical acceptance decision: {accepted}.
This decision is limited to the assessed pilot domain; it is not automatic
deployment approval or evidence of structural safety.

## Paired changes

{chr(10).join('- '+p['cohort']+' at'+str(p['steps'])+': '+str(len(p['gains']))+' family-pass gains, '+str(len(p['losses']))+' losses versus raw G10.' for p in a['paired'])}

Individual gains, losses, certificate failures and stability failures are
listed in audit.json and the per-case records. Every model uses the same
condition and request for each comparison. Three requests share each geometry
and are correlated; do not interpret12 requests as12 independent sites.

## What is learned and what is procedural

R3 is a hybrid. A geometry planner supplies a legal thick connection and minimum
coverage witness, and mandatory witness additions bypass neural firing/scores.
G10 weights choose other eligible additions subject to global admission guards.
Cumulative allowance carries unused capacity forward, allowing later individual
steps to exceed K while retaining the original cumulative and final ceilings.

Median planner-born share of added voxels at128:
regression {a['stats']['regression_R3'].get('planner_share_median',0)*100:.1f}%,
new sample {a['stats']['fresh_R3'].get('planner_share_median',0)*100:.1f}%.
Seed excluded; admission provenance is not causal attribution.
Hard-cap stability and planner-enforced connections are not independently
learned NCA capabilities. Preserving meaningful volumes does not define rooms,
habitable interiors, program or structural performance.

## Verification and visual review

All frozen input/source hashes remain unchanged. Checkpoint identity and hash
were verified. Every successful hybrid trajectory was checked for no deletion,
legal additions, cumulative/global ceilings, exact witness-union capacity and
agreement with saved trace masses. All saved hybrid horizon states are finite.
Both model outputs were checked for connected full-cube volume at each horizon.

Visual selection is explicit: all fresh cases plus every regression case with
a family or certificate failure in either model, at both horizons.
{len(v['cases'])} paired cases were rendered and every comparison sheet inspected.
Other regression cases received numerical review, not a claim of exhaustive
visual inspection. Raw voxel surfaces and context wireframes are shown without
smoothing; no interior-section or aesthetic certification is implied.

## Decision and next work

{nextstep}

Do not call this a freshly trained model, silently replace MG7, or claim a pure
local NCA system. Preserve raw G10 and hybrid outputs side by side. The nine
constraint families and building-volume concept remain unchanged.
No new paid training was needed.

## Preservation and resume

Source, fixed checkpoint, protocol, all81 contexts and certificate attempts,
324 per-model/horizon records (including failures), successful fields,
hybrid trajectories/provenance, fresh raw G10 trajectories, metrics and figures
are preserved in this folder and a verified archive. The archived original
G10 evidence remains the source for old raw trajectories.

Read RESUME.json here next. Repository synchronization is pending.
No Drive access, publication, remote push or live-model change occurred.
The verified archive is on the same disk, not an off-device backup.
"""
(OUT/'REVIEW.md').write_text(report,encoding='utf-8')
decision=dict(frozen_pilot_gates_pass=accepted,live_promoted=False,general_release_approved=False,next=nextstep,summary=r['summary'],gates=r['gates'])
(OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
shutil.copyfile(OUT/'RESUME.json',OUT/'RESUME-frozen.json')
(OUT/'RESUME.json').write_text(json.dumps(dict(status=headline,previous=str(OUT.parent/'G11-R3-Ledger-2026-10-04/RESUME.json'),next=nextstep,exposed_regression_cases_for_future=81,paid_training=False,live_model='MG7 unchanged',repository_sync_pending=True,off_device_backup_pending=True),indent=2),encoding='utf-8')
(OUT/'project-record.json').write_text(json.dumps(dict(event='R3 independent review complete',decision=decision,fresh_cases_now_consumed=True),indent=2),encoding='utf-8')
for n in ['render_g10.py','prepare_g11_independent_visuals.py',Path(__file__).name]:shutil.copyfile(Path(__file__).with_name(n),OUT/n)
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
print(json.dumps(dict(accepted=accepted,receipt=receipt),indent=2))

