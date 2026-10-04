from pathlib import Path
import json,hashlib,zipfile,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs')
OUT=BASE/'G10-Final-Review-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
r=json.loads((OUT/'result.json').read_text())
ex=json.loads((OUT/'execution.json').read_text())
audit=json.loads((OUT/'result-audit.json').read_text())
selected=[o for o in r['observations'] if not o['cohort'].startswith('baseline_')]
stability=[s for s in r['stability'] if not s['cohort'].startswith('baseline_')]
failed=[o for o in selected if o['steps']==128 and not o['score']['contract_pass']]
assert not r['accepted'] and len(selected)==138 and len(stability)==69
assert len(failed)==18 and all(o['unused_capacity']==0 for o in failed)
assert all(s['relative_mass_change']<=.05 for s in stability)
table=[]
for cohort,steps in r['summary'].items():
 for step,s in steps.items():
  table.append(f"| {cohort} | {step} | {s['valid']}/{s['expected']} | {s['median_absolute_fraction_error']*100:.3f} | {s['max_absolute_fraction_error']*100:.3f} |")
ledger=[]
for o in selected:
 if o['steps']!=64:continue
 later=next(x for x in selected if x['case']==o['case'] and x['steps']==128)
 st=next(s for s in stability if s['case']==o['case'])
 failures=', '.join(k for k,v in later['score']['family_pass'].items() if not v) or 'none'
 ledger.append(f"| {o['case']} | {sum(o['score']['family_pass'].values())}/9 | {sum(later['score']['family_pass'].values())}/9 | {st['relative_mass_change']*100:.2f}% | {failures} |")
pairs=[]
for group,steps in audit['paired_summary'].items():
 for step,p in steps.items():
  pairs.append(f"### {group}, {step} steps\n\nBaseline {p['baseline']}; G10 {p['g10']}. Gains: {', '.join(p['improved']) or 'none'}. Losses: {', '.join(p['regressed']) or 'none'}.\n")
report=f"""# G10 final review — 2026-10-04

G10 passes the volume-error and stability gates but fails overall acceptance:
45/57 regression cases and 6/12 new cases pass all nine families at both horizons.
The one-sided ranking experiment resolves the measured timing problem in this
evaluation, while leaving connection and coverage failures. It is experimental
evidence, not a live replacement. Preserve G8 and G9 for comparison; MG7 remains live.

## Verified experiment

Run 20261004T133117Z_60e4ff105497 completed 427 updates on Tesla T4 in
{ex['run_result']['wall_seconds']:.2f} controlled seconds (about 7m56s), within 600.
Peak reserved memory: {ex['run_result']['worker']['peak_reserved']/1048576:.0f} MiB.
Final checkpoint SHA-256: {ex['checkpoint_sha256']}.

The original returned ZIP and receipt are retained. All 1,719 payload hashes,
unique membership and package identity were verified. Expected Python 3.13.15,
Torch 2.11.0+cu130, NumPy 2.1.3, CUDA 13.0 and cuDNN 92700 matched.
Complete-payload/state recovery checks at updates 2 and 3 passed, along with
12 original and 8 paced device probes. All 427 row choices/start hashes and
saved start arrays were checked, including 27,328 step accounts and ceilings.
The initial numerical payload, all row/start choices and terminal sampler/RNG
states match G9. Pacing/inference source remains byte-identical.
This is one seeded comparison, not independent replication; full training
trajectories were not independently replayed.

Only the ranking auxiliary's gradient through the other-candidate reference
was stopped. Its forward value and advancing-candidate gradient are preserved.
Training phases: 277 seed-access, 15,036 connected, 11,995 advance-access and
20 no-teacher-route steps; all no-route events were below capacity.
Ranking was active on 8,663 steps. These phase counts are descriptive.
The comparison does not prove that all resulting model behavior is caused by
direct gradient suppression: shared parameters and subsequent trajectories interact.

## Frozen evaluation

Final checkpoint 427 only; deterministic CPU float32, two threads, firing seed
2101, horizons 64 and 128, unchanged thresholds and nine constraint families.
The 57 old cases are regression evidence. The four new scenes with three volume
requests each were frozen before training and evaluated identically for G10,
G9 and G8. They are now consumed and must become regression evidence in future work.

| Cohort/model | Steps | All nine | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Regression and fresh_reserved refer to G10; baseline cohorts name their model.
All volume-error gates pass. G10 passes the 5% growth limit on all 69 cases;
maximum growth from 64 to 128 is {max(s['relative_mass_change'] for s in stability)*100:.3f}%.
This is stability under a hard volume cap, not independently learned self-stabilization.

On the same old 57 cases, G9 passed 47 at 64 and 49 at 128, versus G10's 45.
On the same new 12, G9 and G8 each pass 4, versus G10's 6 at both horizons.
G8 passes fresh stability; G9 does not. G10 therefore improves the new sample
without dominating either reference across the full evaluation.
Three requests share each scene, and only one training seed was used; this is
not evidence of broad real-site generalization.

## Failure diagnosis and interpretation

There are 18 persistent failures: 15 access failures and three disjoint coverage
failures. The coverage cases are g1-unequal_building_heights-1-v16,
g7-vertical-reserved-2-v16 and g8-reserved-3-v16.
Every persistent failure exhausts its volume allowance by 128.
Additional monotone growth cannot repair those states once the cap is reached.

The seven other families pass all G10 cases at both horizons: facade, ground,
legality, sparsity, spill, support and thickness. Geometric support is not
structural certification. Some gates follow partly from hard admission rules.

The evidence supports a remaining allocation problem: finite volume has been
irreversibly committed without satisfying all spatial requirements. It does
not establish which replacement mechanism will solve that problem.
Restoring fill timing alone is insufficient.

## Visual and numerical audit

All 62 scene plates were inspected through 16 overview sheets: G10's 69 cases
and both references on the new 12, at both horizons. Plates contain raw exposed
voxel surfaces and front/plan projections at 0.8 m spacing, without smoothing
or filling. The projections are not interior sections.
Stepped volumes have depth, but some upward connections terminate below their
destination; coverage failures remain unevenly distributed.
These are massing experiments, not claims of finished architectural spaces.
A 9/9 plate label reports families, not the overall acceptance decision.

All 186 saved states are finite. All 93 independent 64/128 rollout pairs have
identical first-64 birth arrays. No rerolls, checkpoint selection, altered
thresholds or postprocessing were used.

## Decision and next step

Close G10 as a useful but unaccepted experiment. Retain its timing improvement
and fresh-case gains alongside its regressions. Do not promote it or launch
another loss-weight variant automatically.

Next prepare one architecture and budget-allocation decision using TRAIN data:
compare (a) learned reversible occupancy updates, allowing misplaced volume
to be relocated, with (b) explicitly labelled hybrid, budget-aware completion
that reserves capacity for remaining connections before admitting bulk growth.
Assess locality, connectivity/thickness preservation, runtime, training burden
and fidelity to the project's volume concept. Neither proposal is validated yet.

Choose one bounded prototype after that review. Use existing TRAIN failures
and an optimistic reachability/budget feasibility calculation to reject
infeasible designs before paid training. Keep the same nine families and
occupancy-as-volume semantics. Do not add room/shelter targets, silently apply
procedural repair, extend horizons merely to pass, or begin an unbounded sweep.
Any new experiment needs a frozen protocol and genuinely new reserved scenes;
the current 69 cases are regression cases. No paid run is authorized here.

## Paired case changes

{chr(10).join(pairs)}
## Full G10 case ledger

| Case | Families 64 | Families 128 | Growth 64–128 | Failure at 128 |
|---|---:|---:|---:|---|
{chr(10).join(ledger)}

## Preservation and resume

This directory retains the original full returned ZIP and receipt, all returned
checkpoints inside that ZIP, extracted final checkpoint, training verification,
frozen sources/configuration/splits, 93 contexts, 186 raw observations and birth
traces, metrics, comparisons, 62 plates and 16 overview sheets.
A manifest and verified archive preserve this milestone. The archive is on
the same disk and is not an off-device backup.
Read RESUME.json in this directory to continue. Repository synchronization
remains pending; its older resume record is stale. The original next-phase
report remains untouched. No Drive action, publication, push, paid retry or
live model change occurred.
"""
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('candidate-status.json',dict(run='20261004T133117Z_60e4ff105497',checkpoint=427,checkpoint_sha256=ex['checkpoint_sha256'],accepted=False,live_promoted=False,status='stable_but_connection_and_coverage_failures',live_model='MG7 unchanged',g8_and_g9_retained=True))
save('RESUME.json',dict(status='G10 verified; frozen evaluation and visual review complete; not accepted',previous=str(BASE/'G10-One-Sided-Training-2026-10-04/RESUME.json'),review=str(OUT/'REVIEW.md'),next='Prepare architecture/budget-allocation decision: learned reversible occupancy versus explicitly labelled hybrid capacity-aware completion. Use TRAIN-only feasibility evidence before choosing one bounded prototype. No automatic paid training or loss sweep.',regression_cases_for_next=69,fresh_reserved_consumed=True,paid_run_authorized=False,repository_sync_pending=True,off_device_backup_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save('project-record.json',dict(event='G10 GPU return verified and frozen review complete',summary=r['summary'],gates=r['gates'],paired=audit['paired_summary'],accepted=False,stability_pass_cases=69,persistent_failure_cases=18,visual_review_completed=True,full_evidence_preserved=True,decision='Retain timing improvement as evidence; review irreversible volume allocation before another paid experiment'))
for script in ['make_review_g10.py','render_g10.py','overview_g10.py','close_g10_review.py']:
 shutil.copyfile(Path(__file__).with_name(script),OUT/script)
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',dict(files=files))
files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files)
 assert all(sha(z.read(k))==v for k,v in files.items())
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False)
with archive.with_suffix('.receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))

