from pathlib import Path
import json,hashlib,zipfile,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G8-Final-Review-2026-10-04-v2'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
r=json.loads((OUT/'result.json').read_text());ex=json.loads((OUT/'execution.json').read_text());pair=json.loads((OUT/'g7-prefix-verification.json').read_text());audit=json.loads((OUT/'result-audit.json').read_text())
assert pair['all_numerical_equal']
summary=[]
for cohort in ['regression','fresh_reserved']:
 for step in [64,128]:
  s=r['summary'][cohort][str(step)];summary.append(f"| {cohort} | {step} | {s['valid']}/{s['expected']} | {100*s['median_absolute_fraction_error']:.3f} | {100*s['max_absolute_fraction_error']:.3f} |")
stable={c:sum(x['relative_mass_change']<=.05 for x in r['stability'] if x['cohort']==c) for c in ['regression','fresh_reserved']};maxchange=max(x['relative_mass_change'] for x in r['stability'])
failures=[o for o in r['observations'] if not o['score']['contract_pass']];failtable=[]
for o in failures:
 if o['steps']!=128:continue
 failtable.append(f"| {o['case']} | {', '.join(k for k,v in o['score']['family_pass'].items() if not v)} | {o['direct_interface_contact_voxels']['E_east']} | {o['unused_capacity']} |")
changes=audit['paired_regression_comparison'];paired={}
for step in [64,128]:
 a=[c for c in changes if c['steps']==step];paired[str(step)]=dict(g7_valid=sum(c['g7_pass'] for c in a),g8_valid=sum(c['g8_pass'] for c in a),improved=[c['case'] for c in a if not c['g7_pass'] and c['g8_pass']],regressed=[c['case'] for c in a if c['g7_pass'] and not c['g8_pass']])
save('comparison-summary.json',paired)
details=[]
for s in r['stability']:
 a,b=[next(o for o in r['observations'] if o['case']==s['case'] and o['steps']==t) for t in [64,128]]
 details.append(f"| {s['case']} | {sum(a['score']['family_pass'].values())}/9 | {sum(b['score']['family_pass'].values())}/9 | {100*a['absolute_fraction_error']:.3f} | {100*s['relative_mass_change']:.3f}% |")
assert not r['accepted'] # This report's decision must match the observed failed gate.
report=f'''# G8 exposure experiment — verified final427 review

**Extra training restores volume accuracy and improves legacy connection results,
but G8 still fails the complete acceptance gate.** Keep it as an experimental
research result. MG7 remains live; no model promotion occurred.

## Run integrity and controlled comparison

Run20261004T111257Z_f5eb0598cf17 completed427 retained updates in
{ex['run_result']['wall_seconds']:.3f} controlled seconds
({ex['run_result']['worker']['wall_seconds']:.3f} worker seconds), within600.
Peak reservation:{ex['run_result']['worker']['peak_reserved']/1024**2:.0f}MiB.
Verified the original ZIP/receipt and all1719 payload hashes, exact package
manifest, runtime/identity, admission probes and both full-payload/state recovery
checks. All427 start fields, row selections and27328 transition accounts/caps
were verified against saved traces and geometry. All45 reconstructed TRAIN
contexts and seeds match the packaged fixtures.

G8 at update256 is numerically identical to G7 at update256 across model,
optimizer, sampler, training trace, random state and remaining non-identity
payload fields. Run identity is deliberately separate. This establishes a
reproduced common training prefix, rather than a merely similar initialization.
We evaluated only final427; update256 was checked for reproducibility, not
selected as an alternative model. Final checkpoint SHA256:
`{ex['checkpoint_sha256']}`.

The remaining171 updates are the only training intervention relative to G7.
This supports attribution along this one deterministic training trajectory,
not a multi-seed estimate that more training always helps. Data, architecture,
loss, hard transition, thresholds, firing and review horizons stayed frozen.

## Separate frozen cohorts

| Cohort | Steps | All nine families | Median size error (pp) | Maximum size error (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(summary)}

The33 regression cases combine the original9 development,12 G6 reserved and12
G7 reserved requests. The12 fresh G8 cases are new synthetic combinations, not
the same cohort used to report G7's earlier fresh score. Do not compare those
two fresh-cohort percentages as if they measured the same cases. G7 was not
evaluated on the new G8 fresh cohort. No teacher geometry or route was provided
at inference. Fresh cases are now consumed evidence for any future work.

Stability within5% holds for{stable['regression']}/33 regression and
{stable['fresh_reserved']}/12 fresh requests; maximum growth from64 to128 is
{100*maxchange:.3f}%. Every individual gate is recorded in result.json; passing
the eight other families or matching volume does not excuse access failures.
Full geometric support and thickness are partly enforced by cube admission,
and do not establish structural safety or habitable interior space.

## Paired legacy improvement and remaining failures

On the same33 cases, G7 passed{paired['64']['g7_valid']}/33 at64 and
{paired['128']['g7_valid']}/33 at128; G8 passes{paired['64']['g8_valid']}/33 and
{paired['128']['g8_valid']}/33 respectively. Individual improvements and regressions
are preserved in comparison-summary.json and result-audit.json. This is a
substantial improvement in the previously observed size/timing problem, not
universal monotonic improvement in every connection.

The following cases still fail at128:

| Case | Failed families | East-interface occupied cells | Unused volume capacity |
|---|---|---:|---:|
{chr(10).join(failtable)}

For a cap-exhausted failure, additional monotone growth steps cannot repair the
field: there is no remaining volume allowance and the transition cannot remove
or redistribute occupancy. The remaining problem is where mass is allocated
before saturation. Simply increasing the voxel count, relaxing the threshold,
or accepting a longer horizon is not justified by this result.

## Visual review

Reviewed the full scene plates through64-step and128-step overview sheets.
Outputs have substantial three-dimensional depth, stepped surfaces and exterior
gaps. The failed cases visibly miss a required connection despite bulk elsewhere;
good overall volume is therefore insufficient. The rendered colors and
projections are diagnostic geometry, not interior sections. Plate labels show
the number of passing families; overall acceptance also requires size and
stability gates. Full-resolution plates and all90 saved observations are retained.
This synthetic one-seed study does not establish diversity or architectural
quality across real sites.

## Decision and next action

Close the exposure experiment: the extra updates addressed size/timing but did
not guarantee access. Do not automatically launch another longer training run.
The next focused design task is to make completing required connections before
spending the mass budget an explicit learning priority within the existing
access family. Compare that proposed training objective with the current static
teacher-membership loss using TRAIN-only diagnostics; keep the other eight
families and volume semantics unchanged. Do not silently add a procedural bridge
or relabel repaired output as raw learned generation. Any procedural fallback
would be a separately documented hybrid design decision.

Prepare one concrete access-priority proposal before more paid compute, informed
by the earlier G5 cue failure and G6 objective audit. Preserve G8 as the latest
evaluated reference, with failure labels, without claiming it met the frozen
deployment gate. A saved-result gallery is acceptable for review; unrestricted
live replacement remains deferred.

## Individual case ledger

| Case | Families at64 | Families at128 | Error at64 (pp) | Mass change |
|---|---:|---:|---:|---:|
{chr(10).join(details)}

## Preservation and implementation note

Saved original full ZIP+receipt (including every checkpoint and training array),
final checkpoint,90 evaluated fields/states/proposals/birth traces,45 contexts,
frozen source/config/splits, runtime, comparison and integrity records,30 geometry
plates and overview sheets. The independent128-step runs have exactly matching
64-step birth prefixes, and all final states are finite.

An initial local reviewer adaptation failed before verification or inference
because a broad text substitution changed a hash function name. The failed
script and error note remain in G8-Final-Review-2026-10-04; this corrected review
is in the separate v2 directory. This was a local review-script error, not a
GPU/model failure or scientific retry. Original evidence was never altered.

The verified milestone ZIP is a same-disk archive, not an off-device backup.
Repository synchronization remains pending; resume from this folder's RESUME.json
rather than the checkout's stale D098 record. No Drive operation, further paid
training, push, publication or live-model replacement occurred.
'''
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('candidate-status.json',dict(run='20261004T111257Z_f5eb0598cf17',checkpoint=427,checkpoint_sha256=ex['checkpoint_sha256'],status='experimental_reference_with_access_failures',accepted=False,live_promoted=False,reason='Frozen all-nine acceptance gates fail despite improved timing/volume',all_numerical_prefix_equal_g7_at256=True))
save('RESUME.json',dict(status='G8 final427 reviewed;exposure experiment complete;acceptance failed',run='20261004T111257Z_f5eb0598cf17',previous=str(BASE/'G8-Exposure-Training-2026-10-04/RESUME.json'),checkpoint_sha256=ex['checkpoint_sha256'],review=str(OUT/'REVIEW.md'),next='Prepare one TRAIN-only access-priority objective/design proposal. Use prior G6 objective audit and G5 cue failure;keep nine families. Address connection completion before volume saturation rather than another blind exposure extension. No automatic new paid run or procedural repair. G8 is an experimental reference,not gate-passing deployment.',fresh_reserved_consumed=True,visual_review_completed=True,paid_retry_authorized=False,repository_sync_pending=True,drive_operations=0,off_device_backup_pending=True,live_model='MG7 unchanged'))
save('project-record.json',dict(event='G8 returned evidence and frozen evaluation complete',summary=r['summary'],gates=r['gates'],paired_legacy=paired,stability_passes=stable,maximum_mass_growth=maxchange,integrity_passed=True,common_g7_prefix_exact=True,accepted=False,decision='Preserve G8 reference;propose access-priority learning before further compute',changes='Local evidence verification and frozen review only;no new optimizer updates or live changes'))
for script in ['make_review_g8.py','prepare_g8_review_helpers.py','overview_g8.py','finalize_g8_review.py']:shutil.copyfile(Path(__file__).with_name(script),OUT/script)
failed=BASE/'G8-Final-Review-2026-10-04';shutil.copytree(failed,OUT/'review-attempt-1')
files={f.relative_to(OUT).as_posix():sha(f.read_bytes()) for f in sorted(OUT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
save('milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files) and all(sha(z.read(k))==v for k,v in files.items())
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False)
with archive.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(dict(archive=receipt,stable=stable,max_mass_change=maxchange,paired=paired),indent=2))
