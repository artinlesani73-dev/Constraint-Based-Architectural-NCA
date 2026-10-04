from pathlib import Path
import json,hashlib,zipfile,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G9-Final-Review-2026-10-04'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
r=json.loads((OUT/'result.json').read_text());ex=json.loads((OUT/'execution.json').read_text());t=json.loads((OUT/'training-verification.json').read_text());audit=json.loads((OUT/'result-audit.json').read_text())
assert not r['accepted']
selected=[o for o in r['observations'] if o['cohort']!='baseline_fresh']
stability=[s for s in r['stability'] if s['cohort']!='baseline_fresh'];unstable=[s for s in stability if s['relative_mass_change']>.05]
persistent=[o for o in selected if o['steps']==128 and not o['score']['contract_pass']]
assert len(selected)==114 and len(persistent)==8 and all(o['unused_capacity']==0 for o in persistent)
table=[]
for cohort in ['regression','fresh_reserved','baseline_fresh']:
 for steps in ['64','128']:
  s=r['summary'][cohort][steps];table.append(f"| {cohort} | {steps} | {s['valid']}/{s['expected']} | {s['median_absolute_fraction_error']*100:.3f} | {s['max_absolute_fraction_error']*100:.3f} |")
case_table=[]
for o in selected:
 if o['steps']!=64:continue
 later=next(x for x in selected if x['case']==o['case'] and x['steps']==128)
 st=next(x for x in stability if x['case']==o['case'])
 case_table.append(f"| {o['case']} | {sum(o['score']['family_pass'].values())}/9 | {sum(later['score']['family_pass'].values())}/9 | {st['relative_mass_change']*100:.2f}% |")
report=f'''# G9 final review — 2026-10-04

G9 improves connections on the matched fresh sample but fails the frozen
acceptance criteria. Preserve it as an experimental candidate with timing and
coverage regressions. It is not a live replacement. G8 remains the stable
comparison reference; MG7 remains live. Neither G8 nor G9 passed broad admission.

## Verified run

Run20261004T120338Z_60498d5f0838 completed427 updates on TeslaT4 in
{ex['run_result']['wall_seconds']:.2f} controlled seconds (7m49s), within600.
Peak reserved memory was{ex['run_result']['worker']['peak_reserved']/1048576:.0f}MiB.
Verified all1719 evidence payload hashes, exact unique ZIP membership and package
identity; final checkpoint hash is {ex['checkpoint_sha256']}.
Expected Python3.13.15,Torch2.11.0+cu130,NumPy2.1.3,CUDA13.0,cuDNN92700 matched.
Both complete-payload/state recovery checks at2/3 passed, as did device probes.

All427 row choices/start hashes and saved start/terminal arrays were checked,
with all27328 step accounts and ceilings. Initial numerical payload, all row/start
choices, final sampler and RNG match G8; model weights diverge as expected.
Original inference/pacing source is byte-identical. This controls one seeded
comparison, not multiple-seed reliability or equal wall-clock compute.
Full training trajectories were not independently replayed.

The ranking term was active on7574/27328 steps. Recorded phases:277 seed-access,
10464 advance-access,16561 connected,26 no-teacher-route. All26 no-route events
were below capacity. These are useful diagnostics, not causal proof of the
remaining failures. Loss decomposition, finite fields and phase/account
consistency passed. No optimizer update or GPU retry was launched in this review.

## Frozen numerical evaluation

Final427 only, CPUfloat32, deterministic two-thread execution, firing seed2101,
unchanged64/128 horizons, thresholds and nine families. The45 previously exposed
cases are regression evidence. The12 new cases were frozen before G9 training
and first evaluated here, identically for G8 and G9; their labels were never
training inputs. They are now consumed for future split accounting.

| Cohort/model | Steps | All nine | Median volume error (pp) | Max error (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Regression and fresh_reserved rows are G9; baseline_fresh is G8 on the SAME new12.

Paired old45: G8=40/45 at both horizons; G9=38/45 at64,40/45 at128.
At64 G9 gains one case and loses three. At128 it gains two and loses two.
Paired fresh12: G8=5/12, G9=9/12 at both horizons, four improvements and
zero all-nine regressions. The three volume requests share each of four scenes;
these are correlated synthetic examples, not12 independent real sites.

All cohort volume-error gates pass at both horizons (median<=2pp,max<=4pp).
However,{len(unstable)}/57 G9 cases exceed5% mass growth between64 and128:
10 regression and2 fresh. Maximum growth is{max(x['relative_mass_change'] for x in stability)*100:.2f}%.
G8 passes stability on all57 corresponding cases, combining prior and fresh
evaluations. These numerical stability failures prevent promotion even where
nine families pass. Stability is growth to a hard cap, not independently learned
self-stabilization.

G9 has one persistent coverage failure, g7-vertical-reserved-0-v16, which also
fails access. Every other family except access passes all57 cases at both
horizons. The seven unaffected families are facade,ground,legality,sparsity,
spill,support,thickness. Geometric support is not structural certification.

## Individual gains, regressions and remaining failures

At64, G9 repairs g7-vertical-reserved-2-v16 but loses
g1-unequal_building_heights-1-v16, g7-vertical-reserved-0-v16 and g8-reserved-1-v24.
At128 the first loss recovers, and g8-reserved-2-v16 becomes an additional gain.
Fresh gains at both horizons are g9-reserved-0-v24, g9-reserved-2-v32,
g9-reserved-3-v16 and g9-reserved-3-v24.

The eight persistent failures are:
{chr(10).join('- '+o['case'] for o in persistent)}

All eight retain west contact, miss east contact, and exhaust their volume cap
by128. Extra monotone growth after that cannot repair them. At64 some have
unused volume allowance and later fill; this reveals two distinct problems:
late growth and incorrect final allocation. Extending the display to128 does
not solve all access failures or retroactively satisfy the64/128 stability gate.

## Visual review

Inspected all46 scene plates (three requests each), covering G9's57 cases and
G8's12 fresh counterparts at both horizons. Raw voxel volumes retain depth,
stepped surfaces and exterior gaps. Missed vertical endpoints are visible;
the coverage failure remains skewed toward the starting side. The smaller64
volumes and subsequent additions agree with the recorded growth differences.
These are coarse massing experiments, not interior/architectural-quality claims.
Plates show exposed voxel surfaces plus front/plan projections at0.8m spacing,
without smoothing/filling; projections are not interior sections.
A9/9 plate label indicates families only, not overall stability acceptance.

All138 saved evaluation states are finite. In every one of69 independent
64/128 rollout pairs, the first64 birth arrays match exactly.
No rerolls, best-checkpoint choice, threshold tuning or postprocessing occurred.

## Decision and next concrete work

Close G9 as useful but unqualified evidence. Preserve both G8 and G9 and the
paired fresh comparison; do not automatically choose G9 merely because its
fresh access score is better, and do not discard its gains.

Next perform ONE consolidated TRAIN-only comparison using fixed G8/G9 weights:
record proposal logits, firing, eligible progress/other groups and allowance use
on the same45 TRAIN contexts. At fixed saved states, separate gradients from
membership, volume/band and ranking. Check whether ranking lowers otherwise
useful offers below0.5 or whether step quota rejects well-scored progress.
This is a proposed diagnosis, not a conclusion already established here.

Use that evidence to select one design that preserves connection priority and
timely filling. Do not start an unbounded loss-weight sweep, another blind long
run, change horizon/quota just to pass, or add procedural repair silently.
Keep the nine families and volume semantics. Any next package needs its own
frozen protocol and explicit bounded paid allowance. For product work, a clearly
labelled comparison gallery may use these saved outputs; no live model swap
or claim of deployment readiness follows.

## Case ledger

| Case | Families64 | Families128 | Mass growth64–128 |
|---|---:|---:|---:|
{chr(10).join(case_table)}

## Preservation and resume

Original full returnedZIP+receipt, all checkpoints in thatZIP, source/config,
split hashes,57 G9 contexts plus12 baseline copies,138 raw evaluations and
birth traces, individual metrics,46 visual plates and12 overview sheets are
saved here and in a verified archive. Archive copies are on the same disk,
not an off-device backup. Read this folder's RESUME.json next.
Repository synchronization remains pending; checkout resume is stale.
No Drive operation, push, publication, new paid run or live-model change occurred.
'''
with (OUT/'REVIEW.md').open('x',encoding='utf-8') as f:f.write(report)
save('candidate-status.json',dict(run='20261004T120338Z_60498d5f0838',checkpoint=427,checkpoint_sha256=ex['checkpoint_sha256'],accepted=False,live_promoted=False,status='experimental_access_gains_with_stability_and_coverage_regressions',g8_reference_retained=True))
save('RESUME.json',dict(status='G9 fully verified/evaluated/visually reviewed;not admitted',previous=str(BASE/'G9-Access-Ranking-Training-2026-10-04-v2/RESUME.json'),review=str(OUT/'REVIEW.md'),next='ONE consolidated TRAIN-only G8/G9 fixed-weight proposal/allowance/gradient comparison on45TRAIN contexts, to distinguish score suppression from quota rejection before choosing one change. No automatic paid run, weight sweep, horizon changes or procedural bridge. Preserve both references and MG7 live.',fresh_reserved_consumed=True,paid_run_authorized=False,repository_sync_pending=True,off_device_backup_pending=True,drive_operations=0,live_model='MG7 unchanged'))
save('project-record.json',dict(event='G9 GPU return verified and frozen review complete',summary=r['summary'],gates=r['gates'],paired=audit['paired_summary'],stability_failures=unstable,accepted=False,visual_review_completed=True,full_evidence_preserved=True,decision='Retain access gains as evidence;diagnose score/fill tradeoff onTRAIN before new compute'))
for script in ['make_review_g9.py','render_g9.py','overview_g9.py','close_g9_review.py']:shutil.copyfile(Path(__file__).with_name(script),OUT/script)
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files)
 assert all(sha(z.read(k))==v for k,v in files.items())
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),payloads=len(files),verified=True,off_device_backup=False)
with archive.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))

