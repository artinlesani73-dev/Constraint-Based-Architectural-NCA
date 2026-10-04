from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G11-R3-Ledger-2026-10-04';OLD=BASE/'G11-R1-Prototype-2026-10-04-v2'
sha=lambda b:hashlib.sha256(b).hexdigest()
parent_code=(OLD/'source/g11_reservation.py').read_text()
new_code=(OUT/'source/g11_reservation.py').read_text()
expected_prefix=parent_code.replace('G11-R1:','G11-R3:').replace('cap=C if seed else min(C,int(field.sum())+K)','cap=C if seed else cumulative_cap(C,K,step)')
assert new_code.startswith(expected_prefix) and new_code[len(expected_prefix):].strip().startswith('def cumulative_cap(')
r=json.loads((OUT/'result.json').read_text());a=json.loads((OUT/'audit.json').read_text())['summary']
checked=[]
for p in sorted((OUT/'cases').iterdir()):
 for name in ['context.npz','witness.npz','G10-64.npz','G10-128.npz','G10-trajectory.npz','raw-terminal.npz']:
  with np.load(p/name) as x,np.load(OLD/'cases'/p.name/name) as y:
   assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
 with np.load(p/'hybrid-states.npz') as x:assert all(np.isfinite(x[k]).all() for k in x.files)
 with np.load(p/'G11-R3-trajectory.npz') as x:assert np.isin(x['provenance'],[0,1,2]).all()
 checked.append(p.name)
assert len(checked)==45
gate=all(v['valid']==45 and v['median_error']<=.02 and v['max_error']<=.04 for v in r['summary']['G11-R3'].values()) and a['G11-R3']['stable']==45 and r['certified_witnesses']==45
summaries={}
for label,folder in [('G11-R1',OLD),('G11-R2',BASE/'G11-R2-Packing-2026-10-04'),('G11-R3',OUT)]:
 summaries[label]=dict(metrics=json.loads((folder/'result.json').read_text())['summary'][label],stability=json.loads((folder/'audit.json').read_text())['summary'][label]['stable'])
table=[]
for label,v in summaries.items():
 for step,s in v['metrics'].items():
  table.append(f"| {label} | {step} | {s['valid']}/45 | {s['median_error']*100:.3f} | {s['max_error']*100:.3f} | {v['stability']}/45 |")
nextstep=('Freeze R3 source, checkpoint and protocol. Create one genuinely new geometry set with physical-context disjointness against all prior TRAIN and exposed cases. Evaluate paired G10/R3 once on that new set and all69 old regression cases. Include certificate failures; no tuning or paid training. Only then consider a labelled hybrid preview.' if gate else 'Retain R1 and R3 evidence without promotion. Review remaining individual failures before choosing any further change; no automatic sweep, reserved evaluation or paid training.')
title='R3 passes all frozen TRAIN gates; ready for independent evaluation.' if gate else 'R3 fails at least one frozen TRAIN gate; no admission.'
report=f"""# G11-R3 cumulative allowance review — 2026-10-04

{title}

## One changed rule

R3 starts from R1, restoring witness-first lexicographic ordering followed by
learned score ordering. R2's smallest-first heuristic is discarded.

Previously each non-seed step allowed total mass up to min(C,current_mass+K);
any unused allowance disappeared. R3 uses min(C,27+(t-1)*K) at step t, where
K=max(9,ceil((C-27)/63)). C and K are unchanged, as are the64/128 evaluation
horizons and acceptance thresholds. At step64 the cumulative ceiling is C.

This explicitly relaxes the instantaneous per-step limit: a later step may add
more than K by using earlier unspent allowance. It neither raises the final cap
nor grants more than the original ideal cumulative allowance. The seed still
admits one full cube only. A cap is a maximum, not a guarantee of growth.

No objective, weights, training distribution or evaluator changed.
This is an inference architecture experiment with fixed G10 final427 weights.
The method remains a hybrid: reserved witness cubes bypass learned score/firing;
other admitted cubes require score>0.5 and Bernoulli0.5 firing.

## Frozen TRAIN comparison

All45 cases are existing TRAIN examples. No new held-out data or optimizer
updates were used. Same deterministic CPU float32 execution, two threads,
firing seed2101. Each rollout runs to128 and reports its64 prefix; these are
not independently replayed horizons. Original G10 output arrays were reused
and checked exactly, including contexts and terminal state.
The R3 witnesses are array-identical to R1, so this comparison isolates the
admission schedule, not a new planner.

| Hybrid version | Steps | Nine families | Median volume error (pp) | Max error (pp) | Stable64–128 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Raw G10 on these same45 cases:41/45 nine-family passes at both horizons and
42/45 stability passes. Its earlier69-case results refer to a different set.

R3 maximum growth64–128: {a['G11-R3']['max_growth']*100:.3f}%.
Witnesses fully present by64: {a['witness_complete64']}/45.
R3 median128-step CPU rollout time: {a['G11-R3']['median_seconds128']:.3f}s,
excluding planning. Single local timings do not establish production latency.

Planner-born fraction of added voxels at128:
{a['procedural_share128']['min']*100:.1f}% minimum,
{a['procedural_share128']['median']*100:.1f}% median,
{a['procedural_share128']['max']*100:.1f}% maximum.
Seed excluded. This records the admission path, not causal attribution.
Neither planner-enforced connection/coverage nor hard-cap stability should be
described as something the NCA independently learned.

## Verification

All5,760 hybrid birth accounts pass monotonicity, legality, the NEW cumulative
ceiling, global cap, witness-union reservation and saved-output equality.
Both output horizons pass connectivity/full-cube invariants; all90 saved hybrid
states are finite. Fixed checkpoint hash is unchanged. All45 cached raw G10
cases and witness/context arrays match the prior run exactly.
Pre-inference arithmetic checks covered tiny/saturated caps and the late-seed
single-cube premise. They are not a complete delayed-seed neural replay.

All eight geometry comparison sheets were visually inspected: raw voxel
surfaces, no smoothing or filling, with context wireframes. They show building
massing, not designed interiors or structural certification.
The45 correlated requests and one firing seed are insufficient for a
generalization or diversity claim.

## Decision and next step

{nextstep}

MG7 remains live. A TRAIN pass is preparation for independent assessment,
not deployment acceptance. Keep original NCA outputs distinguishable from
hybrid outputs in any future interface.

## Preservation

All source, protocol, boundary checks, checkpoint, contexts, witnesses, per-case
metrics, birth provenance, states, figures and aggregate decisions are archived.
Previous failures remain intact. Read RESUME.json here next.
Repository synchronization remains pending; original reports are untouched.
No paid run, Drive operation, push, publication or live-model change occurred.
The hash-verified archive is on the same disk, not an off-device backup.
"""
(OUT/'REVIEW.md').write_text(report,encoding='utf-8')
decision=dict(train_gates_pass=gate,deployment_accepted=False,live_promoted=False,summary=summaries,baseline_and_witness_equivalence_cases=checked,next=nextstep)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
shutil.copyfile(OUT/'RESUME.json',OUT/'RESUME-started.json')
(OUT/'RESUME.json').write_text(json.dumps(dict(status=title,previous=str(BASE/'G11-R2-Packing-2026-10-04/RESUME.json'),next=nextstep,live_model='MG7 unchanged',paid_run_authorized=False,repository_sync_pending=True,off_device_backup_pending=True),indent=2),encoding='utf-8')
(OUT/'project-record.json').write_text(json.dumps(dict(event='R3 cumulative admission TRAIN experiment complete',decision=decision),indent=2),encoding='utf-8')
for name in ['make_g11_r3.py','render_g10.py',Path(__file__).name]:shutil.copyfile(Path(__file__).with_name(name),OUT/name)
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
print(json.dumps(dict(train_gates_pass=gate,receipt=receipt),indent=2))

