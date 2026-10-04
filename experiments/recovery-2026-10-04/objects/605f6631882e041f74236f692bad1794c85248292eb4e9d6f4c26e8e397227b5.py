from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Route-Options-2026-10-04');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((P/'freeze.json').read_text());assert all(sha(P/n)==h for n,h in frozen.items())
baseline=json.loads((P/'baseline/manifest.json').read_text());assert all(sha(P/'baseline'/n)==h for n,h in baseline.items())
r=json.loads((P/'result.json').read_text())
for v in r['variants']:
 if v['status']!='evaluated':continue
 with np.load(P/f"{v['variant']}-trajectory.npz") as a:
  assert all(np.isfinite(a[n]).all() for n in a.files)
  assert np.array_equal(a['births'],a['provenance']>0) and a['births'].sum(0).max()<=1
  for n in [64,128]:
   f=a[f'state{n}'][0,0].astype(bool)
   assert f.sum()==v['outputs'][str(n)]['score']['occupied_voxels']
   with np.load(P/f"{v['variant']}-{n}.npz") as b:assert np.array_equal(f,b['field'])
rows='\n'.join(f"| {v['variant']} | {v.get('passes',False)} | {v.get('growth',0):.2%} | {v.get('outputs',{}).get('128',{}).get('planner_share',0):.1%} |" for v in r['variants'])
notes=f'''# Route alternatives — exploratory result, 2026-10-04

R3 can produce different distributions of building volume on this fixed custom site through an explicit planner change, while retaining the same checkpoint, firing seed2102,24% request, nine families, witness-completion policy and cumulative budget. The original R3 reference is unchanged and remains the studio default.

The new experimental helper chooses a legal cube-origin waypoint in the middle X plane near the20th or80th percentile of reachable Y origins and near the median entrance height, with deterministic ties. It concatenates shortest seed-to-waypoint and waypoint-to-east paths, then applies the existing witness growth and certification. This is a procedural style control, not newly learned diversity, not a new constraint family, and not a guarantee that every arbitrary site admits a route. Missing routes/certificates are failures, not replaced candidates. No tuning/retry was performed.

| Alternative | All per-run checks pass | Growth64–128 | Planner-born share128 |
|---|---|---:|---:|
{rows}

All three final fields contain1389 occupied voxels. Volume amount is held fixed; placement changes. Jaccard distances at128: baseline versus low-Y0.268; baseline versus high-Y0.356; low-Y versus high-Y0.464. Jaccard distance is1 minus intersection/union, not a percentage of all voxels moved or a measure of architectural value.

Visual review: all three raw voxel views inspected in comparison.png. The variants visibly redistribute mass toward different sides of the connection. They remain related block-like forms; this does not yet demonstrate broad typological variety or an aesthetic improvement. One exposed custom site, one request and one seed: exploratory feasibility only. No broad release claim.

Verification: frozen pre-run inputs/source unchanged; original baseline payload hashes match; exact condition equality to parent checked before inference; finite trajectories, unique voxel births, provenance and saved-state/field agreement checked. Both alternatives pass all nine families at64/128, absolute volume error<=4pp and late-growth<=5%. Existing frozen source evaluates all metrics. Source, model, contexts through baseline, route/witness fields, trajectories, traces, per-case metrics and figures are preserved.

Next: expose these two modes as explicitly experimental alternatives alongside the original route, with exact route-mode provenance and clear infeasibility display. Assess transfer on a small fixed set before adopting either as a new default. No new training is currently indicated by this result. Do not claim that changing planner routing trains the NCA to understand design diversity.

Resume from this folder. Prior: ../G11-R3-Revisit-2026-10-04/RESUME.md. No active inference. Studio8016 remains unchanged. Repository synchronization and off-device backup are pending; same-disk verified ZIP is not a separate-device backup. No paid training, Drive operation, remote push, publication or MG7 replacement.
'''
(P/'REVIEW.md').write_text(notes,encoding='utf-8');(P/'RESUME.json').write_text(json.dumps(dict(status='two exploratory route alternatives complete; both pass on one fixed site',next='Explicit experimental route selector and bounded transfer assessment; original R3 remains default',report='REVIEW.md',previous='../G11-R3-Revisit-2026-10-04/RESUME.md'),indent=2),encoding='utf-8')
for n in ['render_r3_route_options.py','close_r3_route_options.py','render_g10.py']:shutil.copyfile(n,P/n)
files={p.relative_to(P).as_posix():sha(p) for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(P/'milestone-manifest.json').write_text(json.dumps(files,indent=2));files['milestone-manifest.json']=sha(P/'milestone-manifest.json');zp=P.with_suffix('.verified.zip')
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(P/n,n)
with zipfile.ZipFile(zp) as z:
 assert len(z.namelist())==len(files)
 assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
receipt=dict(bytes=zp.stat().st_size,sha256=sha(zp),payloads=len(files),verified=True);zp.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt)
