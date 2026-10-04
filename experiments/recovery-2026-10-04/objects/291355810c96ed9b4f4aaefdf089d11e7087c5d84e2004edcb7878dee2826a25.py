from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Scale-2026-10-04');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((P/'freeze.json').read_text());assert all(sha(P/n)==h for n,h in freeze.items())
old=P.parent/'G11-R3-Skins-2026-10-04/runs/bbf64c65230c4531bf50a887ade4f59d'
for step in [64,128]:
 with np.load(P/f'32-{step}.npz') as a,np.load(old/f'R3-{step}.npz') as b:assert np.array_equal(a['field'],b['field'])
for n in [32,40]:
 with np.load(P/f'{n}-trajectory.npz') as a:
  assert all(np.isfinite(a[k]).all() for k in a.files) and np.array_equal(a['births'],a['provenance']>0) and a['births'].sum(0).max()<=1
  for step in [64,128]:
   with np.load(P/f'{n}-{step}.npz') as b:assert np.array_equal(b['field'],a[f'state{step}'][0,0].astype(bool))
rows=[]
for n in [32,40]:
 r=json.loads((P/f'{n}-result.json').read_text());rows.append(f"| {n}³ | {n*.8:.1f}m | {r['passed']} | {r['times']['rollout128']:.2f}s | {sum(r['times'].values()):.2f}s | {r['peak_process_working_set_bytes']/2**20:.0f}MiB |")
report='''# Larger-domain CPU probe — 2026-10-04

Original R3 successfully ran at40³ on one expanded physical site with unchanged weights, evaluator, planner and admission policy. This is feasibility evidence, not broad validation or a studio release. The approved visual skins are retained unchanged.

| Grid | Physical edge | Per-run checks pass | Rollout128 | Measured stages total | Peak working set |
|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

Measured stages are context construction, route, witness/certificate and128-step rollout. Their sum excludes interpreter/model loading, saved-output serialization and final scoring. Peak working set is Windows process-wide peak resident memory since startup, including imports, model and trajectory capture; not incremental model memory, GPU VRAM or training memory. Both separate workers used CPU float32, two threads, same seed2102 and24% request. No warm-up or repeated timing trials; these values are observations, not guaranteed latency. Each worker had a180-second cap and completed normally.

40³ has1.953 times the cells of32³. Voxel size stayed0.8m: physical domain edge grew25.6m to32m. Building X/Y coordinates/extents and connection Y scaled1.25 with integer rounding; connection X was reattached to its facade. Building heights, entrance extent, street band, physical opportunity padding and2.4m cube thickness stayed fixed. Therefore this is not finer resolution, a uniform3D enlargement or the same physical problem. The allowed-domain fraction and volume cap recompute for the changed scene. The same RNG seed on different tensor shapes does not mean identical per-cell firing.

Both grids passed all nine families at64 and128, absolute volume error<=4pp and late growth<=5%; both had zero late growth. Finite trajectory/state fields, unique voxel births, provenance and state/field agreement verified.32³ final fields exactly match the archived parent at both horizons. Pre-run frozen input/source hashes unchanged. Both final views inspected at the same drawing scale; renderer boundary handling was adapted to the actual grid dimensions (original renderer assumed32).

Decision:40³ is a viable next experimental studio scale for this case; do not extrapolate to64³ costs or claim larger-grid generalization. Next prepare a versioned40³ studio path with dimension-aware rendering/validation and clear separation of larger-domain versus finer-resolution modes, then one focused integration check. Keep32³ as the reference. Increasing resolution at constant physical size would require explicit thickness/interface/padding semantics and is not authorized by this result alone.

All scenes/configs, contexts, certificates, route/witness fields, weights/source, trajectories, traces, runtime versions, metrics, controller limits and logs are preserved. No paid training or GPU run. Original studios including skins8018 and MG7 unchanged. Repository synchronization/off-device backup pending; no Drive, push or publication. Same-disk archive is not off-device backup.
'''
(P/'REVIEW.md').write_text(report,encoding='utf-8');(P/'RESUME.json').write_text(json.dumps(dict(status='32 and40 grid CPU probe complete; both pass on one site',next='dimension-aware experimental40³ studio preparation; keep32³ reference',previous='../G11-R3-Skins-2026-10-04/RESUME.md',report='REVIEW.md'),indent=2));
for n in ['render_r3_scale.py','close_r3_scale.py','render_g10.py']:shutil.copyfile(n,P/n)
files={p.relative_to(P).as_posix():sha(p) for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts};(P/'milestone-manifest.json').write_text(json.dumps(files,indent=2));files['milestone-manifest.json']=sha(P/'milestone-manifest.json');zp=P.with_suffix('.verified.zip')
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(P/n,n)
with zipfile.ZipFile(zp) as z:
 assert len(z.namelist())==len(files) and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
r=dict(bytes=zp.stat().st_size,sha256=sha(zp),payloads=len(files),verified=True);zp.with_suffix('.receipt.json').write_text(json.dumps(r,indent=2));print(r)
