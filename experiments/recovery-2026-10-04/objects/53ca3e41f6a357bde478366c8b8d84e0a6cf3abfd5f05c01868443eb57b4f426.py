from pathlib import Path
import json,hashlib,zipfile,shutil
import numpy as np
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Modes-2026-10-04');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'transfer/result.json').read_text());identity=json.loads((P/'identity.json').read_text());assert all(sha(P/n)==h for n,h in identity['files'].items())
for v in r:
 p=P/'runs'/v['id'];old=P.parent/'G11-R3-Variety-2026-10-04/cases'/v['scene']
 with np.load(p/'context.npz') as a,np.load(old/'context.npz') as b:assert np.array_equal(a['condition'],b['condition'])
 for step in [64,128]:
  with np.load(p/f'G10-{step}.npz') as a,np.load(old/f'G10-2101-{step}.npz') as b:assert np.array_equal(a['field'],b['field'])
summary=json.loads((P/'transfer/verification.json').read_text());summary.update(same_context_as_baseline=True,all_12_raw_g10_fields_exact_to_baseline=True,frozen_source_hashes=True,browser_default_original=True,browser_draft_mode_restored=True);(P/'transfer/verification.json').write_text(json.dumps(summary,indent=2))
notes='''# Experimental route modes — 2026-10-04

Studio http://127.0.0.1:8017/ now offers Original R3 (default), Low Y (experimental), High Y (experimental). These are explicit procedural route choices; checkpoint, nine families, downstream witness growth, cumulative admission and neural inference are unchanged. No new trained capability is claimed. Mode is preserved in requests, results, route.json, history labels, draft save/restore and edit-from-run. Older runs/drafts lacking a mode use original. No automatic fallback between modes. Invalid modes are rejected before run creation.

One bounded transfer check: wider gap, offset Y, partial obstacle, each at24%,seed2101 with both alternatives. Six of six passed all nine families at64/128,<=4pp per-output volume error and<=5% late growth. All were retained; no retries/tuning. These are three previously exposed synthetic sites, not held-out generalization or a production gate. Each route changes mass distribution while final volume remains equal to its site's baseline. Shape difference does not measure design quality. Nine raw final views (original plus alternatives) were visually inspected in transfer/comparison.png.

Verification: all run hashes and route-mode provenance verified. All six physical/request condition arrays exactly equal their original-route comparison inputs. All12 new raw G10 horizon fields exactly match the prior seed2101 baseline, supporting isolation of the planner change. Frozen source hashes verified. Browser checked original default, selectable modes, and saved high-Y draft restored after reload. No additional paid compute or training.

The route helper may find no route/certificate on other sites; existing explicit-failure handling remains. A helper exception is recorded as a worker error, never shown as a valid generation. Aesthetic quality, many seeds, other volume requests and broader environments remain unassessed for these alternatives.

Next: review which spatial differences are useful, then move to a bounded larger-domain cost/feasibility benchmark. Keep original R3 as reference, modes experimental, model unchanged. Avoid introducing another training objective until a concrete failure or missing capability warrants it.

Prior milestone ../G11-R3-Revisit-2026-10-04/RESUME.md; route prototype ../G11-R3-Route-Options-2026-10-04/REVIEW.md. Studio8016 and MG7 unchanged. This folder copies prior runs/drafts and preserves original evidence. Repository synchronization and off-device backup pending. No Drive, push, publication or paid training. Verified archive is same-disk and covers only files present at closure; later user runs/drafts require a later archive.

Resume by reading this file and transfer/protocol.json/result.json. Server runs with the project .venv Python and this folder's server.py on loopback8017. Check for an existing process first. No active transfer job remains.
'''
(P/'RESUME.md').write_text(notes,encoding='utf-8');(P/'CHANGELOG.md').write_text('Added experimental route selector and provenance throughout generation/revisit/drafts. Completed six exposed-site transfer attempts with six passes. Original default unchanged.\n',encoding='utf-8');shutil.copyfile(__file__,P/'close.py')
files={p.relative_to(P).as_posix():sha(p) for p in P.rglob('*') if p.is_file() and p.suffix not in ['.log','.tmp'] and '__pycache__' not in p.parts};(P/'milestone-manifest.json').write_text(json.dumps(files,indent=2));files['milestone-manifest.json']=sha(P/'milestone-manifest.json');zp=P.with_suffix('.verified.zip')
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(P/n,n)
with zipfile.ZipFile(zp) as z:
 assert len(z.namelist())==len(files) and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
receipt=dict(bytes=zp.stat().st_size,sha256=sha(zp),payloads=len(files),verified=True);zp.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt)
