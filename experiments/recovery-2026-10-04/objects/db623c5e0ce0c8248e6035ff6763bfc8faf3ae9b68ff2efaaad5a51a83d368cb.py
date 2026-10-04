from pathlib import Path
import json,hashlib,shutil,zipfile
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Revisit-2026-10-04');OLD=P.parent/'G11-R3-Editor-2026-10-04'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original='43182d564aa54abeb7d69c1d59752450'
for p in (OLD/'runs'/original).rglob('*'):
 if p.is_file():assert sha(p)==sha(P/'runs'/original/p.relative_to(OLD/'runs'/original))
new=[p for p in (P/'runs').iterdir() if p.name!=original];assert len(new)==1;run=new[0];q=json.loads((run/'request.json').read_text());state=json.loads((run/'status.json').read_text());assert state['state'] in ['completed','failed_checks']
assert q['parent_run']==original and q['seed']==2102 and q['custom_scene']['buildings'][0]['z'][1]==26 and q['custom_scene']['buildings'][0]['x'][1]==7 and q['custom_scene']['entrances'][0]['z']==9
drafts=[json.loads(p.read_text()) for p in (P/'drafts').glob('*.json')];assert len(drafts)==2 and any(d['draft']['values']['edit1']=='' for d in drafts) and any(d['draft']['values']['edit1']=='26' for d in drafts)
manifest=json.loads((run/'manifest.json').read_text());assert all(sha(run/n)==h for n,h in manifest.items())
identity=json.loads((P/'identity.json').read_text());assert all(sha(P/n)==h for n,h in identity['files'].items())
evidence=dict(original_run_unchanged=True,parent_link_verified=True,edited_geometry_seed_preserved=True,immutable_draft_versions=2,incomplete_field_preserved=True,run_hashes_verified=True,frozen_source_hashes_verified=True,new_run=run.name,status=state)
(P/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
notes=f'''# R3 revisit and draft workflow — 2026-10-04

Ready: http://127.0.0.1:8016/ . Select a saved run, then Edit selected run to load its exact scene, volume and seed into the editor. New generations preserve parent_run lineage. Changing to a preset clears the parent link. Outputs and original runs are immutable; edits only affect a new request.

Save draft locally persists a new UUID version on disk, including blank/incomplete numeric fields. Restore draft loads it after a page reload. Draft validation is deferred to generation, which retains the full geometry checks. Saving is explicit, not automatic: save before reloading/switching sites. Draft endpoints inherit loopback Host/Origin/header restrictions and16KiB body limit. All draft versions are retained; no deletion feature. The UI displays saved-draft timestamps explicitly in UTC.

Verified through the browser: loaded parent{original}, retained facade X7 and connection Z9, saved a blank roof field, reloaded/restored that blank and its validation message, then changed roof to26 and seed to2102, saved a second draft and generated child{run.name}. Child status:{state['state']}. Exact parent request link and edited values verified on disk. All original run files match their prior hashes, new run manifest matches, and frozen source hashes match. No model/training/evaluator changes. This is workflow validation, not a generalization result. Screenshot reviewed; mobile and concurrent multi-tab editing unassessed.

Preservation: prior milestone ../G11-R3-Editor-2026-10-04/RESUME.md remains unchanged at8015. The new studio copies the earlier run so it can be reopened, preserving all bytes. Check verification.json. New runs/drafts after archive creation require a new archive; same-disk ZIP is not off-device backup. Repository synchronization pending. No Drive access, paid training, publication, push or MG7 promotion.

Next: meaningful form alternatives on one fixed custom site, retaining saved lineage and the R3 reference. Avoid conflating changing geometry/seed with an improvement in the scientific method. Start with a bounded, versioned diversity change and compare shape differences and all nine families.

Restart: run this folder's server.py using the project's .venv Python after checking8016 is not already serving. Each job is local CPU and serialized. Restart marks incomplete jobs interrupted; it does not resume mid-rollout. Exact source identity and model are bundled.
'''
(P/'RESUME.md').write_text(notes,encoding='utf-8');(P/'CHANGELOG.md').write_text('Added edit-from-run, parent lineage, immutable local draft versions and restoration of unfinished inputs. Verified one linked generation and original evidence preservation. No scientific changes.\n',encoding='utf-8');shutil.copyfile(__file__,P/'close.py')
files={p.relative_to(P).as_posix():sha(p) for p in P.rglob('*') if p.is_file() and p.suffix not in ['.tmp','.log'] and '__pycache__' not in p.parts}
(P/'milestone-manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8');files['milestone-manifest.json']=sha(P/'milestone-manifest.json')
zp=P.with_suffix('.verified.zip')
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(P/n,n)
with zipfile.ZipFile(zp) as z:
 assert len(z.namelist())==len(files)
 assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
receipt=dict(bytes=zp.stat().st_size,sha256=sha(zp),payloads=len(files),verified=True);zp.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt)
