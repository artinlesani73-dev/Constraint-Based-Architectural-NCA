from pathlib import Path
import json,hashlib,shutil,zipfile
P=Path('C:/Users/artin/Documents/Codex/outputs/G11-R3-Editor-2026-10-04');OLD=P.parent/'G11-R3-Studio-2026-10-04'
history=P/'implementation-history';history.mkdir(exist_ok=True)
shutil.copyfile(P/'studio.js',history/'studio-before-utf8-fix.js');shutil.copyfile(P/'identity.json',history/'identity-before-utf8-fix.json')
j=(OLD/'studio.js').read_text(encoding='utf-8').replace('seed})',"seed,...(window.editedScene?{custom_scene:window.editedScene()}: {})})")
(P/'studio.js').write_text(j,encoding='utf-8')
identity=json.loads((P/'identity.json').read_text());identity['files']['studio.js']=hashlib.sha256((P/'studio.js').read_bytes()).hexdigest();(P/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
notes='''# R3 custom-site editor — 2026-10-04

Ready at http://127.0.0.1:8015/ . Choose a generation site, enable Use edited geometry for generation, change numeric geometry, then Generate comparison. Twelve controls adjust facing X positions, roofs, Y extents and both connection Y/Z coordinates. Connection X stays attached to its facade. Wireframe preview shows building extents. This is a bounded two-facing-buildings editor on the existing32³ grid, not arbitrary geometry/CAD import. A preset obstacle is retained but not editable. No new constraint families or model changes.

Server validates the full normalized custom scene before allocating a run, including fixed grid/units/street band, dimensions, facade pairing and connection placement. Every accepted edited scene is saved exactly with request, result, source identity and existing run evidence. Invalid submissions receive explanatory errors; no generation begins. Draft edits are not retained across reloads; submitted runs are persistent. Historic studio8014 and gallery8013 are unchanged.

Verification: browser rejected West connection Z31 (out of grid). Valid edited scene changed West facade X8 to7 and West connection Z8 to9. Actual run43182d564aa54abeb7d69c1d59752450 completed, both G10 and R3 passed nine families at64/128. This single functional check is not generalization evidence. Verified exact normalized scene equality across request/scene/result, edited coordinates, all run payload hashes, and oversized-grid rejection before creating a run. Visual review found a label encoding defect; fixed UTF8 reading while preserving the earlier JS/identity under implementation-history. The scientific source and run arrays are unchanged.

Run uses the UI identity active when submitted. Final UI identity differs only for the encoding correction; prior identity and JS are preserved. The running process retains the original in-memory identity until restart, while inference code/model remain unchanged. No claim that this UI revision changes generation.

Next: improve edit/revisit workflow and meaningful form alternatives on a fixed custom site. Keep the R3 reference, exact scene/seed history and raw G10 comparison. Larger-grid scaling remains deferred. No paid training, publication, Drive or MG7 replacement.

Resume: previous ../G11-R3-Studio-2026-10-04/RESUME.md. Restart this folder's server.py with the project's .venv Python after checking8015 is not already in use. Repository synchronization and off-device backup remain pending. This same-disk verified archive includes only runs present at closure; future runs need a later archive.
'''
(P/'RESUME.md').write_text(notes,encoding='utf-8');(P/'CHANGELOG.md').write_text('Added bounded geometry editing, pre-generation validation, exact submitted-scene persistence and verified a custom run. Preserved/fixed UI text encoding. See RESUME.md for evidence and limits.\n',encoding='utf-8')
for n in ['verify_r3_editor.py','finish_r3_editor.py']:shutil.copyfile(n,P/n)
