from pathlib import Path
import json,zipfile,hashlib,shutil
BASE=Path('C:/Users/artin/Documents/Codex/outputs');OUT=BASE/'G4-Block-Training-2026-10-03-v2'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
notebook=OUT/'NCA-G4-Block.ipynb'
before=notebook.read_bytes();nb=json.loads(before)
for cell in nb['cells']:
    cell['source']=[s.replace('supplied G1 package ZIP','supplied G4 package ZIP') for s in cell['source']]
    if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')
(OUT/'notebook-before-label-fix.ipynb').write_bytes(before)
notebook.write_text(json.dumps(nb,indent=2),encoding='utf-8')
receipt=json.loads((OUT/'package-receipt.json').read_text());assert receipt['sha256'] in notebook.read_text()
(OUT/'notebook-label-correction.json').write_text(json.dumps(dict(reason='Correct inherited upload error text from G1 to G4. SHA256 and all executable logic unchanged.',previous_sha256=sha(before),current_sha256=sha(notebook.read_bytes()),package_unchanged=True),indent=2))
# Keep builder reproducible and preserve its earlier source.
builder=OUT/'build-package.py';old=builder.read_text();shutil.copyfile(builder,OUT/'build-package-before-label-fix.py')
needle=".replace('nca-g3-','nca-g4-')";assert needle in old
builder.write_text(old.replace(needle,needle+".replace('supplied G1 package ZIP','supplied G4 package ZIP')"),encoding='utf-8')
shutil.copyfile(__file__,OUT/'archive-milestone.py')
archives=[]
for folder in [BASE/'G4-Block-Training-2026-10-03',OUT]:
    files={p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='milestone-manifest.json'}
    manifest=folder/'milestone-manifest.json';manifest.write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')
    files['milestone-manifest.json']=sha(manifest.read_bytes())
    archive=folder.with_suffix('.verified.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(folder/name,name)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))==len(files)
        assert all(sha(z.read(k))==v for k,v in files.items())
    record=dict(archive=str(archive),payloads=len(files),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes()),verified=True,off_device_backup=False)
    archive.with_suffix('.receipt.json').write_text(json.dumps(record,indent=2),encoding='utf-8');archives.append(record)
print(json.dumps(archives,indent=2))
