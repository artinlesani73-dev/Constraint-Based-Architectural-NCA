"""Create and verify a non-overwriting local evidence package."""
import hashlib,json,shutil,zipfile
from pathlib import Path
from r3_transfer_study import ROOT,OUT,sha,save,verify

def main():
    verify()
    assert json.loads((OUT/'analysis.json').read_text())['complete']
    destination=Path('C:/Users/artin/Documents/Codex/outputs/R3-Transfer-2026-10-08')
    destination.mkdir(exist_ok=False)
    files={}
    for p in OUT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:files['evidence/'+p.relative_to(OUT).as_posix()]=p
    for name in ['r3_transfer_study.py','analyze_r3_transfer.py','render_r3_transfer.py','archive_r3_transfer.py','continue_r3_contribution.py']:
        files['scripts/'+name]=ROOT/'scripts'/name
    for name in ['R3_TRANSFER_RESULTS.md','RESUME.md','PLAN.md','DECISIONS.md','CHANGELOG.md']:
        files['docs/'+name]=ROOT/'docs/next-phase'/name
    for p in (ROOT/'experiments/reports').glob('R3-transfer-*.json'):files['reports/'+p.name]=p
    manifest={n:sha(p) for n,p in files.items()}
    archive=destination/'R3-Transfer-2026-10-08.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('payload-manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(manifest)+1
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    receipt=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,payloads=len(manifest),verified=True,backup_scope='Same disk; not off-device backup')
    save(destination/'archive-receipt.json',receipt)
    save(ROOT/'experiments/reports/R3-transfer-archive-2026-10-08.json',receipt)
    shutil.copyfile(ROOT/'docs/next-phase/R3_TRANSFER_RESULTS.md',destination/'REVIEW.md')
    for p in OUT.glob('comparison-*.png'):shutil.copyfile(p,destination/p.name)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
