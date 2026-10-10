"""Freeze and archive one representation check, without model inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from nca.urban_context import summary
OUT=R/'.local-artifacts/urban-context-v1-2026-10-10'
def main():
    OUT.mkdir(exist_ok=False)
    fixture=R/'experiments/scenes/urban_context_v1/block-01.json'
    scene=json.loads(fixture.read_text());result=summary(scene)
    check=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_urban_context.py','-v'],cwd=R,capture_output=True,text=True)
    (OUT/'checks.log').write_text(check.stdout+check.stderr,encoding='utf-8')
    result.update(passed=check.returncode==0,check='test_urban_context.py',no_generation=True,
        source_sha256=hashlib.sha256((R/'nca/urban_context.py').read_bytes()).hexdigest(),
        fixture_file_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
    shutil.copyfile(fixture,OUT/'scene.json')
    text=json.dumps(result,indent=2)+'\n';(OUT/'summary.json').write_text(text,encoding='utf-8')
    (R/'experiments/reports/Urban-context-v1-2026-10-10.json').write_text(text,encoding='utf-8')
    print(text)
    if check.returncode:raise SystemExit(check.returncode)
if __name__=='__main__':main()
