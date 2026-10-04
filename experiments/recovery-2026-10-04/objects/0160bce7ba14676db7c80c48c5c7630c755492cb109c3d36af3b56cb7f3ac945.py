from pathlib import Path
cwd=Path(__file__).parent
for source,target in [('audit_g11_results.py','audit_g11_r2_results.py'),('render_g11.py','render_g11_r2.py')]:
 s=(cwd/source).read_text().replace('G11-R1-Prototype-2026-10-04-v2','G11-R2-Packing-2026-10-04').replace('G11-R1','G11-R2')
 (cwd/target).write_text(s,encoding='utf-8')

