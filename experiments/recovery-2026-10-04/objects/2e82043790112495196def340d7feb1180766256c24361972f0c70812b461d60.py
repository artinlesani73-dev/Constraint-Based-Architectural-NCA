from pathlib import Path
import json
cwd=Path(__file__).parent
source=(cwd/'g11_reservation.py').read_text()
source=source.replace('G11-R1:','G11-R3:')
source=source.replace('cap=C if seed else min(C,int(field.sum())+K)','cap=C if seed else cumulative_cap(C,K,step)')
source+="""

def cumulative_cap(C,K,step):
    if type(C) is not int or C<27 or type(K) is not int or K<1 or type(step) is not int or step<1:
        raise ValueError('Invalid cumulative allowance')
    return min(C,27+(step-1)*K)
"""
(cwd/'g11_reservation_r3.py').write_text(source,encoding='utf-8')
# R2 runner has cached raw baseline, but explicitly load R1-derived R3 adapter.
run=(cwd/'run_g11_r2.py').read_text()
run=run.replace('G11-R2-Packing-2026-10-04','G11-R3-Ledger-2026-10-04').replace("with_name('g11_reservation_r2.py')","with_name('g11_reservation_r3.py')").replace('G11-R2','G11-R3')
run=run.replace('All eligible witness cubes before learned proposals; dynamically smallest addition first in each group, original order breaks ties; witness bypass unchanged','R1 order restored: witness lexicographic then learned score; witness bypass unchanged')
run=run.replace("quota='unchanged max(9,ceil((C-27)/63))'","quota='K=max(9,ceil((C-27)/63)); CHANGED non-seed cap=min(C,27+(step-1)*K), unused allowance carried; original single-cube seed phase'")
run=run.replace("import numpy as np,torch","import numpy as np,torch\nfrom g11_reservation import cumulative_cap")
run=run.replace("save('protocol.json',protocol);","""# Pre-inference arithmetic boundary checks, including a delayed seed and saturation.
checks=[]
for C in [27,28,569,1859]:
 K=max(9,int(np.ceil((C-27)/63)))
 caps=[cumulative_cap(C,K,t) for t in range(1,129)]
 assert caps[0]==27 and caps[63]==C and caps[-1]==C
 assert all(a<=b<=C for a,b in zip(caps,caps[1:]))
 # A seed that waits until step20 still admits only one27-voxel cube.
 assert 27<=caps[19] and cumulative_cap(C,K,21)>=27
 checks.append(dict(C=C,K=K,cap64=caps[63],late_seed_step20_cap=caps[19]))
save('boundary-checks.json',dict(arithmetic_cases=checks,interpretation='Cap arithmetic and single-cube-seed premise; not a complete late-seed model replay'))
save('protocol.json',protocol);""")
(cwd/'run_g11_r3.py').write_text(run,encoding='utf-8')
for src,dst in [('audit_g11_results.py','audit_g11_r3_results.py'),('render_g11.py','render_g11_r3.py')]:
 s=(cwd/src).read_text().replace('G11-R1-Prototype-2026-10-04-v2','G11-R3-Ledger-2026-10-04').replace('G11-R1','G11-R3')
 if src.startswith('audit'):
  s=s.replace("min(wr['C'],before+K)","min(wr['C'],27+i*K)")
 (cwd/dst).write_text(s,encoding='utf-8')

