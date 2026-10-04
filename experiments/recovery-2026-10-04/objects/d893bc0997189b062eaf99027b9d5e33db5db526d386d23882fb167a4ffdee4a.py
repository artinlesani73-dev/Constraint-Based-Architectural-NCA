from pathlib import Path
import json
cwd=Path(__file__).parent
source=(cwd/'g11_reservation.py').read_text()
source=source.replace('G11-R1:','G11-R2:')
source=source.replace('for idx in ids:\n            p=np.unravel_index',"""pending=list(map(int,ids))
          while pending:
            # Dynamic smallest incremental union cost; original rank breaks ties.
            def cost(idx):
                p=np.unravel_index(idx,valid.shape)
                return 27-int(field[tuple(slice(int(v),int(v)+3) for v in p)].sum())
            position=min(range(len(pending)),key=lambda i:(cost(pending[i]),i))
            idx=pending.pop(position)
            p=np.unravel_index""")
(cwd/'g11_reservation_r2.py').write_text(source,encoding='utf-8')
run=(cwd/'run_g11_r1.py').read_text()
run=run.replace("G11-R1-Prototype-2026-10-04-v2","G11-R2-Packing-2026-10-04")
run=run.replace("with_name('g11_reservation.py')","with_name('g11_reservation_r2.py')")
run=run.replace("G11-R1","G11-R2")
run=run.replace("All eligible witness cubes before learned proposals, lexicographic order; bypass scores and firing explicitly","All eligible witness cubes before learned proposals; dynamically smallest addition first in each group, original order breaks ties; witness bypass unchanged")
old="tick=time.perf_counter();raw=model.rollout(occ,static,allowed,torch.Generator().manual_seed(2101),steps=128,capture=True);rawsec=time.perf_counter()-tick"
new="""parent=BASE/'G11-R1-Prototype-2026-10-04-v2/cases'/case
  with np.load(parent/'G10-trajectory.npz') as a:cached_births=a['births'].copy()
  with np.load(parent/'raw-terminal.npz') as a:cached_state=a['state'].copy()
  raw=dict(births=torch.from_numpy(cached_births[:,None,None]),state=torch.from_numpy(cached_state))
  rawsec=json.loads((parent/'G10-128.json').read_text())['seconds128']"""
assert old in run
run=run.replace(old,new)
run=run.replace("required='All nine witness", "baseline='Cached unchanged G10 paired outputs from R1, not re-inferred',required='All nine witness")
(cwd/'run_g11_r2.py').write_text(run,encoding='utf-8')

