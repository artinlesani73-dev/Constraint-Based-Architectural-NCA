from pathlib import Path
import json,math
OLD=Path('C:/Users/artin/Documents/Codex/outputs/G11-R1-Prototype-2026-10-04-v2')
OUT=Path('C:/Users/artin/Documents/Codex/outputs/G11-R2-Packing-2026-10-04')
records=[]
for p in sorted((OLD/'cases').iterdir()):
 wr=json.loads((p/'witness.json').read_text());trace=json.loads((p/'hybrid-trace.json').read_text())
 C=wr['C'];K=max(9,math.ceil((C-27)/63));m0=1
 for t in trace:
  envelope=min(C,27+(t['step']-1)*K)
  assert t['mass']<=envelope
  assert (C if m0==1 else min(C,m0+K))==t['cap']
  m0=t['mass']
 records.append(dict(case=p.name,K=K,C=C,ceiling64=min(C,27+63*K),mass64=trace[63]['mass'],hypothetical_available64=min(C,27+63*K)-trace[63]['mass']))
assert all(x['ceiling64']==x['C'] for x in records)
(OUT/'cumulative-allowance-analysis.json').write_text(json.dumps(dict(cases=records,all_original_trajectories_within_proposed_envelope=True,proposed_cap64_equals_original_global_cap=True,rollouts_under_new_rule=0,interpretation='Upper-envelope arithmetic only. Does not establish usable proposals, shape quality, timing or stability. Proposal relaxes instantaneous quota while retaining cumulative budget and global cap.'),indent=2),encoding='utf-8')
import shutil;shutil.copyfile(__file__,OUT/'allowance-analysis.py')
print('45 saved trajectories checked; no new rollout.')

