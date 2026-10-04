from pathlib import Path
import sys,json,hashlib,shutil,zipfile,math
BASE=Path('C:/Users/artin/Documents/Codex/outputs')
OUT=BASE/'G9-Access-Objective-Design-2026-10-04-v2'
sys.path.insert(0,str(OUT/'source'))
shutil.copyfile(Path(__file__).with_name('access_ranking_v1.py'),OUT/'source/access_ranking_v1.py')
import numpy as np
import torch
from access_ranking_v1 import access_ranking, ranked_loss,teacher_graph,priority
from nca.paced_generation import eligibility,full_origins,block_loss
from nca.budget_reference import budget
torch.set_num_threads(2)
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
sha=lambda b:hashlib.sha256(b).hexdigest()
rows=json.loads((OUT/'source/data.json').read_text())['rows']
checks=[];capacity=[]
for row in rows:
 with np.load(OUT/'source'/row['arrays'],allow_pickle=False) as a:c=a['condition'];target=a['target'].astype(bool)
 with np.load(OUT/'oracle'/(row['id']+'.npz'),allow_pickle=False) as a:states=a['states']
 graph=teacher_graph(target,c[5].astype(bool),split=row['split']);allowed=c[0].astype(bool)
 D=int(allowed.sum());B,C=budget(D,float(c[6,0,0,0]),3)
 for i,field in enumerate(states):
  p,phase=priority(field,allowed,graph);e,seed=eligibility(field,full_origins(allowed))
  tensor=lambda x:torch.from_numpy(x)[None,None]
  # Nonuniform finite logits exercise all-pair algebra and score-focused gradients.
  z=torch.linspace(-2,2,e.size).reshape(1,1,*e.shape).requires_grad_()
  mask=tensor(e);orig=tensor(full_origins(target));pos=tensor(p)
  rank=access_ranking(z,mask,orig,pos,phase)
  grad=torch.autograd.grad(rank,z)[0]
  A=mask&orig&pos;O=mask&orig&~pos;active=bool(A.any() and O.any() and phase not in ('connected','no_teacher_route'))
  if active:
   assert (grad[A]<0).all() and (grad[O]>0).all() and (grad[~(A|O)]==0).all()
   direct=torch.log1p(torch.exp(1+z[O][:,None]-z[A][None,:]).mean())
   assert torch.allclose(rank,direct,atol=1e-6,rtol=1e-6)
  else:assert rank==0 and (grad==0).all()
  if i==len(states)-1:
   args=(z,tensor(field),mask,tensor(target.astype(np.float32)),orig,seed,D,B,C)
   b=block_loss(*args);r=ranked_loss(*args,pos,phase)
   assert all(torch.equal(x,y) for x,y in zip(b,r[:4])) and r[4]==0
  checks.append(dict(case=row['id'],stage=i,phase=phase,ranking_active=active,loss=float(rank.detach())))
 k=max(9,math.ceil((C-27)/63));mass=int(states[-1].sum());steps=len(states)-1
 optimistic=min(C,mass+(64-steps)*k)
 capacity.append(dict(case=row['id'],B=B,C=C,quota=k,connection_step=steps,connection_mass=mass,optimistic_mass_at64=optimistic,short_of_requested_B=max(0,B-optimistic),optimistic_error_pp=max(0,float(c[6,0,0,0])-optimistic/D)*100))
# Under no firing / absent comparison group, rank is exactly inactive.
z=torch.tensor([1000.,-1000.,3.,-2.]).reshape(1,1,1,1,4).requires_grad_()
e=torch.ones_like(z,dtype=torch.bool);p=e.clone();p[...,2:]=False
loss=access_ranking(z,e,e,p,'advance_access');g=torch.autograd.grad(loss,z)[0]
assert torch.isfinite(loss) and torch.isfinite(g).all()
assert access_ranking(z,~e,e,p,'advance_access')==0
assert access_ranking(z,p,e,p,'advance_access')==0
save('ranking-checks.json',checks);save('capacity-analysis.json',capacity)
summary=dict(oracle_states_checked=len(checks),active_ranking_checks=sum(c['ranking_active'] for c in checks),connected_baseline_exact=45,explicit_pairwise_formula_verified=True,extreme_logits_finite=True,no_firing_and_no_other_group_zero=True,hard_gating_oracle_cases_below_requested_B=sum(c['short_of_requested_B']>0 for c in capacity),hard_gating_max_shortfall=max(c['short_of_requested_B'] for c in capacity),optimizer_updates=0,trained_model_inference=0,heldout_access=False)
save('ranking-result.json',summary)
report=f'''# G9: access-priority objective proposal

## Decision

Prepare a **ranking-only loss supplement** to G8. Retain the original teacher
membership, volume and band losses, architecture, all45 TRAIN examples, start
schedule, stochastic firing, budget, quota, and detached cube admission.
This is an implemented and locally checked loss prototype, not an integrated
trainer, GPU-ready package or trained candidate. G8 remains the evaluated
experimental reference; MG7 remains live.

## Evidence and interpretation

G8 passed32/33 old and8/12 new cases at both64/128 steps. All45 passed volume and
stability gates. The five access failures exhausted their mass allowance by128.
G6's earlier objective audit established that static teacher membership rewards
both advancing and other eventual teacher cubes. G5's extra destination inputs
did not solve the earlier system's failures. Together these support testing
relative growth priority, without claiming it is the sole cause or guaranteed fix.

The new graph/label calculations used only the existing45 TRAIN examples.
No new held-out labels, checkpoint selection, model inference or optimizer
updates occurred. Previously evaluated G8 scenes are now regression evidence;
they must never be called fresh in a later report.

## Rejected design: postpone all other growth

An initial temporal-BCE prototype suppresses non-advancing cubes and suspends
volume losses until both interfaces are touched. Its deterministic all-fire
teacher oracle connects all45 TRAIN cases in14–22 steps, with at least338 cells
of global allowance remaining. However, with the unchanged per-step quota,
{summary['hard_gating_oracle_cases_below_requested_B']}/45 of these oracle trajectories cannot reach requested B by64 even if
every remaining step uses its full quota (maximum shortfall{summary['hard_gating_max_shortfall']} cells).
This is an upper-bound calculation for those particular oracle trajectories,
not proof that every possible gated policy fails or that every shortfall breaks
the frozen tolerance. Firing randomness can add delays. Reject hard postponement
for the next controlled experiment; preserve its code and diagnostic fields.

## Exact proposed training objective

For each TRAIN target, form its full3x3x3 cube-origin graph with six-neighbour
origin moves. Multi-source BFS starts at teacher cubes touching the east
interface. This teacher-derived distance is supervision only: it never enters
model inputs, admission scores, inference or postprocessing.

From a seed, advancing positives are eligible teacher origins with the minimum
finite distance. From a later state, they are eligible teacher origins with
distance strictly below the best finite distance among existing full origins.
Stop this auxiliary supervision once the connected field touches both west
and east interfaces. The current generator guarantees a connected field; this
contact test alone would not establish access for a disconnected generator.
If no teacher-route progress is available, record the fallback and use G8 loss.
This does not solve off-teacher recovery; any future run must disclose fallback
counts, including late capacity-saturated states, separately.

Intersect both groups with the actual firing mask. Let A be advancing teacher
origins and O the other eligible fired teacher origins. Add:

    Lrank = log(1 + mean over a in A,b in O of exp(1 + z[b] - z[a]))
    Lnew = LG8 + 1.0 * Lrank

Use log-sum-exp and softplus to compute this stably without constructing the
pair matrix. Margin1 and weight1 are fixed design choices, not tuned results.
With either group empty, after connection, or on no-route fallback, Lrank is
exactly zero. Every original positive remains a positive in the G8 BCE.
No volume term is disabled. This changes relative score gradients without
forbidding simultaneous mass growth. Ranking affects the gradient, not the
hard sort/admission algorithm. It does not guarantee a connection or timely fill.

## Local checks and what they establish

All427 saved G8 training starts reconstructed with matching original hashes:
214 seed-access,153 advance-access,60 already connected; zero no-route starts.
The hard-gating diagnostic had678 gradient-direction checks and45 exact
post-connection baseline-loss checks. The selected ranking prototype checked
{summary['oracle_states_checked']} stored oracle states; {summary['active_ranking_checks']} have both comparison groups.
Explicit pairwise calculations match the efficient formula; its gradients
raise advancing and lower other teacher logits. No-group/no-firing behavior,
extreme-logit finiteness, and45 exact post-connection baseline delegations pass.
These are mathematical and supervision-feasibility checks, not learned quality.
The oracle ignores firing randomness and is not a deployment fallback.

## Next implementation, then one bounded Colab comparison

1. Integrate this versioned loss into a separate G9 training session. Keep
   inference byte-equivalent for fixed weights, seed and context. Cache TRAIN
   graphs deterministically; log auxiliary loss, phase counts, empty-group and
   no-route events without changing random-number consumption.
2. One consolidated local check: unchanged initial weights/start/firing sequence,
   inference parity, finite backward and exact checkpoint recovery at a seed
   and partial-start update. No new local model-quality sweep.
3. Freeze one fresh seed1201 paired427-update,64-step T4 proposal, at most600
   controlled seconds on the admitted runtime, with no automatic retry. This
   is a proposed allowance; no paid run is authorized or launched by this file.
   Preserve the full evidence and final427 checkpoint; no best-checkpoint search.
4. Freeze unchanged gates and all45 existing cases as regression before training.
   Freeze a genuinely new reserved split before its first inference. Compare G8
   and G9 on that same fresh split once, with identical firing and64/128 horizons,
   to avoid comparing success rates from different cohorts. Exclude held-out
   targets/route labels from both packages. A fresh split is still synthetic,
   and one seed cannot establish broad reliability.

Acceptance is the same nine-family conjunction plus volume/stability gates.
Inspect raw results, geometry and regressions even if aggregate scores improve.
Do not weaken gates or add a hidden procedural bridge. If access/volume still
trade off, diagnose this single comparison before considering removals or other
architecture changes. Greater grids and production deployment remain later work.

## Preservation and resume

All changes are local under this folder; checkout synchronization remains pending.
The first audit failed because its diagnostic passed a Boolean target to the
baseline convolution loss. That attempt is retained separately; the corrected
audit is v2 and uses float targets. This was an audit implementation error,
not a model/training failure. No historical evidence was overwritten.
Read RESUME.json here next. The verified archive is same-disk preservation,
not off-device backup. No Drive operation, paid compute, publication or live
replacement occurred. The original next-phase report remains untouched.
'''
with (OUT/'PROPOSAL.md').open('x',encoding='utf-8') as f:f.write(report)
save('RESUME.json',dict(status='G9 TRAIN-only access ranking proposal implemented and checked; trainer integration pending',previous=str(BASE/'G8-Final-Review-2026-10-04-v2/RESUME.json'),selected='access_ranking_v1.py; additive margin1 weight1; retain all G8 loss terms',rejected='hard pre-connection postponement: conflicts with fixed64-step fill timing on oracle trajectories',next='Integrate separate G9 session; one consolidated inference parity/backward/recovery check; freeze package and fresh paired G8/G9 reserved protocol before requesting one bounded Colab run',paid_run_authorized=False,repository_sync_pending=True,live_model='MG7 unchanged',off_device_backup_pending=True))
save('project-record.json',dict(event='G9 access-objective design',results=summary,prior_evidence=str(REVIEW) if 'REVIEW' in globals() else str(BASE/'G8-Final-Review-2026-10-04-v2'),selected_objective='G8 loss plus TRAIN-only access pair ranking',architecture_changed=False,inference_changed=False,optimizer_updates=0,drive_operations=0))
shutil.copyfile(__file__,OUT/'finalize_access_design.py')
shutil.copytree(BASE/'G9-Access-Objective-Design-2026-10-04',OUT/'failed-first-audit')
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
save('milestone-manifest.json',dict(files=files));files['milestone-manifest.json']=sha((OUT/'milestone-manifest.json').read_bytes())
archive=OUT.with_suffix('.verified.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name in files:z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(files)
 assert all(sha(z.read(k))==v for k,v in files.items())
with archive.with_suffix('.receipt.json').open('x') as f:json.dump(dict(archive=str(archive),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,payloads=len(files),verified=True,off_device_backup=False),f,indent=2)
print(json.dumps(summary,indent=2));print(archive)
