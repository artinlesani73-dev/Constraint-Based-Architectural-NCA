from pathlib import Path
r=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')
p=r/'nca/connected_repair.py';s=p.read_text();s=s.replace('CGR1','CGR2').replace('connected_constructive_repair_v2','bulk_constructive_repair_v1').replace("'intact_negative':1.5,'volume':.25", "'intact_negative':3.,'volume':.25,'bulk_completion':.5")
s=s.replace('def step_loss(logits,m,eligible,target,intact):', '''def bulk_completion(soft,eligible,target):
 # Full target 3-cubes only: penalize deficits without cancellation by excess
 # outside that cube. This is a local surrogate, not a connectivity guarantee.
 full=F.conv3d(target,target.new_ones((1,1,3,3,3)))==27
 active=full & (F.max_pool3d(eligible.float(),3,1)>0)
 deficit=(1-cube_mean(soft)).square()
 return deficit[active].mean() if active.any() else soft.sum()*0

def step_loss(logits,m,eligible,target,intact):''')
s=s.replace('(1.5 if intact else 1.)','(3. if intact else 1.)').replace('return front+.25*volume,front,volume','return front+.25*volume+.5*bulk_completion(soft,eligible,target),front,volume')
(r/'nca/bulk_repair.py').write_text(s,encoding='utf-8')
for src,dst in [('nca/connected_package.py','nca/bulk_package.py'),('scripts/build_connected_repair.py','scripts/build_bulk_repair.py'),('scripts/colab_connected_repair.py','scripts/colab_bulk_repair.py')]:
 s=(r/src).read_text().replace('CGR1','CGR2').replace('cgr1','cgr2').replace('connected_repair','bulk_repair').replace('connected_package','bulk_package').replace('CONNECTED_REPAIR','BULK_REPAIR').replace('connected-runs','bulk-runs').replace('connected-attempt','bulk-attempt')
 s=s.replace("if set(uploaded) != {{ARCHIVE_NAME}}:","if len(uploaded) != 1:").replace('raw = uploaded[ARCHIVE_NAME]','raw = next(iter(uploaded.values()))')
 (r/dst).write_text(s,encoding='utf-8')
s=(r/'tests/test_connected_repair.py').read_text().replace('from nca.connected_repair import','from nca.bulk_repair import').replace('CGR1','CGR2')
s=s.replace(' def test_fixed_cube_mean_value_and_gradient_parity(self):', ''' def test_bulk_direction_and_intact_suppression(self):
  from nca.bulk_repair import bulk_completion
  from nca.connected_repair import step_loss as previous
  t=torch.zeros(1,1,7,7,7);t[:,:,1:6,1:6,1:6]=1
  m=t.bool();m[:,:,3,3,3]=False;e=torch.zeros_like(m);e[:,:,3,3,3]=True;e[:,:,0,0,0]=True
  x=torch.zeros_like(t,requires_grad=True)
  soft=m.float()+e.float()*x.sigmoid();v=bulk_completion(soft,e,t)
  grad=torch.autograd.grad(v,x)[0]
  self.assertLess(float(grad[0,0,3,3,3]),0)
  self.assertEqual(float(grad[0,0,0,0,0]),0)
  e=torch.zeros_like(m);e[:,:,0,0,0]=True
  a=step_loss(x,t.bool(),e,t,True)[0];b=previous(x,t.bool(),e,t,True)[0]
  ga=torch.autograd.grad(a,x,retain_graph=True)[0];gb=torch.autograd.grad(b,x)[0]
  self.assertGreater(float(ga.sum()),float(gb.sum()))
 def test_fixed_cube_mean_value_and_gradient_parity(self):''')
s=s.replace("with self.assertRaises(ValueError):b.restore(root/'old.pt')", "with self.assertRaises(ValueError):b.restore(root/'old.pt')\n   from nca.connected_repair import ConnectedSession as PriorSession\n   with self.assertRaises(ValueError):PriorSession(root,rows,{'test':'CGR2'}).restore(p)")
(r/'tests/test_bulk_repair.py').write_text(s,encoding='utf-8')
protocol=(r/'docs/next-phase/CONNECTED_REPAIR_PROTOCOL.md').read_text().replace('CGR1','CGR2').replace('CONNECTED_REPAIR_SPEC','BULK_REPAIR_SPEC')
(r/'docs/next-phase/BULK_REPAIR_PROTOCOL.md').write_text(protocol,encoding='utf-8')
(r/'docs/next-phase/BULK_REPAIR_SPEC.md').write_text('''# CGR2: bulk completion and intact stopping

D093, 2026-09-28. One objective revision, not an architecture change.
Preserve CGR1 v2 code, weights and all evidence. Separate bulk_repair module,
semantic identity bulk_constructive_repair_v1; reject CGR1 checkpoint restores.

Keep CGR1 architecture, initialization seed1201, TRAIN81 distribution, Adam.001,
256 updates,32 steps, stochastic firing0.5, monotonic six-face births, no teacher
at inference, detached hard decisions and no deletion. Same nine families and
building-volume semantics. No rooms or blind filling of exterior gaps.

At each step let P=M+eligible*sigmoid(logits), with eligible restricted to fired
legal empty six-face neighbors. Retain frontier positive weight0.5, damaged
negative weight1 and local absolute3-cube mean-volume loss weight0.25.
Change intact negative weight1.5 to3. Add weight0.5 times mean squared deficit
(1-mean_cube(P))^2 over valid3-cubes wholly occupied in the target and touching
eligible cells. Empty selected sets contribute differentiable zero. Full cubes
are determined by fixed convolution sum==27 on binary targets; no average-pool
backward. Use deterministic convolution. This provides direct gradients toward
completion of target bulk without offsetting deficits with excess outside that
cube. It does not differentiate connectivity, enforce a bulk path, or guarantee
all-nine validity. Target cubes are supervision only. Wrong births remain
irreversible; stronger stopping can reduce correct repairs through shared weights.
Coefficients are fixed design choices, not validated optima. This combined loss
trial cannot attribute improvement separately to its two terms.

Local checks: gradient directions, empty masks, target-free inference, attachment,
exact CPU next-update recovery and semantic restore rejection; TRAIN-only audit
at logits0 checks finite bulk gradients and loss scale. Eight-update packaged CPU
rehearsal is engineering evidence only. No validation/TEST tuning.

Frozen evaluation: final256, CPUfloat32,32steps,firing2101, same27 development rows.
Accepted occupancy only, no cleanup. Retain proposals/births/states. Compare CGR1,
NR5 and closing3. Retain original gates: all9intact IoU>=.99 and valid; damaged
valid>=17/18, medianIoU>=.9705768039313023, excess<=325,recovery>=1945,
median absolute request error<=19; zero surviving input removals. Report any
regression relative to CGR1 (.975039 overlap,245excess,1959recovery,15.5error)
even if original gates pass. No automatic live admission or new TEST evaluation.

Proposed one Colab T4 job: seed1201,256updates,600controlledseconds maximum;
setup/download/idle extra. Separate approval needed after local preparation.
No retry, extra seed, Drive access or push. Successful previous CGR1 GPU execution
does not establish CGR2 GPU compatibility or exact GPU recovery.
''',encoding='utf-8')
