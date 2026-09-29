# CGR3 readiness

## D097 — CGR3 implemented and packaged; GPU approval pending — 2026-09-29

Separate curriculum_repair session implements D096 with CGR1 model/loss unchanged.
Checkpoint visit counters are checked against full trace; each training update
exports its exact starting occupancy and SHA256. Curriculum only in step(),
original tensors/evaluate inherited unchanged. Same-shape prior restores rejected.
Consolidated CPU test passed2.848s: first original update exactly matches CGR1,
augmented next step restores exact start/loss/state/optimizer/weights, evaluation
uses original input and does not change payload; invalid counters rejected.
Eight-update packaged CPU rehearsal20260929T070735Z_0580de3e2476 completed27.437s;
41payload hashes and8start hashes verified, checkpoint visits sum8,cleanup0active.
Rehearsal first visits are originals; augmented recovery covered by focused test.
Engineering evidence only; no new GPU compatibility/recovery or quality claim.

Ready folder C:/Users/artin/Documents/Codex/outputs/CGR3-Curriculum/:
NCA-CGR3-Curriculum.ipynb and NCA-CGR3-Curriculum-Package.zip.
ZIP SHA256498617094d60b12c10a6455daeae3046f9f56acf26667996c3c661b3f036cb97.
Manifest79a8719b2f35b07e7acb510947723bd9ffc09ff186b3dbcbb0580126cb0fa59e.
TRAIN81,heldout0,103payloads; notebook approval gateFalse. See CGR3-readiness.json.
Next request ONE seed1201 T4 job,256updates32steps600controlledseconds; setup,
export,idle extra. Download ZIP+receipt locally,including failures. No retry,
extra seed,Drive,push or live admission. MG7 stays live. Upon returned evidence,
adapt frozen final256 reviewer for new semantics/manifest and verify start visits,
trace and arrays; retain original-input27development evaluation and all prior gates.

Local copies are same-disk archives, not off-device backup.
