# G5: destination guidance — implementation and handoff

G4 solved thin-fringe growth but consumed its allowed volume within14–22steps,
leaving7of9development cases short of the far interface. G5tests whether giving
local cube proposals immediate information about that interface helps them
allocate volume more effectively. This is a hypothesis,not a demonstrated fix.

## The focused change

Two new network inputs describe each legal cube position:shortest cube-graph
distance to the destination and whether the destination is reachable. The map
is calculated once per context from legal space and interface masks. It has no
teacher target,current mass,requested volume or training-stage input. Its
destination follows the existing oppositeXinterface convention;general
arbitrary-interface support is outside this pilot.

The first layer expands from61 to63inputs,adding128trainable weights. Existing
freshly initialized core weights exactly match G4;both new inputs start with
zero weights. All dataset bytes,losses,teacher stages,start schedule,origin
firing and whole-cube budget admission are unchanged. No trained checkpoint is
used to initialize the paid pilot. G4 remains available as the reference.

The cue is global context preprocessing,so this remains a hybrid NCA. It is
information for the network,not a forced route or a new constraint family.
The hard budget can still lock in mistakes. Coverage,facade and other existing
families still need independent evaluation.

## Completed local checks

All27TRAIN cases were audited without reading development or reserved data.
An independent graph-distance implementation agrees exactly with the cue.
Analytic distances,unreachable regions,invalid interface inputs and immutable
cache reproducibility passed. Both new inputs receive finite nonzero gradients.
G4checkpoints are rejected by semantic identity.

Each TRAIN scene has a context-only path of full overlapping cubes connecting
both interfaces within the current volume ceiling. These witness paths contain
144–189voxels. They are feasibility witnesses,not model output,
not all-nine valid massing designs,and are not included as teacher routes in
the training package. Existing27targets are unchanged. The cue's cold compute
time had median28.71ms locally;this is not a GPU
training-time prediction. Bounded caching reuses geometry across volume requests.

Packaged CPU rehearsal20261004T072052Z_9051ff1a572f completed3retained updates plus two exact recovery
replays in31.860s. All23evidence payload hashes and192step
accounts verified. Both cue channels acquired nonzero weights. Training rows,
saved starting fields and firing-generator consumption match the G4CPU rehearsal.
No quality benchmark or held-out inference was performed. GPU compatibility for
the expanded model is checked inside the proposed capped job,not assumed proven.

## Ready job and user action

One Tesla T4 job,seed1201,256updates,64steps,maximum600controlledseconds. Setup,
export,download and idle time are extra. Runtime guard and12admission probes,
cue-device check,union backward and exact recovery are embedded in that job.
No separate paid preflight and no automatic retry. See PROTOCOL.md for all
effective settings and unchanged final-checkpoint evaluation gates.

After explicit approval,open NCA-G5-Destination.ipynb in Colab,upload this
folder's NCA-G5-Destination-Package.zip,set APPROVED_G5_JOB=True and run once.
Return the full evidenceZIP and receipt,including after failure. The distributed
notebook still has its approval flagFalse. No paid job was launched here.

## Persistence and limits

Source,package hashes,individual TRAIN witness arrays,checks,rehearsal evidence,
decisions and exact continuation instructions are saved locally. Earlier stages
are untouched. Repository synchronization remains pending because this session
has no granted write access to the checkout;use this folder's RESUME.json.
The verified milestone archive is on the same disk,not an off-device backup.
No Drive operation,push,publication or live-model replacement was performed.
