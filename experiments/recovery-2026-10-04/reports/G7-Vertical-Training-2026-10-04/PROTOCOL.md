# G7: vertical training diversity — one bounded run

G6 passed 9/9 reused development requests and 10/12 first-use reserved requests.
The two failures missed a higher east connection. Every original training pair
had connection origins at z=8/8. G7 tests broader vertical training data.

Retain all 27 original TRAIN payloads byte-for-byte and add 18 TRAIN examples
from six scenes, each at 16%,24%,32% requested volume. Connection heights vary
in both directions, with paired elevations and unequal building heights.
The same anchored teacher, seed=0, 15-second generation cap, contact weight=12
and full-cube origin BFS stages are used. All18 new labels pass all9 families
and the unchanged budget band; none was filtered or rerolled.

Fresh seed1201, same G6 initialization, 61->64->8 model, loss, Adam0.001,
clip1, batch1, float32, 64 training steps, alternating seed/teacher stages,
origin firing0.5 and proposal threshold0.5. Keep 32cubed grid and 0.8m cells.
Nine existing families; occupancy means overall building volume.
Neural model and loss implementation bytes are unchanged. Global admission:
B=ceil(request*D); C=min(B+8,floor(.4*D)); K=max(9,ceil((C-27)/63)). First cube
uses C; later allowance=min(C,current_mass+K). Same K at128 review steps.
This remains a hybrid learned/algorithmic method; thickness and budget are
partly enforced. It is not architectural or structural certification.

45 uniformly shuffled rows, 256 updates, same fresh seed as G6. Mean visits
fall from256/27 to256/45. This is a fixed-compute data-diversity intervention,
not equal per-example exposure or a multi-seed causal result. Do not alter
updates to rescue poor results or warm-start from the G6 checkpoint.

ONE Tesla T4 job, at most600 controlled seconds, including device probes,
256 retained updates, exact recovery replays at2/3, and checkpoint/evidence
writes. Setup/upload/export/download/idle time is extra. Expected runtime:
Python3.13.15, Torch2.11.0+cu130, NumPy2.1.3, CUDA13.0, cuDNN92700.
Stop on mismatch, recovery/probe failure, nonfinite values, or GPU reserved
memory above80%. Same-runtime completed-update recovery only. No automatic
retry or extension. Always download the full evidence ZIP and receipt.

Frozen review: final256 only, CPUfloat32, firing2101, scene-defined single
seed, horizons64/128, unchanged thresholds/quota. First report 21 legacy
regression requests (9 reused development +12 consumed G6 reserved), separately
from12 newly frozen G7 reserved requests. Require21/21 regression and12/12 fresh
all-nine passes at both horizons, median absolute fraction error<=.02,
maximum<=.04 in each cohort/horizon, and each mass change<=5%. Report every
failure; do not select checkpoints or tune from these evaluations. No fresh
reserved teachers or inference have been generated in preparation. All fresh
reserved scene/request specifications remain local, outside this TRAIN package.
These are related synthetic scenes, not external validation. Passing does not
automatically replace MG7; visual review and a separate integration decision follow.

Keep APPROVED_G7_JOB=False until the user approves this exact paid allowance.
No Drive access, paid retry, extra seed, push, publication or live promotion.
