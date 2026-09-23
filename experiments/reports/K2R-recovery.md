# K2 actual-loop recovery gate

Run `20260923T102414Z_eaec7bd1510e`; source33f6858. All seven exact comparisons pass independently.
Three logical updates / eight executions in four fresh processes: whole3,
prefix1, resumed2 and repeated-resume2. Uses the ACTUAL K2 16-step training loop,
Adam1e-4 with constant learning rate, mass_3/seed0, and the first three scenes in
the complete frozen17-scene order. Full checkpoint state, trace values and saved
material/raw arrays agree exactly. No claim of trained quality from this gate.

First three scenes: legacy-easy-seed-002, legacy-easy-seed-010, legacy-easy-seed-003.

Source, proposal, config, input/scene hashes, complete order and RNG are recorded.
This verifies CPU completed-update continuation after orderly process exit;
CUDA, AMP, sample pools and abrupt writes remain outside this evidence.
The separate K2 study is required for coefficient outcomes.
