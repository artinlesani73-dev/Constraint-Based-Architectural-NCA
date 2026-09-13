# Working agreement for the NCA next phase

Read `docs/next-phase/RESUME.md` first, then `PLAN.md`, `DECISIONS.md`, and `CHANGELOG.md` in that directory. These records must be sufficient to continue in a new session without chat history.

## User requirements

- Preserve the nine existing constraint families. Improve their meaning and implementation; do not introduce additional constraint categories without a new user decision.
- Document every meaningful change, experiment, failure, and decision. Record limitations as well as successful results.
- Never delete or overwrite historical experiment evidence to make a run look successful. New attempts get new run IDs and link to the previous attempt.
- Keep `NCA-Next-Phase-Report` files local and Git-ignored. The report is guidance; tracked plan/decision documents are the implementation record.
- Preserve the original notebook, checkpoint, evaluation, and history as historical evidence. Any new training path must be versioned and must not silently rewrite those files.
- Before expensive training, establish a tested checkpoint/recovery path, frozen evaluation scenes, and an explicit compute allowance.
- Use Google Drive plus a local archive for training artifacts. Verify copied hashes. A second directory on the same disk is not an off-device backup.
- Do not start paid training, publish, or push repository changes merely because preparation is complete. Record the concrete ready-to-run configuration and required user action.

## Implementation discipline

- Work in small, reviewable changes and run relevant tests. Avoid simultaneous changes to model architecture, objectives and scene distribution before a corrected baseline exists.
- Record effective configuration, model/code hashes, random seeds, environment, metrics definitions, and individual-scene results for experiments. Register failures and interruptions too.
- Keep large artifacts in `.local-artifacts/` or the configured artifact root; commit small experiment summaries and decisions under `experiments/` and `docs/next-phase/`.
- Never describe a synthetic regression test as a trained-model benchmark. Distinguish geometric support from mechanical safety, and corrected metrics from historical scores.
- At each meaningful milestone and before ending a session, update `RESUME.md` with completed work, exact next steps, commands, open questions, and artifact/run locations. If interrupted mid-command, inspect files and running processes before retrying; do not assume completion.
- Local commits may preserve completed, verified milestones; do not stage the user's unrelated files or ignored reports. Do not push without user authorization.

## Start here

`python scripts/verify_foundation.py` runs and archives the full suite; use the documented isolated environment with NumPy, PyTorch, FastAPI and the test client. Dependency-light checks can run with `python -m unittest discover -s tests -p test_evaluation.py -v` and `python -m unittest discover -s tests -p test_experiments.py -v`. See `docs/next-phase/RESUME.md` for verified commands.
