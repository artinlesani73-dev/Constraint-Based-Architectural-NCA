# Working agreement for the NCA next phase

Read `docs/next-phase/RESUME.md` first, then `PLAN.md`, `DECISIONS.md`, and `CHANGELOG.md` in that directory. These records must be sufficient to continue in a new session without chat history.

## User requirements

- The research target is overall building mass; future occupied voxels mean building volume, with interiors and construction left for later (D058). Do not treat them as solid construction material or require internal cavities now. Preserve meaningful exterior gaps; do not blindly fill the VA1 blue diagnostic mask. SP1/VA1 retain historical material semantics. Read MASSING_BRIEF.md before geometry/objective work; retain volumetric depth without prescribing rooms, shelters or functions.
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

## Google Drive boundary and approval rule (2026-09-23)

- The only Drive scope for this project is folder ID `1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H`, named `Constraint-Based-Architectural-NCA`, and its actual descendants: https://drive.google.com/drive/folders/1fS34Yy0-oMzSxWaYJFiPTkGgrZstgc0H . Use the ID, not a name match.
- Ask for and receive explicit user approval BEFORE EVERY Drive operation, including viewing, metadata reads, listing, searching, downloading, verifying, creating, uploading/saving, editing, renaming, copying, moving, sharing, or deleting. State the exact action and target; a batch is allowed only if the user explicitly approves its complete stated scope. An upload approval alone does not authorize a later readback/download unless included in that approval.
- Do not search or list the whole Drive, inspect other folders or account information, access outside files, follow shortcuts/links outside this folder, or move/copy files across its boundary. If scope or ancestry cannot be established from already approved evidence, stop and ask; do not inspect outside resources to resolve it. Expanding scope requires an explicit user revision of this rule.
- Approval is not standing authorization for later operations, background sync, Colab Drive writes, retries that could create duplicates, or other access methods. No automatic Drive backup/sync is authorized. Local project work remains governed by the existing local working agreement.
- Folder creation was specifically authorized and completed on 2026-09-23. No files were uploaded. Backup verification remains pending; the older proposed `NCA-Next-Phase` Drive root is superseded by this folder.
- This is an assistant operating rule, not an OAuth or server-enforced folder restriction. Do not claim the connector itself is limited to this folder.
