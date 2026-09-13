# Experiment evidence and recovery

## Minimum record

Every attempt needs a unique ID, purpose/hypothesis, code commit and dirty-source snapshot, effective config, explicit random seeds, environment/device, model/checkpoint hash, scene-set version/hash, rollout profile, metric/threshold versions, per-scene outputs, aggregate summaries, and interpretation. Declare missing fields; never infer provenance that was not recorded. GPU timing must include device synchronization and distinguish load/preprocessing/growth/evaluation/rendering.

`scripts/experiment.py` provides the storage mechanism, not the future training loop. `scripts/verify_foundation.py` records the local regression suite and a real checkpoint smoke test. It does not train a model or measure architectural quality.

## Layout

```
experiments/records/<run-id>.json       # Small reviewed summary, tracked
.local-artifacts/runs/<run-id>/
  run.json                            # Immutable initial configuration/provenance
  events/<timestamp_uuid>.json         # Individual append-only events
  artifacts/<uuid>_<original-name>     # Copied payloads; hashes in artifact events
  result.json                         # One immutable completed/failed/interrupted result
```

Use `null` plus a reason for unavailable metrics; NaN/infinity are rejected. Keep scene IDs and failure flags in per-scene JSON/CSV, not just one average. Attach continuous occupancy where feasible, thresholded geometry, representative views, and the full checkpoint. Checkpoint files should be finalized before attaching them. Archive stdout/stderr and exception traces. Do not delete failed runs or overwrite their final result when retrying.

## Commands (from the repository root)

Use the project's `.venv/Scripts/python.exe` on Windows or the active Colab Python.

```text
python scripts/verify_foundation.py
python scripts/experiment.py create --name E0-replay --kind inference --seed 17 --config path/to/config.json --snapshot
python scripts/experiment.py attach RUN_ID path/to/results.json --role per_scene_results
python scripts/experiment.py attach RUN_ID path/to/model.pth --role checkpoint
python scripts/experiment.py event RUN_ID --kind observation --message "Explain a finding or interruption"
python scripts/experiment.py finish RUN_ID --status completed --metrics path/to/metrics.json --interpretation "What this does and does not establish"
python scripts/experiment.py verify RUN_ID
python scripts/experiment.py mirror RUN_ID path/to/backup-runs
```

Commands containing `RUN_ID` or `path/to/...` are templates; replace them with actual recorded paths. `--root PATH` before the subcommand selects a different artifact root. `create --parent OLD_ID ...` links a new attempt to an earlier run in that store.

## Before expensive training

The next trainer must checkpoint model weights, optimizer, scheduler/scaler if used, effective config, step, Python/NumPy/Torch CPU and CUDA RNG states, pool states paired with their scene IDs, sampling/curriculum state and the frozen validation manifest. Save uniquely named checkpoints through a temporary file, flush, validate, then publish. Retain periodic/recovery checkpoints and all selected experimental outputs; any later retention policy needs an explicit decision.

Test continuation after interruption against an uninterrupted short reference run. Save at wall-clock intervals as well as training steps. Train on Colab's local disk and mirror complete artifacts to Drive at checkpoints. Afterward download a verified copy to the local archive. Drive can lag or disconnect; inspect a mirror receipt and perform a restore check before treating it as a backup.

## Recovery protocol

1. Read the resume document and identify the last run ID.
2. Verify artifact hashes. Inspect unregistered files or `.tmp` payloads from interruptions; do not silently discard them.
3. If a live attempt was interrupted, record an interruption event. If it already has a final outcome, create a new attempt with `--parent`.
4. Restore the latest verified checkpoint and exact configuration; link its artifact hash in the new run.
5. Mirror and verify the resumed result. A local test of mirroring does not establish that Google Drive has been configured.

Run records are append-only under the provided API, not tamper-proof. The archive assumes one coordinator per run. Copy operations verify content hashes and refuse conflicting overwrites. Transfer temporaries are retained after interruption; a later transfer retries into a fresh temporary path. There is no automatic cloud account access, scheduled continuation, or paid training in these tools.
