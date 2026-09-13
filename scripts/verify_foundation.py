"""Record a complete foundation verification attempt and all test outcomes."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import subprocess
import sys
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.experiments import RunStore, provenance, snapshot_source, write_once


def main():
    store = RunStore(REPO / ".local-artifacts" / "runs")
    config = {"suite": "foundation_v1", "metric_version": "binary_v1",
              "training": False, "device": "cpu", "purpose": "regression verification"}
    run_id = store.create("Foundation verification", "regression", config, 0, provenance(REPO))
    directory = store.path(run_id)
    print(f"RUN_ID={run_id}", flush=True)
    snapshot = directory / "source-snapshot.zip"
    snapshot_source(REPO, snapshot)
    store.attach(run_id, snapshot, "source_snapshot")
    snapshot.unlink()
    log = directory / "verification.log"
    started = time.perf_counter()
    with log.open("x", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
        suite = unittest.defaultTestLoader.discover(str(REPO / "tests"))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        smoke = subprocess.run([sys.executable, str(REPO / "deploy" / "test_model.py")],
                               cwd=directory, capture_output=True, text=True)
        stream.write("\nSmoke launched from outside repository root:\n")
        stream.write(smoke.stdout + smoke.stderr)
    metrics = {
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "failed_cases": [{"id": test.id(), "traceback": trace}
                         for test, trace in result.failures + result.errors],
        "smoke_exit_code": smoke.returncode, "wall_seconds": time.perf_counter() - started,
        "note": "Duration includes test infrastructure; not an inference benchmark",
    }
    passed = result.wasSuccessful() and not result.skipped and smoke.returncode == 0
    store.attach(run_id, log, "test_output")
    interpretation = ("Synthetic geometry, archive integrity/recovery, real Model C execution, "
                      "and API regression checks only. No repaired-model training or quality claim.")
    store.finish(run_id, "completed" if passed else "failed", metrics, interpretation)
    summary = {
        "run_id": run_id, "config": config, "metrics": metrics,
        "status": "completed" if passed else "failed", "interpretation": interpretation,
        "artifact_location": f".local-artifacts/runs/{run_id}", "drive_backup": "pending",
        "provenance": json.loads((directory / "run.json").read_text())["provenance"],
    }
    write_once(REPO / "experiments" / "records" / (run_id + ".json"), summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
