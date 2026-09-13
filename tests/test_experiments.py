from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nca.experiments import RunStore, digest, read_json, write_once


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = RunStore(self.root / "runs")
        self.run = self.store.create("regression", "synthetic", {"metric_version": "binary_v1"}, 17)

    def artifact(self):
        path = self.root / "checkpoint.bin"
        path.write_bytes(b"a test checkpoint, not a trained model")
        return self.store.attach(self.run, path, "checkpoint")

    def test_artifact_copy_survives_source_change(self):
        record = self.artifact()
        (self.root / "checkpoint.bin").write_bytes(b"changed")
        self.assertEqual(self.store.verify(self.run), [])
        self.assertEqual(digest(self.store.path(self.run) / record["path"]), record["sha256"])

    def test_tampered_or_missing_artifacts_are_detected(self):
        record = self.artifact()
        path = self.store.path(self.run) / record["path"]
        path.write_bytes(b"corrupted")
        self.assertTrue(self.store.verify(self.run))
        with self.assertRaises(ValueError):
            self.store.finish(self.run, "completed", {}, "must fail")

    def test_final_results_cannot_be_overwritten(self):
        self.store.finish(self.run, "failed", {"loss": None}, "Record the failure")
        with self.assertRaises(FileExistsError):
            self.store.finish(self.run, "completed", {}, "Do not replace a failure")
        with self.assertRaises(ValueError):
            self.artifact()
        child = self.store.create("retry", "synthetic", {}, 17, parent_run=self.run)
        self.assertEqual(read_json(self.store.path(child) / "run.json")["parent_run"], self.run)

    def test_mirror_verifies_and_refuses_conflicting_overwrites(self):
        record = self.artifact()
        backup = self.root / "backup"
        receipt = self.store.mirror(self.run, backup)
        self.assertTrue(receipt["files"])
        self.store.mirror(self.run, backup)
        (backup / self.run / record["path"]).write_bytes(b"conflict")
        with self.assertRaises(ValueError):
            self.store.mirror(self.run, backup)

    def test_ids_and_json_values_are_validated(self):
        with self.assertRaises(ValueError):
            self.store.path("../elsewhere")
        with self.assertRaises(ValueError):
            write_once(self.root / "bad.json", {"loss": float("nan")})
        self.assertFalse((self.root / "bad.json").exists())
        with self.assertRaises(ValueError):
            self.store.create("bad", "training", {}, -1)

    def test_interrupted_backup_can_retry_without_partial_published_files(self):
        self.artifact()
        backup = self.root / "interrupted-backup"

        def interrupt(src, dst):
            dst.write(src.read(3))
            raise OSError("Simulated transfer interruption")

        with patch("nca.experiments.shutil.copyfileobj", side_effect=interrupt):
            with self.assertRaises(OSError):
                self.store.mirror(self.run, backup)
        self.assertTrue(list(backup.rglob("*.tmp")))
        receipt = self.store.mirror(self.run, backup)
        self.assertTrue(receipt["files"])
        for item in receipt["files"]:
            self.assertEqual(digest(backup / self.run / item["path"]), item["sha256"])


if __name__ == "__main__":
    unittest.main()
