"""Append-only run evidence, verified artifact copies, and resumable mirroring.

This is a filesystem archive, not a distributed database. Use one coordinator per
run. Never point artifact_root at a directory inside a source snapshot.
"""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import uuid
import zipfile

SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_once(path, data):
    """Publish fully flushed JSON without replacing an existing record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # An exclusive hard link publishes the complete inode atomically and
        # refuses overwrite. On Drive/FUSE without hard links use exclusive copy.
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        except OSError:
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def provenance(repo):
    repo = Path(repo).resolve()
    git = ["git", "-c", f"safe.directory={repo.as_posix()}"]

    def call(*args):
        return subprocess.check_output(git + list(args), cwd=repo).decode().strip()

    versions = {}
    for name in ("torch", "numpy", "fastapi", "pydantic", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "commit": call("rev-parse", "HEAD"),
        "branch": call("branch", "--show-current"),
        "working_tree_status": call("status", "--porcelain"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def snapshot_source(repo, target):
    """Archive tracked and nonignored untracked files, including dirty contents."""
    repo = Path(repo).resolve()
    names = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "ls-files", "-z",
         "--cached", "--others", "--exclude-standard"], cwd=repo
    ).decode().split("\0")
    manifest = []
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(filter(None, names))):
            path = repo / name
            if path.is_symlink():
                raise ValueError(f"Source snapshots require regular files: {name}")
            if not path.is_file():
                continue  # Tracked deletion is recorded in working_tree_status.
            raw = path.read_bytes()
            archive.writestr(name, raw)
            manifest.append({"path": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()})
        archive.writestr("_snapshot_manifest.json", json.dumps(manifest, indent=2))
    return manifest


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid run ID")
        return self.root / run_id

    def create(self, name, kind, config, seed, provenance_data=None, parent_run=None):
        if not name.strip() or not kind.strip():
            raise ValueError("Run name and kind are required")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Use an explicit nonnegative integer random seed")
        if parent_run is not None:
            read_json(self.path(parent_run) / "run.json")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
        directory = self.path(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        write_once(directory / "run.json", {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "name": name,
            "kind": kind, "config": config, "seed": seed, "created_utc": utc_now(),
            "provenance": provenance_data or {}, "parent_run": parent_run,
        })
        self.event(run_id, "started", "Run created; no performance result implied")
        return run_id

    def event(self, run_id, kind, message, **details):
        directory = self.path(run_id)
        read_json(directory / "run.json")
        filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex + ".json"
        record = {"schema_version": SCHEMA_VERSION, "created_utc": utc_now(),
                  "kind": kind, "message": message, "details": details}
        write_once(directory / "events" / filename, record)
        return filename

    def attach(self, run_id, source, role):
        directory = self.path(run_id)
        read_json(directory / "run.json")
        if (directory / "result.json").exists():
            raise ValueError("Finalized run: create a linked attempt to add new artifacts")
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError("Artifact must be an existing regular file")
        target = directory / "artifacts" / (uuid.uuid4().hex + "_" + source.name)
        target.parent.mkdir(exist_ok=True)
        before = source.stat()
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Source changed during copying; retained orphan copy for inspection")
        record = {"path": target.relative_to(directory).as_posix(), "role": role,
                  "bytes": target.stat().st_size, "sha256": digest(target)}
        self.event(run_id, "artifact", "Artifact copied and hashed", **record)
        return record

    def finish(self, run_id, status, metrics, interpretation):
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("Status must be completed, failed, or interrupted")
        problems = self.verify(run_id)
        if problems:
            raise ValueError(f"Artifact verification failed: {problems}")
        write_once(self.path(run_id) / "result.json", {
            "schema_version": SCHEMA_VERSION, "status": status, "metrics": metrics,
            "interpretation": interpretation, "finished_utc": utc_now(),
        })

    def verify(self, run_id):
        directory = self.path(run_id)
        read_json(directory / "run.json")
        problems, registered = [], set()
        for event in sorted((directory / "events").glob("*.json")):
            data = read_json(event)
            if data["kind"] != "artifact":
                continue
            item = data["details"]
            rel = item["path"]
            path = (directory / rel).resolve()
            if not path.is_relative_to(directory) or path.is_symlink():
                problems.append(f"Unsafe artifact path: {rel}")
            elif not path.is_file():
                problems.append(f"Missing: {rel}")
            elif path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                problems.append(f"Corrupt: {rel}")
            registered.add(rel)
        for path in (directory / "artifacts").glob("*"):
            if path.relative_to(directory).as_posix() not in registered:
                problems.append(f"Unregistered artifact: {path.name}")
        return problems

    def mirror(self, run_id, destination):
        """Copy without overwrites; partial transfers can be resumed explicitly.

        A live run may append later files. The receipt certifies this copy's file
        set only. Run again after finalization to include the final result.
        """
        source = self.path(run_id)
        problems = self.verify(run_id)
        if problems:
            raise ValueError(f"Cannot mirror corrupt run: {problems}")
        target = Path(destination).resolve() / run_id
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("Backup must be outside the source run")
        copied = []
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlinks are not allowed in run archives")
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            rel = path.relative_to(source)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            source_hash = digest(path)
            if dst.exists():
                if digest(dst) != source_hash:
                    raise ValueError(f"Backup conflict; refusing overwrite: {dst}")
            else:
                temporary = dst.with_name("." + dst.name + "." + uuid.uuid4().hex + ".transfer.tmp")
                with path.open("rb") as src, temporary.open("xb") as output:
                    shutil.copyfileobj(src, output)
                    output.flush()
                    os.fsync(output.fileno())
                if digest(temporary) != source_hash:
                    raise IOError(f"Incomplete backup payload retained at {temporary}")
                if dst.exists():
                    raise FileExistsError(f"Concurrent backup destination appeared: {dst}")
                # One coordinator per archive. Rename only a fully verified file;
                # interrupted copies remain separate .tmp evidence and can retry.
                temporary.rename(dst)
            if digest(dst) != source_hash:
                raise IOError(f"Backup verification failed: {dst}")
            copied.append({"path": rel.as_posix(), "sha256": source_hash})
        receipt = {"created_utc": utc_now(), "run_id": run_id,
                   "destination": str(target), "files": copied,
                   "scope": "Hash-verified copy; remote persistence depends on storage provider"}
        write_once(target / ("mirror-receipt-" + uuid.uuid4().hex + ".json"), receipt)
        return receipt
