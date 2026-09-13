"""Portable local/Colab run archive CLI. See docs/next-phase/EXPERIMENTS.md."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.experiments import RunStore, provenance, read_json, snapshot_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / ".local-artifacts" / "runs")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--kind", required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--parent")
    create.add_argument("--snapshot", action="store_true")
    for name in ("attach", "finish", "verify", "mirror", "event"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
        if name == "attach":
            command.add_argument("file", type=Path)
            command.add_argument("--role", required=True)
        if name == "finish":
            command.add_argument("--status", required=True, choices=["completed", "failed", "interrupted"])
            command.add_argument("--metrics", type=Path, required=True)
            command.add_argument("--interpretation", required=True)
        if name == "mirror":
            command.add_argument("destination", type=Path)
        if name == "event":
            command.add_argument("--kind", required=True)
            command.add_argument("--message", required=True)
    args = parser.parse_args()
    store = RunStore(args.root)
    if args.command == "create":
        run = store.create(args.name, args.kind, read_json(args.config), args.seed, provenance(REPO), args.parent)
        if args.snapshot:
            temporary = store.path(run) / "source-snapshot.zip"
            snapshot_source(REPO, temporary)
            store.attach(run, temporary, "source_snapshot")
            temporary.unlink()  # Verified payload copy is now the durable artifact.
        print(run)
    elif args.command == "attach":
        print(json.dumps(store.attach(args.run_id, args.file, args.role), indent=2))
    elif args.command == "finish":
        store.finish(args.run_id, args.status, read_json(args.metrics), args.interpretation)
    elif args.command == "verify":
        problems = store.verify(args.run_id)
        print(json.dumps({"run_id": args.run_id, "problems": problems}, indent=2))
        return bool(problems)
    elif args.command == "mirror":
        print(json.dumps(store.mirror(args.run_id, args.destination), indent=2))
    elif args.command == "event":
        print(store.event(args.run_id, args.kind, args.message))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
