"""Write the frozen in-distribution scene set and its manifest.

Scenes are drawn from the historical `easy` distribution at fixed seeds, in
seed order, with no cherry-picking: seeds are consumed from zero upward and
every one is either accepted or recorded with the reason it was refused. The
manifest carries that sampling record, so the set is reproducible and its
selection is auditable.

Like the designed set, this one is frozen: an existing scene file is never
replaced without `--force`, which is for the initial authoring pass only.

Usage, from the repository root:

    python scripts/build_legacy_scenes.py
    python scripts/build_legacy_scenes.py --manifest
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nca.contract import (CONTRACT_VERSION, MANIFEST_NAME, build_manifest,
                          canonical_json, load_reference_set)
from nca.legacy_scenes import (LEGACY_PROVENANCE, LEGACY_SET_VERSION, EASY_PARAMS,
                               sample_validated)

REPO_ROOT = Path(__file__).resolve().parents[1]
SET_DIR = REPO_ROOT / "experiments" / "scenes" / LEGACY_SET_VERSION
TARGET_COUNT = 12
MAX_SEED = 200

STREET_LEVELS = 6
GRID = 32
VOXEL = 0.8


def draw():
    """Accepted scenes in seed order plus the full rejection record."""
    accepted, rejected = [], []
    seed = 0
    while len(accepted) < TARGET_COUNT and seed < MAX_SEED:
        scene_id = f"legacy-easy-seed-{seed:03d}"
        scene, reason = sample_validated(
            seed, grid_size=GRID, street_levels=STREET_LEVELS,
            voxel_size_m=VOXEL, scene_id=scene_id)
        if scene is None:
            rejected.append({"seed": seed, "scene_id": scene_id, "reason": reason})
        else:
            accepted.append((seed, f"{scene_id}.json", scene))
        seed += 1
    if len(accepted) < TARGET_COUNT:
        raise SystemExit(
            f"Only {len(accepted)} of {TARGET_COUNT} scenes accepted below seed {MAX_SEED}")
    return accepted, rejected, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Replace existing scene files. Authoring use only.")
    parser.add_argument("--manifest", action="store_true",
                        help="Rewrite the manifest from the files already present.")
    arguments = parser.parse_args()

    SET_DIR.mkdir(parents=True, exist_ok=True)
    accepted, rejected, seeds_consumed = draw()

    if not arguments.manifest:
        for _, filename, scene in accepted:
            path = SET_DIR / filename
            data = canonical_json(scene)
            if path.exists():
                if path.read_bytes() == data:
                    print(f"unchanged {filename}")
                    continue
                if not arguments.force:
                    raise SystemExit(
                        f"Refusing to change frozen scene {filename}; pass --force only when "
                        "authoring a set that has no recorded results against it")
                print(f"replaced  {filename}")
            else:
                print(f"created   {filename}")
            path.write_bytes(data)

    manifest = build_manifest(SET_DIR, LEGACY_SET_VERSION)
    manifest["sampling"] = {
        "provenance": LEGACY_PROVENANCE,
        "difficulty": "easy",
        "difficulty_params": {key: list(value) if isinstance(value, tuple) else value
                              for key, value in EASY_PARAMS.items()},
        "street_levels": STREET_LEVELS,
        "grid_size": GRID,
        "voxel_size_m": VOXEL,
        "selection": "seeds consumed from 0 upward in order; no cherry-picking",
        "seeds_consumed": seeds_consumed,
        "accepted_seeds": [seed for seed, _, _ in accepted],
        "rejected": rejected,
        "seed_state_builder": ("nca.legacy_scenes.legacy_seed_state; the deployed "
                               "generator writes different anchors for a scene with an "
                               "entrance below the street band and must not be used to "
                               "replay this set"),
    }
    (SET_DIR / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    loaded = load_reference_set(SET_DIR)
    relaxed = sum(1 for scene in loaded.values() if scene["legacy_relaxations"])
    print(f"manifest  {len(loaded)} scenes verified; {relaxed} declare "
          f"facade_below_street_band; {len(rejected)} seeds rejected")
    for entry in rejected:
        print(f"rejected  seed {entry['seed']}: {entry['reason']}")


if __name__ == "__main__":
    main()
