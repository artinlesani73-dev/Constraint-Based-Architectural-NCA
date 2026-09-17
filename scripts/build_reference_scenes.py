"""Write the frozen reference scene set and its manifest.

The set is frozen: every comparison recorded against a reference scene assumes
its bytes never changed. This script therefore refuses to replace an existing
scene file unless ``--force`` is given, and ``--force`` is intended for the
initial authoring pass only. Regenerating a scene after results exist against it
requires a new set version and a recorded decision, not an overwrite.

Usage, from the repository root:

    python scripts/build_reference_scenes.py            # create missing files
    python scripts/build_reference_scenes.py --manifest # rewrite manifest only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nca.contract import (CONTRACT_VERSION, MANIFEST_NAME, REFERENCE_SET_DIR,
                          build_manifest, canonical_json, load_reference_set)

SET_VERSION = "reference_v1"

# street_levels follows the authoritative embedded Model C configuration (6),
# not the external config_step_b.json value (2). At 0.8 m per voxel the band is
# 4.8 m, close to the 5 m street zone of the earlier Step D specification.
STREET_LEVELS = 6
GRID = 32
VOXEL = 0.8


def building(identifier, x, y, z, gap_facing_x=None, side=None):
    return {"id": identifier, "x": list(x), "y": list(y), "z": list(z),
            "gap_facing_x": gap_facing_x, "side": side}


def entrance(identifier, kind, x, y, z, extent=2):
    return {"id": identifier, "kind": kind, "x": x, "y": y, "z": z, "extent": extent}


def scene(scene_id, description, buildings, entrances, notes):
    return {"contract_version": CONTRACT_VERSION, "scene_id": scene_id,
            "description": description, "grid_size": GRID, "voxel_size_m": VOXEL,
            "street_levels": STREET_LEVELS, "ceiling_z": None,
            "buildings": buildings, "entrances": entrances, "notes": notes}


def scenes():
    return [
        ("ref-01-ground-pair.json", scene(
            "ref-01-ground-pair",
            "Symmetric slab pair, 16-voxel gap, two ground entrances only.",
            [building("B_west", (0, 8), (6, 26), (0, 25), 8, "left"),
             building("B_east", (24, 32), (6, 26), (0, 25), 24, "right")],
            [entrance("E_ground_west", "ground", 9, 14, 0),
             entrance("E_ground_east", "ground", 21, 14, 0)],
            ["Ground-only case; anchor zones come from both entrances and both gap facades.",
             "Gap spans x in [8, 24), about 12.8 m at 0.8 m per voxel."])),

        ("ref-02-facade-pair-and-ground.json", scene(
            "ref-02-facade-pair-and-ground",
            "Same slab pair with two facade entrances at different heights plus a ground entrance.",
            [building("B_west", (0, 8), (6, 26), (0, 25), 8, "left"),
             building("B_east", (24, 32), (6, 26), (0, 25), 24, "right")],
            [entrance("E_facade_west", "facade", 8, 14, 10),
             entrance("E_facade_east", "facade", 22, 14, 16),
             entrance("E_ground_mid", "ground", 15, 14, 0)],
            ["Three entrances at distinct heights, matching the Step D access rule that the two "
             "facade points differ in height.",
             "The primary replay case for E0."])),

        ("ref-03-wide-gap.json", scene(
            "ref-03-wide-gap",
            "Narrow slabs with a 20-voxel gap and unequal heights.",
            [building("B_west", (0, 6), (4, 28), (0, 22), 6, "left"),
             building("B_east", (26, 32), (4, 28), (0, 28), 26, "right")],
            [entrance("E_facade_west", "facade", 6, 15, 8),
             entrance("E_facade_east", "facade", 24, 15, 14),
             entrance("E_ground_west", "ground", 7, 15, 0)],
            ["Span stress case: the gap is about 16 m, above the stated 10-30 m pavilion range midpoint.",
             "Tests whether a longer horizontal run stays connected and supported."])),

        ("ref-04-asymmetric-heights.json", scene(
            "ref-04-asymmetric-heights",
            "Offset blocks of very different height with a diagonal entrance relationship.",
            [building("B_tall", (0, 9), (2, 18), (0, 30), 9, "left"),
             building("B_low", (23, 32), (10, 30), (0, 16), 23, "right")],
            [entrance("E_facade_tall", "facade", 9, 10, 20),
             entrance("E_facade_low", "facade", 21, 20, 12),
             entrance("E_ground_gap", "ground", 15, 15, 0)],
            ["Entrances are offset in y as well as z, so a straight corridor cannot satisfy the scene.",
             "Height difference is 8 voxels, about 6.4 m."])),

        ("ref-05-sealed-partition.json", scene(
            "ref-05-sealed-partition",
            "Negative control: a full-height partition makes the two facade entrances unreachable.",
            [building("B_west", (0, 8), (0, 32), (0, 32)),
             building("B_partition", (14, 18), (0, 32), (0, 32)),
             building("B_east", (24, 32), (0, 32), (0, 32))],
            [entrance("E_facade_west", "facade", 8, 16, 10),
             entrance("E_facade_east", "facade", 22, 16, 10)],
            ["No legal path exists between the entrances. Any run reporting full connectivity here "
             "has a defective evaluator, not a good design.",
             "Declared with no gap facades, so anchor zones are empty and the street band is "
             "protected in full."])),

        ("ref-06-minimal-smoke.json", scene(
            "ref-06-minimal-smoke",
            "Small low-rise pair for fast regression use.",
            [building("B_west", (0, 10), (10, 22), (0, 14), 10, "left"),
             building("B_east", (22, 32), (10, 22), (0, 14), 22, "right")],
            [entrance("E_ground_a", "ground", 11, 15, 0),
             entrance("E_ground_b", "ground", 19, 15, 0)],
            ["Cheapest scene in the set; intended for smoke checks, not for reporting quality."])),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Replace existing scene files. Authoring use only.")
    parser.add_argument("--manifest", action="store_true",
                        help="Rewrite the manifest from the files already present.")
    arguments = parser.parse_args()

    directory = REFERENCE_SET_DIR
    directory.mkdir(parents=True, exist_ok=True)

    if not arguments.manifest:
        for filename, payload in scenes():
            path = directory / filename
            data = canonical_json(payload)
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

    manifest = build_manifest(directory, SET_VERSION)
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_reference_set(directory)
    print(f"manifest  {len(loaded)} scenes verified: {', '.join(sorted(loaded))}")


if __name__ == "__main__":
    main()
