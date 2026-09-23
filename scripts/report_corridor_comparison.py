"""Render C1 tables from hash-verified registered records, never working copies."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import json

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.experiments import RunStore, read_json, write_once
from nca.e0 import aggregate

VERSIONS = ('legacy_v31', 'corridor_bounded_v1', 'corridor_legal_v1')


def load_verified(run_id):
    store = RunStore(REPO / '.local-artifacts/runs')
    problems = store.verify(run_id)
    if problems:
        raise ValueError(problems)
    directory = store.path(run_id)
    final = read_json(directory / 'result.json')
    if final['status'] != 'completed':
        raise ValueError('Cannot report an incomplete or failed comparison as complete')
    items = defaultdict(list)
    for path in (directory / 'events').glob('*.json'):
        event = read_json(path)
        if event['kind'] == 'artifact' and event['details']['role'] in ('target_record', 'case_record', 'summary', 'protocol'):
            items[event['details']['role']].append(read_json(directory / event['details']['path']))
    if len(items['protocol']) != 1 or len(items['summary']) != 1:
        raise ValueError('Ambiguous protocol/summary')
    protocol, summary = items['protocol'][0], items['summary'][0]
    targets, cases = items['target_record'], items['case_record']
    if len(targets) != protocol['expected_targets'] or len(cases) != protocol['expected_cases']:
        raise ValueError('Incomplete registered evidence')
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    if len({(t['scene_set'], t['scene_id'], t['version']) for t in targets}) != len(targets):
        raise ValueError('Duplicate targets')
    if aggregate(cases) != summary['groups']:
        raise ValueError('Summary differs from registered cases')
    return protocol, summary, targets, cases


def render(run_id):
    protocol, summary, targets, cases = load_verified(run_id)
    lines = [f'# C1 corridor comparison: {run_id}', '',
             f'Source commit: `{summary["provenance"]["commit"]}`. Protocol C1_v1, '
             'one historical checkpoint, seed 0, CPU, 50 steps. No optimizer updates.', '',
             f'{len(targets)} target records and {len(cases)} forward cases; '
             f'{summary["e0_bitwise_equal_cases"]}/36 legacy final states exactly reproduce E0.', '',
             'All registered artifact hashes verified before rendering. Width 1, extra '
             'vertical envelope 1. Material >0.5, six-neighbor spatial connectivity. '
             'The reference set includes one deliberately infeasible sealed partition.', '',
             '## Target comparison', '',
             '| Set | Version | Connected legal targets | Illegal target voxels (total) | Mean target voxels |',
             '|---|---|---:|---:|---:|']
    for name in sorted({t['scene_set'] for t in targets}):
        for version in VERSIONS:
            rows = [t for t in targets if (t['scene_set'], t['version']) == (name, version)]
            scored = [r for r in rows if r['legal_connectivity']['all_connected'] is not None]
            connected = sum(r['legal_connectivity']['all_connected'] for r in scored)
            lines.append(f'| {name} | {version} | {connected}/{len(scored)} '
                         f'({len(rows)-len(scored)} unscorable) | '
                         f'{sum(r["legality"]["illegal_voxels"] for r in rows)} | '
                         f'{sum(r["legality"]["material_voxels"] for r in rows)/len(rows):.1f} |')
    lines += ['', 'The bounded arm changes only the faulty vertical scan. The legal arm is a '
              'separate package of routing changes: explicit IDs, legal six-neighbor paths, '
              'no distance cutoff or endpoint-height clip, bounded thickening and removal '
              'of disconnected thickening fragments. Its effect cannot be assigned to one '
              'of those individual changes.', '', '## Matched forward results', '',
              '| Set | Profile | Target version | Connected | Unscorable | Mean voxels | Empty | Illegal total | Unsupported total |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['groups']:
        profile, version = row['profile'].split('__')
        lines.append(f'| {row["scene_set"]} | {profile} | {version} | '
                     f'{row["connected_cases"]}/{row["connectivity_scored_cases"]} | '
                     f'{row["connectivity_unscorable_cases"]} | {row["mean_material_voxels"]:.1f} | '
                     f'{row["empty_cases"]} | {row["illegal_voxels_total"]} | {row["unsupported_voxels_total"]} |')
    lines += ['', '## Paired changes in connectivity', '',
              '| Set | Profile | Comparison | Gained | Lost | Same | Unscorable pairs | Mean voxel difference |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for name in sorted({c['scene_set'] for c in cases}):
        for profile in sorted({c['base_profile'] for c in cases}):
            subset = [c for c in cases if c['scene_set'] == name and c['base_profile'] == profile]
            by_key = {(c['scene_id'], c['version']): c for c in subset}
            for old, new in ((VERSIONS[0], VERSIONS[1]), (VERSIONS[1], VERSIONS[2]), (VERSIONS[0], VERSIONS[2])):
                gained = lost = same = unscorable = 0
                differences = []
                for sid in sorted({c['scene_id'] for c in subset}):
                    a, b = by_key[sid, old]['metrics'], by_key[sid, new]['metrics']
                    ac, bc = a['connectivity']['all_connected'], b['connectivity']['all_connected']
                    differences.append(b['legality']['material_voxels'] - a['legality']['material_voxels'])
                    if ac is None or bc is None:
                        unscorable += 1
                    else:
                        gained += int(bc and not ac)
                        lost += int(ac and not bc)
                        same += int(ac == bc)
                lines.append(f'| {name} | {profile} | {old} -> {new} | {gained} | {lost} | {same} | '
                             f'{unscorable} | {sum(differences)/len(differences):+.1f} |')
    lines += ['', '## Individual targets', '',
              '| Scene | Version | Voxels | Illegal | Legal connected | Z range | Added vs legacy | Removed vs legacy |',
              '|---|---|---:|---:|---|---|---:|---:|']
    for row in sorted(targets, key=lambda t: (t['scene_set'], t['scene_id'], VERSIONS.index(t['version']))):
        lines.append(f'| {row["scene_id"]} | {row["version"]} | {row["legality"]["material_voxels"]} | '
                     f'{row["legality"]["illegal_voxels"]} | {row["legal_connectivity"]["all_connected"]} | '
                     f'{row["z_range_inclusive"]} | {row["differences"]["legacy"]["added"]} | '
                     f'{row["differences"]["legacy"]["removed"]} |')
    lines += ['', '## Individual rollouts', '',
              '| Scene | Profile | Version | Connected | Voxels | Unsupported |',
              '|---|---|---|---|---:|---:|']
    for row in sorted(cases, key=lambda c: (c['scene_set'], c['scene_id'], c['base_profile'], VERSIONS.index(c['version']))):
        m = row['metrics']
        lines.append(f'| {row["scene_id"]} | {row["base_profile"]} | {row["version"]} | '
                     f'{m["connectivity"]["all_connected"]} | {m["legality"]["material_voxels"]} | '
                     f'{m["support"]["unsupported_voxels"]} |')
    lines += ['', '## Evidence and limits', '',
              f'Run payloads: `.local-artifacts/runs/{run_id}/`. Each target record links '
              'the saved seed, all three targets, permitted field and legal centerline. '
              'Legal-router records include exact selected paths and infeasibility status. '
              'Each rollout links its input artifact, full continuous final state, profile, '
              'RNG seed and per-case metrics. Target JSON retains endpoint contact, '
              'permitted-space connectivity, radius-zero parity and both target-difference '
              'comparisons; the accompanying target-audit JSON exposes those details.', '',
              'Radius-zero parity: ' + str(sum(t['radius_zero_parity'] for t in targets)) + '/54 '
              'records (one check per scene, repeated across its three target records).', '',
              'These are development scenes and a single rollout seed. Spatial connectivity '
              'does not establish headroom, walking clearance or structural safety. '
              'A smaller target or fewer material voxels is not automatically a better '
              'architecture. Hard projection accounts for legality of the model output. '
              'The checkpoint was not trained against the new targets. Full E2 learned-value '
              'comparisons and corrected losses/gradients remain pending. Production defaults '
              'and the original notebook/checkpoint are unchanged. Archives remain local.', '']
    return '\n'.join(lines), targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    report, targets = render(args.run_id)
    destination = REPO / 'docs/next-phase/reports' / (args.run_id + '-C1.md')
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(report)
    write_once(destination.with_suffix('.target-audit.json'), targets)
    print(destination)


if __name__ == '__main__':
    main()
