"""Write an evidence-linked E0 report from verified, finalized run artifacts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca.e0 import aggregate
from nca.e0_analysis import target_audit
from nca.experiments import RunStore, read_json, write_once

BASELINES = ('historical-training', 'historical-evaluation', 'historical-serving')
COMPARISONS = (
    ('serving-seed-015', 'historical-serving', 'Serving seed scale 0.005 -> 0.15'),
    ('serving-no-noise', 'historical-serving', 'Remove serving noise'),
    ('serving-no-mask', 'historical-serving', 'Remove serving step mask'),
    ('training-fire-1', 'historical-training', 'Training firing rate 0.65 -> 1'),
    ('serving-state-fire-065', 'historical-serving', 'Serving state firing 1 -> 0.65'),
    ('serving-delta-fire-065', 'serving-state-fire-065', 'State blend -> delta mask at rate 0.65'),
)


def build_report(run_id):
    store = RunStore(REPO / '.local-artifacts/runs')
    problems = store.verify(run_id)
    if problems:
        raise ValueError(f'Cannot report corrupt evidence: {problems}')
    directory = store.path(run_id)
    final = read_json(directory / 'result.json')
    if final['status'] != 'completed':
        raise ValueError('This comparison report requires a completed run; inspect retained failure evidence instead')
    metadata = read_json(directory / 'run.json')
    protocol = metadata['config']
    cases = []
    for path in sorted((directory / 'events').glob('*.json')):
        event = read_json(path)
        if event['kind'] == 'artifact' and event['details']['role'] == 'case_record':
            cases.append(read_json(directory / event['details']['path']))
    if len(cases) != protocol['expected_cases'] or any(c['status'] != 'completed' for c in cases):
        raise ValueError('The full completed case matrix is required')
    identities = {(c['scene_set'], c['scene_id'], c['profile_name'], c['seed']) for c in cases}
    if len(identities) != len(cases):
        raise ValueError('Duplicate experimental case identities')
    groups = aggregate(cases)
    audit = target_audit(REPO, directory, protocol, cases)
    lines = [
        '# E0: what the historical Model C checkpoint actually does', '',
        f'Run: `{run_id}`. Protocol: `{protocol["protocol"]}`. Rollout: `{protocol["rollout_version"]}`.',
        f'Source commit: `{metadata["provenance"]["commit"]}`.',
        f'All {len(cases)} prescribed cases completed; registered artifacts passed SHA-256 verification.', '',
        '## Main comparison', '',
        'All profiles use the same frozen scenes and 50 update steps. Each row pools the same',
        'three predetermined seeds (0, 1, 2). Training denotes forward dynamics at epoch',
        'position 60 with fixed weights; no training or optimizer step occurred.', '',
        '| Scene set | Profile | Cases | Mean material voxels | Connected / scored | Unscorable | Empty |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for group in groups:
        if group['profile'] in BASELINES:
            lines.append(f'| {group["scene_set"]} | {group["profile"]} | {group["cases"]} | '
                         f'{group["mean_material_voxels"]:.1f} | {group["connected_cases"]}/{group["connectivity_scored_cases"]} | '
                         f'{group["connectivity_unscorable_cases"]} | {group["empty_cases"]} |')
    lines += ['', 'The historical evaluation profile is deterministic: its three seeded repeats',
              'check consistency, not three independent realizations. Repeated seeds do not',
              'increase the number of distinct sites: there are 6 reference and 12 legacy scenes.', '',
              '## Paired setting changes', '',
              'Ablations were fixed in advance at seed 0 only. Every comparison below pairs the',
              'same scene and seed; it does not compare a one-seed ablation with a three-seed mean.',
              'These are preliminary diagnostic effects, without confidence intervals or a',
              'claim that a setting is reliably better across random seeds.', '',
              '| Set | Change | Pairs | Mean voxel change | Connectivity gained | Connectivity lost | Unscorable pairs |',
              '|---|---|---:|---:|---:|---:|---:|']
    index = {(c['scene_set'], c['scene_id'], c['profile_name'], c['seed']): c for c in cases}
    for scene_set in sorted({c['scene_set'] for c in cases}):
        scenes = sorted({c['scene_id'] for c in cases if c['scene_set'] == scene_set})
        for variant, baseline, label in COMPARISONS:
            pairs = [(index[(scene_set, scene, baseline, 0)], index[(scene_set, scene, variant, 0)]) for scene in scenes]
            delta = sum(b['metrics']['legality']['material_voxels'] - a['metrics']['legality']['material_voxels']
                        for a, b in pairs) / len(pairs)
            gained = lost = unscorable = 0
            for a, b in pairs:
                before, after = a['metrics']['connectivity']['all_connected'], b['metrics']['connectivity']['all_connected']
                if before is None or after is None:
                    unscorable += 1
                else:
                    gained += int(not before and after)
                    lost += int(before and not after)
            lines.append(f'| {scene_set} | {label} | {len(pairs)} | {delta:+.1f} | {gained} | {lost} | {unscorable} |')
    lines += ['', '## Per-seed consistency', '',
              '| Set | Profile | Seed | Connected / scored | Unscorable | Mean voxels |',
              '|---|---|---:|---:|---:|---:|']
    for scene_set in sorted({c['scene_set'] for c in cases}):
        for profile in BASELINES:
            for seed in (0, 1, 2):
                subset = [c for c in cases if (c['scene_set'], c['profile_name'], c['seed']) == (scene_set, profile, seed)]
                group = aggregate(subset)[0]
                lines.append(f'| {scene_set} | {profile} | {seed} | {group["connected_cases"]}/{group["connectivity_scored_cases"]} | '
                             f'{group["connectivity_unscorable_cases"]} | {group["mean_material_voxels"]:.1f} |')
    lines += ['', '## Target and permitted-space audit', '',
              'This is a post-run diagnostic derived from the saved training/seed-0 inputs for',
              'every scene. It did not change the frozen experiment protocol or any rollout.',
              'Permitted connectivity asks whether ANY face-connected material route could',
              'join the entrance regions under the existing hard legality rule. It is a spatial',
              'upper bound, not proof of a usable or structurally sound architecture.', '',
              '| Set | Scene | Permitted region connects entrances | Legal part of target connects | Illegal target voxels / total |',
              '|---|---|---|---|---:|']
    for row in audit:
        legality = row['raw_target_legality']
        lines.append(f'| {row["scene_set"]} | {row["scene_id"]} | '
                     f'{row["permitted_connectivity"]["all_connected"]} | '
                     f'{row["legal_target_connectivity"]["all_connected"]} | '
                     f'{legality["illegal_voxels"]}/{legality["material_voxels"]} |')
    lines += ['',
              'The sealed-partition reference scene is an intentional negative control; failure',
              'to connect it is expected and is not a performance defect. Keep it separate',
              'when interpreting the aggregate counts above. Check the audit rows before',
              'attributing a failure to the network: permitted-space connectivity and target',
              'connectivity are distinct from the network output.', '',
              'An illegal target voxel cannot be occupied after hard projection. A disconnected',
              'legalized target requires routing/target corrections, not just deleting illegal',
              'voxels. The complete learned-value comparison to procedural controls remains E2.', '']
    lines += ['', '## Geometry and measurement limits', '',
              '| Set | Profile | Illegal voxels across cases | Blocked protected voxels | Geometrically unsupported voxels |',
              '|---|---|---:|---:|---:|']
    for group in groups:
        if group['profile'] in BASELINES:
            lines.append(f'| {group["scene_set"]} | {group["profile"]} | {group["illegal_voxels_total"]} | '
                         f'{group["blocked_protected_voxels_total"]} | {group["unsupported_voxels_total"]} |')
    lines += ['',
        '- Material uses strict structure-channel value > 0.5. Counts at 0.3 and 0.7 are retained per case; the connectivity results here are only at 0.5.',
        '- Connectivity is through grown material with six face neighbors, starting at the lexicographically first named entrance. It requires reaching every other entrance region. It does not establish walking clearance, deck support, usable rooms or access quality.',
        '- If the source entrance intersects several disconnected material pieces, connectivity is explicitly unscorable. Such cases remain in the evidence and are shown separately, not converted into successes or silently dropped.',
        '- Legality and protected-ground openness are largely consequences of hard projection in the historical model. An empty design can satisfy both. These columns do not prove that the network learned architectural intent.',
        '- Geometric support means connection to a declared support region. Eroded-core fraction is a voxel thickness proxy. Neither is a mechanical analysis. This is not a complete validator for all nine constraint families.',
        '- The legacy corridor operator, including its known vertical-envelope defect and final height-band clamp, is unchanged in this baseline. Correct it under a new operator version and compare again.',
        '- The legacy scenes are sampled from the easy generator but filtered by the explicit geometry contract; the reference scenes are authored development cases. Neither set is a sealed, representative architectural test benchmark.',
        '- This diagnoses one checkpoint. The target audit is a limited scaffold-connectivity check; a full learned-value comparison across geometry criteria and cost remains E2. These results alone do not justify a concept pivot.',
        '- CPU timings are retained per case, without warmup or hardware control. They are diagnostic runtime records, not deployment latency or GPU performance benchmarks.',
        '', '## Reproduction and evidence', '',
        f'Raw run folder: `.local-artifacts/runs/{run_id}/`.',
        f'Small summary: `experiments/records/{run_id}.json`.',
        'The run has source snapshot, exact effective config, checkpoint/scene-manifest hashes,',
        'all per-case JSON records and compressed NPZ fields. Each NPZ contains the input,',
        'full final state, corridor target when used, and thresholded material. Fields remain',
        'local and excluded from Git; no Drive upload occurred.', '',
        'Run `python scripts/experiment.py verify ' + run_id + '` to check the archive.',
        'Run `python scripts/run_e0.py --parent-run ' + run_id + '` for a fresh linked retry.',
        'A retry repeats the full frozen protocol and retains the prior attempt.', '',
        '## Next implementation decision', '',
        'First correct the bounded vertical-envelope operator and compare its targets on the',
        'same scenes. Then repair loss semantics and gradients and run the corrected NCA',
        'against explicit procedural/scaffold-only/direct-optimization controls. Keep all',
        'existing constraint families. Choose architectural changes from those comparisons,',
        'while product work uses the stable scene and result records established here.', '',
    ]
    return '\n'.join(lines), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    text, audit = build_report(args.run_id)
    path = REPO / 'docs/next-phase/reports' / (args.run_id + '-E0.md')
    path.parent.mkdir(exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_once(path.with_suffix('.target-audit.json'), {'source_run': args.run_id, 'analysis_version': 'target_audit_v1', 'scenes': audit})
    print(path)


if __name__ == '__main__':
    main()
