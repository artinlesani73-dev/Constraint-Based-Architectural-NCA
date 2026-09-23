"""Verify R2 full checkpoint/trace/field parity from registered immutable artifacts."""
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import RunStore,read_json,write_once
from nca.recovery import tree_equal


def verify(run):
    store=RunStore(REPO/'.local-artifacts/runs');assert not store.verify(run)
    d=store.path(run);assert read_json(d/'result.json')['status']=='completed'
    branches={};summary=None
    for path in sorted((d/'events').glob('*.json')):
        event=read_json(path)
        if event['kind']!='artifact':continue
        role=event['details']['role']
        if role=='recovery_branch':
            r=read_json(d/event['details']['path']);branches[r['branch']]=r
        if role=='summary':summary=read_json(d/event['details']['path'])
    assert set(branches)=={'uninterrupted','prefix','resumed','repeat'}
    def payload(name,index=-1):return torch.load(d/branches[name]['checkpoints'][index]['checkpoint']['path'],weights_only=True,map_location='cpu')
    reference=payload('uninterrupted')
    assert tree_equal(payload('prefix'),payload('uninterrupted',1))
    for name in ('resumed','repeat'):
        assert tree_equal(reference,payload(name))
        assert branches['prefix']['trace']+branches[name]['trace']==branches['uninterrupted']['trace']
        combined=branches['prefix']['checkpoints']+branches[name]['checkpoints']
        assert len(combined)==4
        for left,right in zip(branches['uninterrupted']['checkpoints'],combined):
            with np.load(d/left['fields']['path'],allow_pickle=False) as a,np.load(d/right['fields']['path'],allow_pickle=False) as b:
                assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    assert len({r['scene'] for r in branches['uninterrupted']['trace']})==3
    assert all(summary['checks'].values())
    return summary,branches['uninterrupted']['trace']


def render(run,summary,trace):
    lines=['# R2 composed-objective CPU recovery','',f'Run `{run}`; source `{summary["provenance"]["commit"]}`.',
        '', 'Four logical updates,ten executed across four fresh processes. All seven recovery comparisons independently rechecked against registered checkpoint/field artifacts: prefix boundary, full model/optimizer/scheduler/RNG checkpoint trees, traces and fields for resume and repeated resume. All three scheduled scenes exercised.',
        '', '| Update | Scene | Steps | Total objective | Gradient norm before clipping |', '|---:|---|---:|---:|---:|']
    for r in trace:lines.append(f'| {r["update"]} | {r["scene"]} | {r["steps"]} | {r["total_loss"]:.8g} | {r["gradient_norm_before_clip"]:.8g} |')
    lines+=['','The mass_3 candidate recipe composes all nine corrected families and three retained regularizers. These updates validate recovery mechanics, not coefficient quality or a learning curve across differing scenes. Source/scene/proposal hashes and full coefficients are checkpoint metadata.','',
        'Boundary: completed CPU updates after orderly exit. This does not certify CUDA, mixed precision, sample pools, abrupt mid-write failure or the later K2 training loop. The original checkpoint remains unchanged. No paid compute or Drive operation.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    run=sys.argv[1];summary,trace=verify(run)
    with (REPO/'experiments/reports/R2-recovery.md').open('x',encoding='utf-8') as f:f.write(render(run,summary,trace))
    write_once(REPO/'experiments/reports/R2-verification.json',{'run_id':run,'full_checkpoint_comparisons':3,'field_trace_continuations':2,'all_three_scenes_exercised':True})
    print('R2 independently verified')
