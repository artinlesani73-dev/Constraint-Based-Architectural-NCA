"""NR3 frozen study settings and complete-case primary quality gate."""
from itertools import product
import numpy as np

SEEDS=(1201,1202,1203)
FIRING=(2101,2102,2103)
HORIZONS=(16,32,64)
CHECKPOINTS=(0,64,128,192,256)
SETTINGS={'version':'NR3_quality_v1','seeds':list(SEEDS),'updates':256,'train_steps':16,
          'seconds_per_job':600,'max_controlled_training_seconds':1800,
          'proposed_gpu_allocation_minutes':60,'threshold':.5,'primary_steps':32,
          'firing_seeds':list(FIRING),'horizons':list(HORIZONS),
          'diagnostic_checkpoints':list(CHECKPOINTS),'iou_gain':.02,'intact_iou':.99,
          'dataset_run':'20260925T094341Z_316cff241020',
          'gpu_preflight_run':'20260925T134015Z_e14a0afc966e',
          'evaluation_device':'cpu','evaluation_dtype':'float32',
          'validation_diagnostics':'27 rows,32 steps,firing2101,checkpoints0/64/128/192/256',
          'gpu_job_requires_approval':True,'drive_operations_require_separate_approval':True}


def primary_gate(records, examples):
    """A missing/duplicate primary observation cannot silently improve a score."""
    test=[x for x in examples if x['split']=='test']
    lookup={(x['case'],x['damage']):x for x in test}
    if len(test)!=54 or len(lookup)!=54:raise ValueError('Exactly 54 frozen test rows required')
    expected={(s,f,c,d) for s,f in product(SEEDS,FIRING) for c,d in lookup}
    actual={}
    for x in records:
        if x['split']!='test' or x['steps']!=32 or x['checkpoint']!=256:continue
        key=(x['seed'],x['firing_seed'],x['case'],x['damage'])
        if key in actual:raise ValueError('Duplicate primary observation')
        actual[key]=x
    if set(actual)!=expected:raise ValueError('Incomplete or unexpected primary observations')
    def aggregate(rows):
        return {'count':len(rows),'median_iou':float(np.median([x['iou'] for x in rows])),
                'all_nine_pass_rate':float(np.mean([x['targets']['contract_pass'] for x in rows]))}
    damaged=[x for x in test if x['damage']!='intact']
    baselines={m:aggregate([x['metrics'][m] for x in damaged]) for m in ('unchanged','closing3')}
    by_seed=[]
    for seed in SEEDS:
        rows=[x for x in actual.values() if x['seed']==seed]
        damaged_rows=[x for x in rows if x['damage']!='intact']
        stats=aggregate([x['metrics'] for x in damaged_rows])
        gains={m:stats['median_iou']-v['median_iou'] for m,v in baselines.items()}
        intact_ok=all(x['metrics']['iou']>=.99 and x['metrics']['targets']['contract_pass'] for x in rows if x['damage']=='intact')
        volume_ok=all(abs(x['metrics']['request_error_cells'])<=max(8,.01*x['metrics']['targets']['domain_voxels'])
                      for x in damaged_rows if x['metrics']['targets']['contract_pass'])
        passed=all(v>=.02 for v in gains.values()) and all(stats['all_nine_pass_rate']>=v['all_nine_pass_rate'] for v in baselines.values()) and intact_ok and volume_ok
        strata={d:aggregate([x['metrics'] for x in rows if x['damage']==d]) for d in ('intact','cube5','slab2')}
        by_seed.append({'seed':seed,'passed':bool(passed),'damaged':stats,'iou_gains':gains,
                        'all_intact_preserved':bool(intact_ok),'accepted_repairs_meet_volume_tolerance':bool(volume_ok),'strata':strata})
    return {'version':SETTINGS['version'],'primary_observations':len(actual),'baselines':baselines,
            'by_seed':by_seed,'admit_further_repair_study':all(x['passed'] for x in by_seed),
            'studio_promotion':False,'note':'Model seeds are the three replications; firing seeds and related synthetic sites are not independent models.'}
