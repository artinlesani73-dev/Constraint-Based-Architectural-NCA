"""Build a self-contained local viewer only from completed, verified result artifacts."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from nca.experiments import read_json,write_once,digest
from scripts.run_sensitivity import STORE,records
from scripts.diagnostic_inputs import load_inputs

def main(run):
    assert not STORE.verify(run)
    assert read_json(STORE.path(run)/'result.json')['status']=='completed'
    receipt=read_json(REPO/'experiments/reports/D1-verification.json');assert receipt['run_id']==run
    protocol=records(run,'protocol')[0];assert protocol['mode']=='full'
    compare=protocol['config']['source_comparison_run'];assert not STORE.verify(compare)
    _,inputs=load_inputs(REPO);scenes={};models={}
    for sid in protocol['config']['scenes']:
        s=inputs[sid]['scene'];scenes[sid]={'title':sid.replace('ref-01-','').replace('ref-02-','').replace('ref-03-','').replace('ref-04-','').replace('ref-06-','').replace('-',' '),
            'description':s['description'],'buildings':s['buildings'],'entrances':s['entrances'],'voxel_size_m':s['voxel_size_m'],'results':{}}
    def add(source,fields,row,key,label,method):
        with np.load(STORE.path(source)/fields['path'],allow_pickle=False) as f:voxels=np.argwhere(f['material'][0]>.5).tolist()
        assert len(voxels)==row['metrics']['legality']['material_voxels']
        models[key]=label;scenes[row.get('scene',row.get('scene_id'))]['results'][key]={'voxels':voxels,
            'mass_ratio':row['mass_ratio'],'connected':row['metrics']['connectivity']['all_connected'],
            'totals':row.get('totals',row.get('totals_under_both_recipes')),'terms':{**row['terms'],**row['regularizers']},'run':source,'method':method}
    for row in records(compare,'evaluation_record'):
        b=row['branch'];h=row['steps'];key=b+'-'+str(h)
        if b=='W1_procedural':label='Procedural control / static';method='Static procedural witness. No learned generation or architectural quality claim.'
        elif b=='original_checkpoint':label=f'Original NCA / {h} growth steps';method=f'Original checkpoint, {h} growth steps, firing seed 2. No optimization for this scene.'
        else:label=f'NCA weight {30 if b.startswith("mapped") else 3}, seed {b[-1]} / {h} growth steps';method=f'K2 model after 17 shared-weight training updates; {h} growth steps here, firing seed 2.'
        add(compare,row['fields'],row,key,label,method)
    for case in records(run,'direct_case'):
        recipe=case['recipe'];label=f'Direct voxels / material weight {30 if recipe=="mapped_30" else 3}'
        add(run,case['final']['fields'],case['final'],'direct-'+recipe,label,f'32 optimizer updates for this scene only. Started from weak 0.15 scaffold; no learned update rule.')
    assert len(scenes)==17 and all(len(s['results'])==13 for s in scenes.values())
    payload={'direct_run':run,'models':models,'scenes':scenes}
    encoded=json.dumps(payload,separators=(',',':')).replace('<','\\u003c')
    text=(REPO/'assets/experiment_viewer.html').read_text(encoding='utf-8-sig').replace('__DATA__',encoded)
    folder=REPO/'.local-artifacts/viewers'/('D1-'+run);folder.mkdir(parents=True,exist_ok=False)
    with (folder/'index.html').open('x',encoding='utf-8') as f:f.write(text)
    manifest={'direct_run':run,'comparison_run':compare,'scenes':17,'variants_per_scene':13,'path':str(folder/'index.html'),
        'sha256':digest(folder/'index.html'),'binary_threshold':.5,'source_coordinate_order':'z,y,x','total_result_fields':221}
    write_once(folder/'manifest.json',manifest);write_once(REPO/'experiments/reports/D1-viewer-manifest.json',manifest)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main(sys.argv[1])
