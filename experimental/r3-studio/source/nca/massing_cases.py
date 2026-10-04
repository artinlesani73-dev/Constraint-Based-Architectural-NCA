"""MT1 declared development contexts and analytic controls; no learned generation."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from nca.contract import validate_scene


def target_context(scene, config, padding=(6.4,6.4,0.0)):
    from deploy.model_utils import UrbanSceneGenerator
    from nca.contract import fields_from_state, to_generator_params
    from nca.massing import opportunity_region
    state,_=UrbanSceneGenerator(dict(config)).generate(to_generator_params(scene))
    fields=fields_from_state(state,config,scene)
    domain,report=opportunity_region(scene,fields['permitted'],padding)
    return fields,domain,report


def target_scenes():
    path = Path(__file__).resolve().parents[1]/'experiments/scenes/reference_v1/ref-01-ground-pair.json'
    base = json.loads(path.read_text())
    base['entrances'] = [{'id':'E_west','kind':'facade','x':8,'y':15,'z':8,'extent':2},
                         {'id':'E_east','kind':'facade','x':22,'y':15,'z':8,'extent':2}]
    scenes = []
    for name in ('aligned','wide_gap','offset_interfaces','blocked_gap'):
        scene=deepcopy(base);scene['scene_id']='mt1-'+name
        scene['description']='MT1 analytic massing development context: '+name
        scene['notes']=['Constructed development audit, not held-out generalization evidence.']
        if name=='wide_gap':
            scene['buildings'][0]['x'][1]=6;scene['buildings'][0]['gap_facing_x']=6
            scene['buildings'][1]['x'][0]=26;scene['buildings'][1]['gap_facing_x']=26
            scene['entrances'][0]['x']=6;scene['entrances'][1]['x']=24
        if name=='offset_interfaces':
            scene['entrances'][1]['z']=12;scene['entrances'][1]['y']=18
        if name=='blocked_gap':
            scene['buildings'].append({'id':'B_partition','x':[15,17],'y':[0,32],'z':[0,32],'side':None,'gap_facing_x':None})
        scenes.append((name,validate_scene(scene)))
    return scenes


def target_controls(scene, domain):
    """Unclipped positive proposals and adversarial counterexamples, saved as-is."""
    empty=np.zeros_like(domain)
    entrances=scene['entrances'];x0=min(e['x'] for e in entrances);x1=max(e['x']+e['extent'] for e in entrances)
    z0=min(e['z'] for e in entrances)-1;z1=max(e['z']+e['extent'] for e in entrances)+3
    y0=min(e['y'] for e in entrances)-2;y1=max(e['y']+e['extent'] for e in entrances)+2
    mid=(x0+x1)//2
    compact=empty.copy();compact[z0:z1,y0:y1,x0:x1]=True
    articulated=empty.copy()
    articulated[z0:z1-1,y0:y1,x0:mid+2]=True
    articulated[z0+1:z1,y0+1:y1,mid-2:x1]=True
    thin=empty.copy();thin[z0+1,y0:y1,x0:x1]=True
    fragmented=compact.copy();fragmented[:,:,mid:mid+2]=False
    neck=fragmented.copy();neck[z0+1,y0+2,mid:mid+2]=True
    unsupported=empty.copy();unsupported[z0:z1,y0:y1,x0+3:x1-3]=True
    collision=compact.copy();collision[z0:z0+3,y0:y0+3,x0-1]=True
    ground=compact.copy();ground[0,0,mid]=True
    satellite=compact.copy();satellite[min(z1+2,29):min(z1+2,29)+2,y0:y0+2,mid:mid+2]=True
    spill=compact.copy();spill[26:29,13:16,mid:mid+3]=True
    return {'compact_mass':compact,'articulated_mass':articulated,'empty':empty,'thin_sheet':thin,
            'fragmented':fragmented,'thin_neck':neck,'unsupported':unsupported,'context_collision':collision,
            'ground_intrusion':ground,'detached_satellite':satellite,'outside_domain':spill,'excessive_fill':domain.copy()}
