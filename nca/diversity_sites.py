"""ED1 designed development contexts; unchanged massing constraints."""
from copy import deepcopy
from nca.scale_study import scale_sites
from nca.contract import validate_scene


def diversity_sites():
    base=next(s['scene'] for s in scale_sites(48) if s['kind']=='compact')
    result=[]
    for kind,label in [('split_heights','Different connection heights'),('staggered','Staggered obstacles'),('overhead','Overhead crossing'),('asymmetric','Asymmetric frontages')]:
        scene=deepcopy(base)
        scene['scene_id']='ed1-48-'+kind
        scene['description']='ED1 designed development site: '+label
        scene['notes']=['Overall building mass; interiors deferred. Development case, not held-out evaluation.']
        if kind=='split_heights':
            scene['entrances'][0].update(z=10,y=18)
            scene['entrances'][1].update(z=24,y=27)
        elif kind=='staggered':
            scene['entrances'][0].update(y=18)
            scene['entrances'][1].update(y=28)
            scene['buildings'] += [
                dict(id='B_obstacle_a',x=[18,21],y=[12,24],z=[6,26],side=None,gap_facing_x=None),
                dict(id='B_obstacle_b',x=[28,31],y=[25,37],z=[6,26],side=None,gap_facing_x=None)]
        elif kind=='overhead':
            scene['buildings'].append(dict(id='B_crossing',x=[8,40],y=[22,26],z=[18,22],side=None,gap_facing_x=None))
        else:
            scene['buildings'][0].update(y=[12,32],z=[0,28])
            scene['buildings'][1].update(y=[18,42],z=[0,38])
            scene['entrances'][0].update(y=17,z=12)
            scene['entrances'][1].update(y=30,z=16)
        result.append(dict(case='ed1_48__'+kind,label=label+' / 48 cubed',scene=validate_scene(scene)))
    return result
