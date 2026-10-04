"""spatial_platform_v1: deterministic, single-level geometric research prototype.

Derived surfaces and clearance are evaluated independently of construction.
No trained weights, change to historical objectives, or mechanical validation.
"""
from collections import deque
from dataclasses import asdict, dataclass
import numpy as np
from nca.contract import validate_scene, declared_existing
from nca.evaluation import geometric_support


@dataclass(frozen=True)
class PlatformSpec:
    version: str = 'spatial_platform_v1'
    surface_z: int = 8
    width_cells: int = 3
    landing_cells: int = 5
    headroom_cells: int = 3
    floor_depth_cells: int = 1

    def validate(self, scene):
        if self.version != 'spatial_platform_v1':
            raise ValueError('Unknown spatial contract')
        for name, value in asdict(self).items():
            if name != 'version' and (type(value) is not int or value < 1):
                raise ValueError('Spatial dimensions must be positive integers')
        if self.width_cells % 2 != 1 or self.landing_cells % 2 != 1:
            raise ValueError('Footprint dimensions must be odd')
        if self.landing_cells < self.width_cells:
            raise ValueError('Landing must contain the required footprint')
        if self.surface_z < self.floor_depth_cells or self.surface_z + self.headroom_cells > scene['grid_size']:
            raise ValueError('Floor and headroom must fit inside the grid')


def layout(scene, spec=PlatformSpec()):
    scene = validate_scene(scene); spec.validate(scene)
    entrances = sorted(scene['entrances'], key=lambda e:e['x'])
    if len(entrances) != 2 or any(e['kind'] != 'facade' for e in entrances):
        raise ValueError('Prototype requires exactly two facade approach regions')
    west, east = entrances
    if west['y'] != east['y'] or east['x'] <= west['x']:
        raise ValueError('Prototype requires approaches aligned along X')
    if any(e['z'] != spec.surface_z for e in entrances):
        raise ValueError('Different approach levels require another routing method')
    # Regions remain the original 2-cube scene markers. They designate external
    # approaches, not physical doors or a path inside a solid context building.
    x0, x1, cy = west['x'], east['x']+east['extent'], west['y']
    cx = (x0+x1-1)//2
    r = spec.landing_cells//2
    if cx-r < x0 or cx+r >= x1:
        raise ValueError('The declared landing does not fit between approaches')
    if not 0 <= cy-r <= cy+r < scene['grid_size']:
        raise ValueError('Landing crosses the site boundary')
    return scene, (x0,x1,cy,cx), entrances


def construct_platform(scene, spec=PlatformSpec()):
    """A proposed straight deck; do not conceal collisions by clipping them."""
    scene = validate_scene(scene); spec.validate(scene)
    material = np.zeros((scene['grid_size'],)*3, dtype=bool)
    try:
        scene, (x0,x1,cy,cx), _ = layout(scene,spec)
    except ValueError as error:
        return material, {'status':'unsupported_layout','reason':str(error)}
    r=spec.width_cells//2; lr=spec.landing_cells//2
    slab=material[spec.surface_z-spec.floor_depth_cells:spec.surface_z]
    slab[:,cy-r:cy+r+1,x0:x1]=True
    slab[:,cy-lr:cy+lr+1,cx-lr:cx+lr+1]=True
    return material, {'status':'candidate','reason':'Raw proposed deck; independent evaluation determines compatibility'}


def full_footprint(mask, width):
    """Square footprint erosion; outside the site is false, never wrapped."""
    radius=width//2; padded=np.pad(mask,radius,constant_values=False)
    out=np.ones_like(mask,dtype=bool)
    for dy in range(width):
        for dx in range(width):
            out &= padded[dy:dy+mask.shape[0],dx:dx+mask.shape[1]]
    return out


def reachable(mask, seeds):
    reached=mask & seeds
    queue=deque(map(tuple,np.argwhere(reached)))
    while queue:
        y,x=queue.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<mask.shape[0] and 0<=xx<mask.shape[1] and mask[yy,xx] and not reached[yy,xx]:
                reached[yy,xx]=True;queue.append((yy,xx))
    return reached


def evaluate_platform(scene, material, spec=PlatformSpec()):
    scene=validate_scene(scene);spec.validate(scene)
    n=scene['grid_size'];material=np.asarray(material)
    if material.shape!=(n,n,n) or material.dtype!=bool:
        raise ValueError('Material must be a matching boolean (z,y,x) grid')
    existing=declared_existing(scene); occupied=material|existing
    z=spec.surface_z
    # Only proposed material counts as the declared floor. Context boxes cannot
    # supply fictional interior floors. Clearance includes both solid fields.
    floor=material[z-spec.floor_depth_cells:z].all(axis=0)
    clear=~occupied[z:z+spec.headroom_cells].any(axis=0)
    surface=floor & clear
    centers=full_footprint(surface,spec.width_cells)
    seed=np.zeros((n,n),bool);target=seed.copy();landing=seed.copy()
    reason=None
    try:
        _,(_,_,cy,cx),entrances=layout(scene,spec)
        for mask,e in zip((seed,target),entrances):
            mask[e['y']:e['y']+e['extent'],e['x']:e['x']+e['extent']]=True
        r=spec.landing_cells//2;landing[cy-r:cy+r+1,cx-r:cx+r+1]=True
    except ValueError as error:
        reason=str(error)
    reached=reachable(centers,seed)
    connected=bool((reached&target).any()) and reason is None
    landing_clear=bool(landing.any() and surface[landing].all())
    # Landing must belong to the same accessible center graph as the approaches.
    landing_accessible=landing_clear and bool((reached&full_footprint(landing,spec.width_cells)).any())
    support=geometric_support(material,existing)
    legal=not bool((material&existing).any())
    # The standalone geometric gate also enforces an entirely open street band.
    # Historical anchor exceptions are reported separately by legacy evaluation.
    street_clear=not bool(material[:scene['street_levels']].any())
    supported=bool(material.any()) and support['unsupported_voxels']==0
    gate=connected and landing_accessible and legal and street_clear and supported
    s=scene['voxel_size_m']
    report={'version':spec.version,'spec':asdict(spec),'layout_supported':reason is None,
        'layout_reason':reason,'surface_elevation_m':z*s,'required_width_m':spec.width_cells*s,
        'required_headroom_m':spec.headroom_cells*s,'floor_depth_m':spec.floor_depth_cells*s,
        'landing_side_m':spec.landing_cells*s,'material_voxels':int(material.sum()),
        'floor_area_m2':int(floor.sum())*s*s,'clear_surface_area_m2':int(surface.sum())*s*s,
        'landing_area_m2':int(landing.sum())*s*s,
        'width_qualified_centers':int(centers.sum()),'floor_cells_with_insufficient_clearance':int((floor&~clear).sum()),
        'approach_connected':connected,'landing_clear':landing_clear,'landing_accessible':landing_accessible,
        'material_context_collision_voxels':int((material&existing).sum()),
        'street_band_clear':street_clear,'geometric_support':support,
        'spatial_gate':gate,
        'interpretation':'Single-level external approach connectivity for a square footprint, clear landing, collision-free floor and geometric attachment. Not interior access, stairs, accessibility-code compliance or structural safety.'}
    surface_3d=np.zeros_like(material);surface_3d[z]=surface
    center_3d=np.zeros_like(material);center_3d[z]=centers
    void=np.zeros_like(material);void[z:z+spec.headroom_cells]=surface[None]
    return report, {'surface':surface_3d,'centers':center_3d,'clearance':void,'landing_xy':landing}


def legacy_diagnostics(scene, material, config):
    """Evaluate the unchanged nine-family objective on arbitrary binary material.

    Fixed centerline/envelope is derived from the scene, never the proposed deck.
    The old connectivity target is occupied entrance blocks; floor-under-void
    semantics intentionally remain different and are reported without repair.
    """
    import torch
    from deploy.model_utils import UrbanSceneGenerator
    from nca.contract import fields_from_state,to_generator_params
    from nca.legal_corridor import route_legal_corridor
    from nca.losses import material_envelope,context_from_scenes
    from nca.facade import endpoint_allowance
    from nca.objective import research_terms
    from nca.evaluation import endpoint_connectivity,material_legality
    with torch.inference_mode():
        state,_=UrbanSceneGenerator(dict(config)).generate(to_generator_params(scene))
        fields=fields_from_state(state,config,scene)
        route=route_legal_corridor(fields['permitted'],fields['endpoints'])
        guide=torch.from_numpy(route['centerline'])[None]
        permitted=torch.from_numpy(fields['permitted'])[None]
        envelope=material_envelope(guide,permitted,6)
        context=context_from_scenes(state,config,[scene],guide,envelope,torch.tensor([route['report']['all_endpoints_connected']]))
        allowance,_=endpoint_allowance(scene,permitted)
        p=torch.from_numpy(material.copy()).float()[None]
        state[:,config['ch_structure']]=p
        terms=research_terms(state,p,context,config,allowance)
        connectivity=endpoint_connectivity(material&fields['permitted'],fields['endpoints'],sorted(fields['endpoints'])[0])
        ratio=float(terms['mass_ratio'][0]);in_budget=.03-1e-6<=ratio<=.12+1e-6
        return {'objective_version':'research_objective_v1','metric_version':'binary_v1',
            'families':{k:float(v[0]) for k,v in terms['terms'].items()},
            'material_ratio':ratio,'in_budget':in_budget,'envelope_voxels':int(envelope.sum()),
            'context_valid':bool(terms['context_valid'][0]),'connectivity':connectivity,
            'joint_budget_connectivity':bool(terms['context_valid'][0]) and connectivity['all_connected'] and in_budget,
            'legality':material_legality(material,fields['permitted'])}
