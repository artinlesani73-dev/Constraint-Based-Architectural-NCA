"""volumetric_audit_v1: descriptive solid/void diagnostics, never a quality score."""
import math
import numpy as np
from nca.evaluation import flood_fill

VERSION='volumetric_audit_v1'


def boolean_grid(value):
    a=np.asarray(value)
    if a.ndim!=3 or a.dtype!=np.bool_ or any(n==0 for n in a.shape):
        raise ValueError('Expected a nonempty boolean (z,y,x) grid')
    return a


def exterior_free(occupied):
    occupied=boolean_grid(occupied)
    seeds=np.zeros_like(occupied)
    for axis in range(3):
        for side in (0,-1):
            sl=[slice(None)]*3;sl[axis]=side;seeds[tuple(sl)]=True
    return flood_fill(~occupied,seeds)


def bracket_axes(material):
    material=boolean_grid(material)
    counts=np.zeros(material.shape,dtype=np.uint8)
    for axis in range(3):
        forward=np.maximum.accumulate(material,axis=axis)
        reverse=np.flip(np.maximum.accumulate(np.flip(material,axis=axis),axis=axis),axis=axis)
        counts+=(forward&reverse&~material).astype(np.uint8)
    return counts


def free_cube_centers(free,width):
    free=boolean_grid(free)
    if type(width) is not int or width<1 or width%2!=1:
        raise ValueError('Cube width must be a positive odd integer')
    if width>min(free.shape):return np.zeros_like(free)
    r=width//2;p=np.pad(free,r,constant_values=False);out=np.ones_like(free)
    for z in range(width):
        for y in range(width):
            for x in range(width):
                out &= p[z:z+free.shape[0],y:y+free.shape[1],x:x+free.shape[2]]
    return out


def components(material):
    remaining=material.copy();count=0
    while remaining.any():
        seed=np.zeros_like(remaining);seed[tuple(np.argwhere(remaining)[0])]=True
        remaining &= ~flood_fill(remaining,seed);count+=1
    return count


def measure_volume(material,existing,region,voxel_size_m=.8):
    material,existing,region=map(boolean_grid,(material,existing,region))
    if material.shape!=existing.shape or material.shape!=region.shape or not region.any():
        raise ValueError('Matching grids and a nonempty fixed analysis region required')
    if isinstance(voxel_size_m,bool) or not isinstance(voxel_size_m,(float,int)) or not math.isfinite(voxel_size_m) or voxel_size_m<=0:
        raise ValueError('Voxel size must be finite and positive')
    free=~(material|existing)
    brackets=bracket_axes(material)
    two=(brackets>=2)&free&region;three=(brackets==3)&free&region
    sealed_form=(~material & ~exterior_free(material)) & free & region
    exterior=exterior_free(material|existing)
    sealed_context=free & ~exterior & region
    coords=np.argwhere(material)
    spans=(coords.max(0)-coords.min(0)+1).tolist() if len(coords) else [0,0,0]
    bounds=np.stack((coords.min(0),coords.max(0)+1),axis=1).tolist() if len(coords) else None
    volume=voxel_size_m**3
    clearance={str(w):int((two&free_cube_centers(free,w)).sum()) for w in (1,3,5)}
    report={'version':VERSION,'voxel_size_m':voxel_size_m,'material_voxels':int(material.sum()),
        'material_volume_m3':int(material.sum())*volume,'bbox_zyx':bounds,
        'extent_cells_zyx':spans,'extent_m_zyx':[n*voxel_size_m for n in spans],
        'extent_balance':min(spans)/max(spans) if max(spans) else None,
        'material_components_6':components(material),'context_collision_voxels':int((material&existing).sum()),
        'analysis_region_voxels':int(region.sum()),'bracketed_2_axes_voxels':int(two.sum()),
        'bracketed_3_axes_voxels':int(three.sum()),'bracketed_2_axes_volume_m3':int(two.sum())*volume,
        'sealed_by_form_voxels':int(sealed_form.sum()),'sealed_with_context_voxels':int(sealed_context.sum()),
        'bracketed_exterior_connected_voxels':int((two&exterior).sum()),
        'free_cube_centers_in_bracketed_void':clearance,
        'interpretation':'Axis-bracketed emptiness and grid topology are descriptive proxies, not an architectural validity score, room definition or walkability test.'}
    return report,{'bracketed':two,'bracketed_three':three,'sealed_form':sealed_form,'sealed_context':sealed_context}


def fixed_probes():
    """Exact diagnostic fields; all constructed without looking at loss values."""
    shape=(32,32,32)
    empty=np.zeros(shape,bool)
    slab=empty.copy();slab[7:9,8:24,8:24]=True
    solid=empty.copy();solid[7:15,12:20,12:20]=True
    tube=empty.copy();tube[7:16,11:20,8:24]=True;tube[8:15,12:19,8:24]=False
    aperture=tube.copy();aperture[9:14,11,13:18]=False
    closed=tube.copy();closed[8:15,12:19,8]=True;closed[8:15,12:19,23]=True
    decoy=empty.copy()
    for z in (7,15):
        for y in (11,19):
            for x in (8,23):decoy[z,y,x]=True
    return {'empty':empty,'slab_512':slab,'solid_512':solid,'open_ends_512':tube,
            'side_aperture_487':aperture,'closed_shell_610':closed,'extent_decoy_8':decoy}
