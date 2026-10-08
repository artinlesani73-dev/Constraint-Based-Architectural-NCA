"""Verified run evidence and exact exposed-voxel OBJ surfaces, in metres."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np

def obj(field,scale):
    # Outward-wound quads, XYZ world coordinates, Z up; array is ZYX.
    faces=[((-1,0,0),[(0,0,0),(0,0,1),(0,1,1),(0,1,0)]),
           ((1,0,0),[(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
           ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
           ((0,1,0),[(0,1,0),(0,1,1),(1,1,1),(1,1,0)]),
           ((0,0,-1),[(0,0,0),(0,1,0),(1,1,0),(1,0,0)]),
           ((0,0,1),[(0,0,1),(1,0,1),(1,1,1),(0,1,1)])]
    vertices={};quads=[];nz,ny,nx=field.shape
    for z,y,x in np.argwhere(field):
        for (dx,dy,dz),corners in faces:
            xx,yy,zz=x+dx,y+dy,z+dz
            if 0<=xx<nx and 0<=yy<ny and 0<=zz<nz and field[zz,yy,xx]:continue
            ids=[]
            for a,b,c in corners:
                v=(int(x+a),int(y+b),int(z+c))
                if v not in vertices:vertices[v]=len(vertices)+1
                ids.append(vertices[v])
            quads.append(ids)
    lines=['# NCA exact voxel boundary; units metres; XYZ; Z up; no smoothing']
    lines += ['v '+' '.join(f'{c*scale:.8f}' for c in v) for v in vertices]
    lines += ['f '+' '.join(map(str,q)) for q in quads]
    return ('\n'.join(lines)+'\n').encode()

def bundle(folder):
    folder=Path(folder)
    if not (folder/'status.json').is_file():raise FileNotFoundError(folder)
    if not (folder/'manifest.json').is_file():raise ValueError('Run is still being saved; retry after completion.')
    manifest=json.loads((folder/'manifest.json').read_text())
    raw={}
    for name,h in manifest.items():
        if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Invalid run manifest path')
        p=folder/name
        if p.is_symlink() or not p.is_file():raise ValueError('Run evidence is missing or redirected')
        value=p.read_bytes()
        if hashlib.sha256(value).hexdigest()!=h:raise ValueError('Run evidence hash differs; export stopped')
        raw[name]=value
    required={'status.json','request.json','scene.json','identity.json','result.json'}
    if not required<=raw.keys():raise ValueError('Run has no complete result to export')
    status=json.loads(raw['status.json']);result=json.loads(raw['result.json']);scene=json.loads(raw['scene.json']);identity=json.loads(raw['identity.json'])
    if status['state'] not in ['completed','failed_checks']:raise ValueError('Run is not finished')
    payload={'evidence/'+n:v for n,v in raw.items()}
    payload['evidence/manifest.json']=(folder/'manifest.json').read_bytes()
    outputs={}
    for label in ['G10','R3']:
        accepted=all('record' in result['outputs'][f'{label}-{s}'] and result['outputs'][f'{label}-{s}']['record']['score']['contract_pass'] and result['outputs'][f'{label}-{s}']['record']['volume_error']<=.04 for s in [64,128]) and result['stability'].get(label,float('inf'))<=.05
        outputs[label]={'accepted_both_horizons':accepted,'stages':{}}
        for step in [64,128]:
            key=f'{label}-{step}';entry=result['outputs'][key]
            if 'error' in entry:
                outputs[label]['stages'][str(step)]={'available':False,'reason':entry['error']};continue
            filename=key+'.npz'
            if filename not in raw:raise ValueError('Saved voxel field is missing')
            with np.load(io.BytesIO(raw[filename]),allow_pickle=False) as archive:field=archive['field']
            if field.shape!=(scene['grid_size'],)*3 or field.dtype!=np.bool_:raise ValueError('Invalid saved field')
            if np.flatnonzero(field).tolist()!=entry['voxels']:raise ValueError('Viewer and saved geometry disagree')
            payload['geometry/'+key+'.obj']=obj(field,scene['voxel_size_m'])
            outputs[label]['stages'][str(step)]={'available':True,'occupied_voxels':int(field.sum()),'measured':entry['record']}
    meta=dict(format='nca_run_export_v1',run_id=result['id'],status=status,units='metres',coordinate_axes='XYZ, Z up',array_axes='ZYX',cell_extent='[i*s,(i+1)*s]',scene=scene,request=json.loads(raw['request.json']),model_sha256=identity['files']['model/checkpoint-0427.pt'],source_identity=identity,outputs=outputs,geometry='Generated mass only; context buildings remain in scene.json. Exact exposed voxel faces; no smoothing or filling.',limitations='Experimental building mass, not construction or structural certification. R3 uses learned scores plus a planner and guards. OBJ carries no intrinsic unit declaration; import with metres. Keep this metadata with the geometry.')
    payload['metadata.json']=(json.dumps(meta,indent=2)+'\n').encode()
    payload['README.txt']=('NCA saved run '+result['id']+'\nImport geometry/*.obj using METRES and Z UP. Geometry contains generated mass only.\nMetadata.json contains exact scene, seed, model identity and separate G10/R3 acceptance.\nFailed outputs remain present and labelled; unavailable R3 has no invented OBJ.\nThe evidence folder preserves saved inputs, trajectories, metrics and original hashes.\nThe export manifest covers all payloads except itself. Keep the full package.\nExperimental voxel surface; no smoothing, structural certification or guaranteed manifold CAD solid.\n').encode()
    payload['export-manifest.json']=json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},indent=2).encode()
    target=io.BytesIO()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in payload.items():archive.writestr(name,value)
    return target.getvalue()
