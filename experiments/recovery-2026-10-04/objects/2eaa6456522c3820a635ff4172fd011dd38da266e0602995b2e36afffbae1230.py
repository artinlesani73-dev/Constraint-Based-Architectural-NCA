"""Loopback-only experimental R3 generation; one CPU job at a time."""
from pathlib import Path
import json,sys,hashlib,threading,uuid,datetime,traceback,platform
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'source'))
from context_route import route_from_context
from r3_route_options import route_via
from g11_reservation import witness,rollout
from nca.generation_data import seed_inputs
from nca.repair_training import perceive
from nca.repair_benchmark import condition
from nca.paced_generation import PacedNCA
from nca.repair_portable import read_portable
from nca.massing_cases import target_context
from nca.massing_targets import evaluate_targets,neighbors
from nca.contract import entrance_masks,validate_scene
from nca.budget_reference import budget
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2),encoding='utf-8');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
identity=json.loads((ROOT/'identity.json').read_text())
for name,digest in identity['files'].items():
 if sha(ROOT/name)!=digest:raise RuntimeError('Frozen source hash differs: '+name)
model=PacedNCA();model.load_state_dict(read_portable(ROOT/'model/checkpoint-0427.pt',json.loads((ROOT/'model/identity.json').read_text()))['model']);model.eval()
config=json.loads((ROOT/'config.json').read_text());scenes=json.loads((ROOT/'scenes.json').read_text());byid={s['scene_id']:s for s in scenes};lock=threading.Lock()
for p in RUNS.glob('*/status.json'):
 old=json.loads(p.read_text())
 if old['state'] in ['queued','running']:
  write(p.parent/'interrupted-status.json',old);write(p,dict(id=old['id'],state='interrupted',message='Server stopped before completion. Evidence retained; start a new run to retry.',updated=now()))
def generate(runid,request):
 p=RUNS/runid
 def status(state,message):write(p/'status.json',dict(id=runid,state=state,message=message,updated=now()))
 try:
  status('running','Preparing the site and checking the planner certificate.')
  s=request.get('custom_scene') or byid[request['scene']];req=request['volume'];seed=request['seed'];fields,domain,_=target_context(s,config);c=condition(s,fields,domain,req);x=seed_inputs(c);B,C=budget(int(domain.sum()),req,3)
  np.savez_compressed(p/'context.npz',condition=c);mode=request.get('route_mode','original');route,route_meta=(route_from_context(c),{}) if mode=='original' else route_via(c,mode);write(p/'route.json',dict(mode=mode,details=route_meta));W=None;cert=dict(certified=False,reason='No legal thick route was found.')
  if route is not None:
   W,plan=witness(route,x['allowed'],s,fields,domain,C);score,_=evaluate_targets(W,s,fields,domain);cert=dict(certified=bool(score['contract_pass'] and W.sum()<=C),score=score,plan=plan,reason='Planner witness evaluated against all nine families and the volume cap.')
   np.savez_compressed(p/'witness.npz',field=W)
  write(p/'certificate.json',cert)
  status('running','Generating the raw G10 comparison and R3 hybrid.')
  with torch.no_grad():raw=model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(seed),steps=128,capture=True)
  rb=raw['births'].numpy()[:,0,0];np.savez_compressed(p/'G10-trajectory.npz',births=rb,state128=raw['state'].numpy());hy=None
  if cert['certified']:
   contact=neighbors(fields['existing'].astype(bool),diagonal=True)&~fields['existing'].astype(bool);face=neighbors(fields['existing'].astype(bool));ends=entrance_masks(s)
   for e in s['entrances']:
    if e['kind']=='facade':contact &= ~(ends[e['id']]&face&x['allowed'])
   hy=rollout(model,torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],W,contact,firing_seed=seed)
   np.savez_compressed(p/'R3-trajectory.npz',births=hy['births'],provenance=hy['provenance'],state64=hy['states'][64],state128=hy['states'][128]);write(p/'trace.json',hy['trace'])
  result=dict(route_mode=mode,id=runid,cohort='generated',scene=s,request=req,seed=seed,outputs={},certificate=cert,stability={})
  for label in ['G10','R3']:
   fs={}
   for step in [64,128]:
    if label=='R3' and hy is None:
     result['outputs'][f'{label}-{step}']=dict(error='No certified R3 output. The planner could not satisfy the frozen rules.');continue
    f=x['occupancy'].astype(bool)|rb[:step].any(0) if label=='G10' else hy['states'][step][0,0].astype(bool);fs[step]=f
    score,_=evaluate_targets(f,s,fields,domain);ids=np.flatnonzero(f).tolist();labels=[0 if x['occupancy'].reshape(-1)[i] else 1 for i in ids] if label=='G10' else np.max(hy['provenance'][:step],axis=0).reshape(-1)[ids].tolist()
    rec=dict(score=score,volume_error=float(abs(f.sum()/domain.sum()-req)),planner_voxels=labels.count(2),learned_voxels=labels.count(1))
    result['outputs'][f'{label}-{step}']=dict(voxels=ids,labels=labels,record=rec);np.savez_compressed(p/f'{label}-{step}.npz',field=f)
   if fs:result['stability'][label]=float((fs[128].sum()-fs[64].sum())/fs[64].sum())
  write(p/'result.json',result)
  passed=hy is not None and all(result['outputs'][f'R3-{n}']['record']['score']['contract_pass'] and result['outputs'][f'R3-{n}']['record']['volume_error']<=.04 for n in [64,128]) and result['stability']['R3']<=.05
  status('completed' if passed else 'failed_checks','Saved. R3 passed the per-run checks.' if passed else 'Saved with failed checks. Inspect the comparison; this run is not accepted.')
 except Exception:
  (p/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');status('error','Generation stopped. Inputs and error details are retained locally.')
 finally:
  write(p/'manifest.json',{f.name:sha(f) for f in p.iterdir() if f.is_file() and f.name not in ['manifest.json']});lock.release()
class Handler(BaseHTTPRequestHandler):
 def reply(self,code,obj):
  raw=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
 def validhost(self):return self.headers.get('Host') in ['127.0.0.1:8018','localhost:8018']
 def do_GET(self):
  if not self.validhost():return self.reply(403,{'error':'Local host only'})
  path=self.path.split('?')[0]
  if path=='/api/drafts':return self.reply(200,[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'drafts').glob('*.json'),reverse=True)])
  if path=='/api/scenes':return self.reply(200,scenes)
  if path=='/api/runs':return self.reply(200,[dict(json.loads(p.read_text()),request=json.loads((p.parent/'request.json').read_text())) for p in sorted(RUNS.glob('*/status.json'),reverse=True)])
  if path.startswith('/api/run/'):
   rid=path.removeprefix('/api/run/')
   if len(rid)!=32 or any(c not in '0123456789abcdef' for c in rid):return self.reply(404,{'error':'Unknown run'})
   p=RUNS/rid
   if not (p/'status.json').exists():return self.reply(404,{'error':'Unknown run'})
   return self.reply(200,dict(request=json.loads((p/'request.json').read_text()),status=json.loads((p/'status.json').read_text()),result=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else None))
  name={'/':'index.html','/data.js':'data.js','/studio.js':'studio.js','/editor.js':'editor.js','/revisit.js':'revisit.js','/studio_skins.js':'studio_skins.js','/studio_skins.css':'studio_skins.css'}.get(path)
  if not name:return self.reply(404,{'error':'Not found'})
  raw=(ROOT/name).read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8' if name.endswith('html') else 'text/css; charset=utf-8' if name.endswith('css') else 'text/javascript; charset=utf-8');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
 def do_POST(self):
  if not self.validhost() or self.headers.get('Origin') not in [None,'http://127.0.0.1:8018','http://localhost:8018'] or self.headers.get('X-NCA-Local')!='1':return self.reply(403,{'error':'Local studio requests only'})
  if self.path=='/api/drafts':
   try:
    size=int(self.headers.get('Content-Length','0'))
    if not 0<size<=16384:raise ValueError('Draft too large')
    value=json.loads(self.rfile.read(size))
    if not isinstance(value,dict) or value.get('site') not in byid or not isinstance(value.get('values'),dict) or not isinstance(value.get('base'),dict):raise ValueError('Invalid draft')
    rid=uuid.uuid4().hex;folder=ROOT/'drafts';folder.mkdir(exist_ok=True);saved=dict(id=rid,saved_at=now(),draft=value);write(folder/(rid+'.json'),saved);return self.reply(201,saved)
   except (ValueError,TypeError):return self.reply(400,{'error':'Invalid draft. Nothing saved.'})
  if self.path!='/api/generate':return self.reply(404,{'error':'Not found'})
  try:
   size=int(self.headers.get('Content-Length','0'))
   if not 0<size<=16384:raise ValueError('Invalid request size')
   req=json.loads(self.rfile.read(size))
   mode=req.pop('route_mode','original')
   if mode not in ['original','low_y','high_y']:raise ValueError('Unknown route mode')
   parent=req.pop('parent_run',None)
   if parent is not None:
    if not isinstance(parent,str) or len(parent)!=32 or any(c not in '0123456789abcdef' for c in parent) or not (RUNS/parent/'request.json').exists():raise ValueError('Parent run not found')
   if set(req) not in [{'scene','volume','seed'},{'scene','volume','seed','custom_scene'}] or req['scene'] not in byid or type(req['seed']) is not int or not 0<=req['seed']<=2147483647 or type(req['volume']) not in [int,float] or req['volume'] not in [.16,.24,.32]:raise ValueError('Choose a listed site, 16/24/32% volume and an integer seed from 0 to 2147483647.')
   if 'custom_scene' in req:
    custom=req['custom_scene']
    if not isinstance(custom,dict) or custom.get('grid_size')!=32 or custom.get('voxel_size_m')!=.8 or custom.get('street_levels')!=6 or custom.get('ceiling_z') is not None or custom.get('legacy_relaxations')!=[]:raise ValueError('Custom sites retain the32-cell grid,0.8m voxels,6-cell street band and no relaxations.')
    if len(custom.get('buildings',[])) not in [2,3] or len(custom.get('entrances',[]))!=2:raise ValueError('Use two facing buildings, two connections and at most one obstacle.')
    custom=validate_scene(custom)
    left,right=custom['buildings'][:2];west,east=custom['entrances']
    if left['side']!='left' or right['side']!='right' or left['x'][1]>=right['x'][0] or left['gap_facing_x']!=left['x'][1] or right['gap_facing_x']!=right['x'][0]:raise ValueError('Buildings must face across a positive X gap.')
    if west['id']!='E_west' or east['id']!='E_east' or west['x']!=left['x'][1] or east['x']!=right['x'][0]-2 or any(e['kind']!='facade' or e['extent']!=2 for e in [west,east]):raise ValueError('Use the two supported facade connections.')
    for b,e in [(left,west),(right,east)]:
     if not(b['y'][0]<=e['y'] and e['y']+2<=b['y'][1] and b['z'][0]<=e['z'] and e['z']+2<=b['z'][1]):raise ValueError('Each connection must fit fully on its building facade.')
    req['custom_scene']=custom
  except (ValueError,TypeError,KeyError,AttributeError) as error:return self.reply(400,{'error':str(error) or 'Invalid site, volume or seed.'})
  req['route_mode']=mode
  if parent is not None:req['parent_run']=parent
  if not lock.acquire(blocking=False):return self.reply(409,{'error':'One run is already active. Wait for its result.'})
  try:
   rid=uuid.uuid4().hex;p=RUNS/rid;p.mkdir();write(p/'request.json',req);write(p/'scene.json',req.get('custom_scene') or byid[req['scene']]);write(p/'identity.json',identity);write(p/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=2));write(p/'status.json',dict(id=rid,state='queued',message='Queued locally.',updated=now()));threading.Thread(target=generate,args=(rid,req),daemon=True).start()
  except Exception:lock.release();raise
  self.reply(202,{'id':rid})
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8018),Handler).serve_forever()
