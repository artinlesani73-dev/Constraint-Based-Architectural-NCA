from pathlib import Path
p=Path('review_g6.py').read_text()
p=p[:p.index('# Verify pairing against')]
p=p.replace("PACKAGE=BASE/'G6-Paced-Growth-2026-10-04';OUT=BASE/'G6-Final-Review-2026-10-04'","PACKAGE=BASE/'G7-Vertical-Training-2026-10-04-v2';OUT=BASE/'G7-Final-Review-2026-10-04'")
p=p.replace('20261004T081739Z_95ddbf523738','20261004T103602Z_1da578905a51').replace('NCA-G6-Paced-Package.zip','NCA-G7-Vertical-Package.zip').replace('seed_generation_training_v6_paced','seed_generation_training_v7_vertical').replace('TrainingOrder(27,1203)','TrainingOrder(45,1203)')
p += '''
sys.dont_write_bytecode=True
from nca.repair_benchmark import condition,context_hash
from nca.contract import entrance_masks
from nca.generation_training import generate
protocol=json.loads((PACKAGE/'frozen-review.json').read_text())
assert sha((PREP/'split-manifest.json').read_bytes())==protocol['regression_manifest_sha256']
assert sha((PACKAGE/'split-manifest.json').read_bytes())==protocol['fresh_split_sha256']
for name in ['frozen-review.json','split-manifest.json','environment.json','package-receipt.json']:
 shutil.copyfile(PACKAGE/name,OUT/name)
shutil.copyfile(PREP/'split-manifest.json',OUT/'regression-split.json')
oldsplit=json.loads((PREP/'split-manifest.json').read_text());newsplit=json.loads((PACKAGE/'split-manifest.json').read_text())
entries=[{**e,'cohort':'regression'} for e in oldsplit['entries'] if e['split'] in protocol['regression_splits']]+[{**e,'cohort':'fresh_reserved'} for e in newsplit['entries'] if e['split']=='reserved']
assert len(entries)==11
config=json.loads((PACKAGE/'environment.json').read_text())['config']
# Input construction checked against every packaged TRAIN fixture before reserved inference.
allscenes={e['id']:e['scene'] for e in oldsplit['entries']+newsplit['entries']}
with zipfile.ZipFile(PACKAGE/'NCA-G7-Vertical-Package.zip') as z:
 for row in train_data['rows']:
  scene=allscenes[row['id'].rsplit('-v',1)[0]];fields,domain,_=target_context(scene,config)
  with np.load(io.BytesIO(z.read(row['arrays'])),allow_pickle=False) as a:
   c=condition(scene,fields,domain,float(a['condition'][6,0,0,0]))
   assert c.tobytes()==a['condition'].tobytes() and np.array_equal(seed_inputs(c)['occupancy'],a['damaged'])
save('context-verification.json',dict(count=45,all_context_bytes_and_seeds_equal=True,before_fresh_reserved_inference=True))
g6=BASE/'G6-Final-Review-2026-10-04'
with zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as z:initial=torch.load(io.BytesIO(z.read('worker/checkpoint-0000.pt')),map_location='cpu',weights_only=False)['model']
with zipfile.ZipFile(g6/'20261004T081739Z_95ddbf523738.zip') as z:oldinitial=torch.load(io.BytesIO(z.read('worker/checkpoint-0000.pt')),map_location='cpu',weights_only=False)['model']
assert initial.keys()==oldinitial.keys() and all(torch.equal(initial[k],oldinitial[k]) for k in initial)
assert (source/'nca/paced_generation.py').read_bytes()==(g6/'source/nca/paced_generation.py').read_bytes()
save('pairing.json',dict(initial_parameters_equal_g6=True,model_and_loss_source_equal_g6=True,training_distribution_changed=True,per_row_exposure_changed=True))
model=PacedNCA().float();model.load_state_dict(p['model'],strict=True);model.eval()
save('execution.json',dict(protocol=protocol,runtime=runtime(torch.device('cpu')),receipt=receipt,verified_payloads=len(m),checkpoint_sha256=sha((OUT/'import/worker/checkpoint-0256.pt').read_bytes()),run_result=run_result,recovery=recovery))
observations=[];stability=[]
for e in entries:
 scene=e['scene'];fields,domain,_=target_context(scene,config)
 assert context_hash(scene,fields,domain)==e['context_sha256']
 for request in protocol['requests']:
  case=e['id']+f'-v{round(request*100)}';c=condition(scene,fields,domain,request);x=seed_inputs(c);outputs={}
  (OUT/'contexts').mkdir(exist_ok=True)
  with (OUT/f'contexts/{case}.npz').open('xb') as f:np.savez_compressed(f,context=c,seed=x['occupancy'])
  for steps in protocol['horizons']:
   tick=time.perf_counter()
   with torch.no_grad():r=model.rollout(torch.from_numpy(x['occupancy'])[None,None],perceive(torch.from_numpy(c)[None]),torch.from_numpy(x['allowed'])[None,None],torch.Generator().manual_seed(2101),steps,capture=True)
   field=r['field'].numpy()[0,0].astype(bool);score,masks=evaluate_targets(field,scene,fields,domain);outputs[steps]=field
   counts=r['admission_counts'].numpy();caps=r['step_ceilings'].numpy();D,B,C=r['budget'].tolist();K=int(r['quota'])
   assert K==max(9,int(np.ceil((C-27)/63))) and np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))
   assert (counts[:,0]+counts[:,6]<=caps).all() and (counts[:,1]==counts[:,2:6].sum(1)).all() and (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()
   assert connected(field) and not(field&~x['allowed']).any() and (field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field))
   coords=np.argwhere(field);hits=np.flatnonzero(counts[:,0]+counts[:,6]>=C)
   record=dict(case=case,scene=e['id'],cohort=e['cohort'],request=request,steps=steps,status='evaluated',score=score,absolute_fraction_error=abs(score['volume_fraction']-request),extent_zyx_cells=(coords.max(0)-coords.min(0)+1).tolist(),direct_interface_contact_voxels={k:int((field&v).sum()) for k,v in entrance_masks(scene).items()},budget=dict(domain=D,target=B,ceiling=C),quota=K,first_ceiling_step=int(hits[0]+1) if len(hits) else None,unused_capacity=int(C-field.sum()),seconds=time.perf_counter()-tick)
   (OUT/'observations').mkdir(exist_ok=True)
   with (OUT/f'observations/{case}-{steps}.npz').open('xb') as f:np.savez_compressed(f,field=field,bulk=masks['bulk'],state=r['state'].numpy(),proposal=r['proposal'].numpy(),admission_counts=counts,step_ceilings=caps,births=r['births'].numpy())
   save(f'observations/{case}-{steps}.json',record);observations.append(record)
   print(e['cohort'],case,steps,score['contract_pass'],[k for k,v in score['family_pass'].items() if not v],flush=True)
  a,b=outputs[64],outputs[128];assert not(a&~b).any()
  stability.append(dict(case=case,cohort=e['cohort'],relative_mass_change=float((int(b.sum())-int(a.sum()))/int(a.sum())),identical_field=bool(np.array_equal(a,b))))
summary={};gates={}
for cohort,expected in [('regression',21),('fresh_reserved',12)]:
 summary[cohort]={}
 for steps in [64,128]:
  selected=[o for o in observations if o['cohort']==cohort and o['steps']==steps];assert len(selected)==expected
  errors=[o['absolute_fraction_error'] for o in selected]
  s=dict(valid=sum(o['score']['contract_pass'] for o in selected),expected=expected,median_absolute_fraction_error=float(np.median(errors)),max_absolute_fraction_error=max(errors),family_pass={k:sum(o['score']['family_pass'][k] for o in selected) for k in selected[0]['score']['family_pass']});summary[cohort][str(steps)]=s
  gates[f'{cohort}_{steps}_all_nine']=s['valid']==expected
  gates[f'{cohort}_{steps}_volume_error']=s['median_absolute_fraction_error']<=.02 and s['max_absolute_fraction_error']<=.04
 gates[cohort+'_stable_mass']=all(s['relative_mass_change']<=.05 for s in stability if s['cohort']==cohort)
save('result.json',dict(summary=summary,gates=gates,accepted=all(gates.values()),observations=observations,stability=stability))
save('scene-index.json',dict(entries=entries))
shutil.copyfile(__file__,OUT/'review-script.py');print(json.dumps(dict(summary=summary,gates=gates),indent=2))
'''
Path('review_g7.py').write_text(p)
