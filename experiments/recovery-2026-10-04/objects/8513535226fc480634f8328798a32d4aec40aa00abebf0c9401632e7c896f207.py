from pathlib import Path
p=Path(__file__).with_name('review_g3.py');s=p.read_text()
s=s.replace("PACKAGE=BASE/'G3-Budget-Training-2026-10-03'","PACKAGE=BASE/'G4-Block-Training-2026-10-03-v2'").replace("OUT=BASE/'G3-Final-Review-2026-10-03'","OUT=BASE/'G4-Final-Review-2026-10-04'").replace("RUN='20261003T201425Z_f94f89d7c7ab'","RUN='20261004T065608Z_176639ac1bc5'")
s=s.replace('NCA-G3-Budget-Package.zip','NCA-G4-Block-Package.zip').replace('from nca.budget_generation import BudgetNCA','from nca.block_generation import BlockNCA,full_origins,connected\nfrom nca.block_reference import cube_union\nfrom nca.budget_reference import budget\nfrom nca.generation_training import SETTINGS\nfrom nca.repair_portable import TrainingOrder').replace('model=BudgetNCA()','model=BlockNCA()')
start=s.index("starts={'seed':0,'teacher_stage':0}");end=s.index('recovery=[',start)
s=s[:start]+'''starts={'seed':0,'cube_teacher_stage':0};training_events={'accepted_blocks':0,'budget_rejected_blocks':0,'redundant_blocks':0,'deferred_seed_blocks':0,'added_voxels':0};training_step_accounts=0
assert identity['model_semantics']=='seed_generation_training_v4_blocks'
assert identity['generation_settings']==request['settings']==SETTINGS
assert request['device']=='cuda:0' and request['seed']==1201 and request['updates']==256 and request['max_seconds']==600
assert run_result['wall_seconds']<=600
probe=json.loads((OUT/'import/worker/device-admission-check.json').read_text());assert probe['passed'] and probe['reference_cases']==12 and probe['union_backward_finite']
order=TrainingOrder(27,1203)
with zipfile.ZipFile(PACKAGE/'NCA-G4-Block-Package.zip') as z,zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as evidence:
 import io
 for i,t in enumerate(p['trace']):
  assert t==json.loads((OUT/f'import/worker/update-{i+1:04d}.json').read_text()) and t['update']==i+1 and t['row_index']==order.next()
  row=train_data['rows'][t['row_index']];raw=z.read(row['arrays']);assert sha(raw)==row['arrays_sha256']
  with np.load(io.BytesIO(raw),allow_pickle=False) as a:
   c=a['condition'].copy();target=a['target'].astype(bool);distance=a['block_distance'].copy();x=seed_inputs(c)
   depth=None if i%2==0 else int.from_bytes(hashlib.sha256(f'{i}:{t["row_index"]}'.encode()).digest()[:8],'little')%int(distance.max())
   expected_start=x['occupancy'].astype(np.uint8) if depth is None else cube_union((distance>=0)&(distance<=depth)).astype(np.uint8)
   expected={'kind':'seed' if depth is None else 'cube_teacher_stage','depth':depth,'occupied':int(expected_start.sum()),'sha256':sha(expected_start.tobytes(order='C'))}
   assert t['start']==expected;starts[expected['kind']]+=1
  with np.load(io.BytesIO(evidence.read(f'worker/training-{i+1:04d}.npz')),allow_pickle=False) as a:
   start_field=a['start'];state=a['state'];assert np.array_equal(start_field,expected_start) and np.isfinite(state).all()
   assert np.isin(state[0],[0,1]).all();field=state[0].astype(bool)
  D=int(x['allowed'].sum());B,C=budget(D,float(c[6,0,0,0]),3);assert t['budget']==[D,B,C]
  counts=np.asarray(t['admission_counts']);assert counts.shape==(64,7) and (counts>=0).all()
  assert (counts[:,1]==counts[:,2:6].sum(1)).all() and (counts[:,0]+counts[:,6]<=C).all()
  assert (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()
  assert counts[0,0]==int(start_field.sum()) and counts[-1,0]+counts[-1,6]==int(field.sum())
  assert not (start_field.astype(bool)&~field).any() and not (field&~x['allowed']).any() and connected(field)
  assert field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field)
  training_step_accounts+=len(counts)
  for k,column in [('accepted_blocks',2),('budget_rejected_blocks',3),('redundant_blocks',4),('deferred_seed_blocks',5),('added_voxels',6)]:training_events[k]+=int(counts[:,column].sum())
save('training-verification.json',dict(updates=256,step_accounts=training_step_accounts,starts=starts,events=training_events,all_start_hashes_and_saved_fields_verified=True,all_terminal_fields_legal_connected_cube_unions_or_single_seed=True,sampler_order_verified=True,device_probe=probe,full_training_rollouts_replayed=False))
''' +s[end:]
s=s.replace("path=PREP/row['arrays']","path=PREP/row['arrays'].replace('\\\\','/')")
s=s.replace("assert (counts[:,0]+counts[:,2]<=C).all() and (counts[:,1]==counts[:,2]+counts[:,3]).all()","assert (counts[:,0]+counts[:,6]<=C).all() and (counts[:,1]==counts[:,2:6].sum(1)).all()")
s=s.replace("assert (counts[1:,0]==counts[:-1,0]+counts[:-1,2]).all()","assert (counts[1:,0]==counts[:-1,0]+counts[:-1,6]).all()")
s=s.replace("hits=np.flatnonzero(counts[:,0]+counts[:,2]>=C)","hits=np.flatnonzero(counts[:,0]+counts[:,6]>=C)\n  assert field.sum()==1 or np.array_equal(cube_union(full_origins(field)),field)\n  assert connected(field) and not (field&~domain).any()")
s=s.replace("budget_rejected_events=int(counts[:,3].sum()),pre_admission_final_candidate_score=raw_score)","budget_rejected_events=int(counts[:,3].sum()),accepted_block_events=int(counts[:,2].sum()),redundant_block_events=int(counts[:,4].sum()),deferred_seed_block_events=int(counts[:,5].sum()),zero_growth_steps=int((counts[:,6]==0).sum()),unused_capacity=int(C-field.sum()),pre_admission_final_candidate_score=raw_score)")
compile(s,'review_g4.py','exec');Path(__file__).with_name('review_g4.py').write_text(s)
