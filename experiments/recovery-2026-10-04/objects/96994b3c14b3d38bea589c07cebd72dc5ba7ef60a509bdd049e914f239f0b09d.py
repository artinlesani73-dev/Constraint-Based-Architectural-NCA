from pathlib import Path
p=Path('review_g8.py');s=p.read_text()
s=s.replace("PACKAGE=BASE/'G8-Exposure-Training-2026-10-04';OUT=BASE/'G8-Final-Review-2026-10-04-v2'","PACKAGE=BASE/'G9-Access-Ranking-Training-2026-10-04-v2';OUT=BASE/'G9-Final-Review-2026-10-04'")
s=s.replace("RUN='20261004T111257Z_f5eb0598cf17'","RUN='20261004T120338Z_60498d5f0838'")
s=s.replace('NCA-G8-Exposure-Package.zip','NCA-G9-Access-Ranking-Package.zip')
s=s.replace("from nca.paced_generation import PacedNCA","from nca.paced_generation import PacedNCA\nfrom nca.ranked_generation import RankedNCA")
s=s.replace('seed_generation_training_v8_exposure','seed_generation_training_v9_access_ranking')
s=s.replace("order=TrainingOrder(45,1203)","phase_totals={};active_ranking=0;no_route_open=0\norder=TrainingOrder(45,1203)")
needle="  training_step_accounts+=len(counts)"
s=s.replace(needle,'''  assert len(t['access_phase_trace'])==64
  assert np.isfinite([t[k] for k in ['loss','frontier_loss','volume_loss','band_loss','ranking_loss','pre_clip_gradient_norm']]).all()
  assert abs(t['loss']-(t['frontier_loss']+.25*t['volume_loss']+t['band_loss']+t['ranking_loss']))<1e-5
  for j,event in enumerate(t['access_phase_trace']):
   assert event['phase'] in ('seed_access','advance_access','connected','no_teacher_route')
   assert event['mass']==int(counts[j,0]) and event['at_capacity']==(counts[j,0]==C)
   assert event['ranking_active']==(event['phase'] in ('seed_access','advance_access') and event['progress_fired']>0 and event['other_teacher_fired']>0)
   assert event['progress_fired']>=0 and event['other_teacher_fired']>=0
   phase_totals[event['phase']]=phase_totals.get(event['phase'],0)+1
   active_ranking+=event['ranking_active']
   no_route_open+=event['phase']=='no_teacher_route' and not event['at_capacity']
  training_step_accounts+=len(counts)''')
s=s.replace("save('training-verification.json',dict(updates=427,","save('training-verification.json',dict(phase_totals=phase_totals,active_ranking_steps=active_ranking,no_route_below_capacity=no_route_open,updates=427,")
s=s.replace("newsplit=json.loads((PACKAGE/'fresh-split-manifest.json').read_text())","g8split=json.loads(Path(protocol['regression_sources'][2]['path']).read_text())\nnewsplit=json.loads((PACKAGE/'fresh-split-manifest.json').read_text())")
s=s.replace("dict(g1=oldsplit,g7=g7split)","dict(g1=oldsplit,g7=g7split,g8=g8split)")
s=s.replace("+[{**e,'cohort':'fresh_reserved'} for e in newsplit['entries']]","+[{**e,'cohort':'regression'} for e in g8split['entries']]+[{**e,'cohort':'fresh_reserved'} for e in newsplit['entries']]+[{**e,'id':'g8-baseline-'+e['id'],'cohort':'baseline_fresh'} for e in newsplit['entries']]")
s=s.replace('assert len(entries)==15','assert len(entries)==23')
s=s.replace("oldsplit['entries']+g7split['entries']+newsplit['entries']","oldsplit['entries']+g7split['entries']+g8split['entries']+newsplit['entries']")
start=s.index("g6=BASE/")
end=s.index("save('execution.json'",start)
s=s[:start]+'''from nca.generation_training import equal_tree
previous=BASE/'G8-Final-Review-2026-10-04-v2'
with zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as z:initial=torch.load(io.BytesIO(z.read('worker/checkpoint-0000.pt')),map_location='cpu',weights_only=False)
with zipfile.ZipFile(previous/'20261004T111257Z_f5eb0598cf17.zip') as z:oldinitial=torch.load(io.BytesIO(z.read('worker/checkpoint-0000.pt')),map_location='cpu',weights_only=False)
assert initial.keys()==oldinitial.keys()
assert all(equal_tree(initial[k],oldinitial[k]) for k in initial if k!='identity')
g8payload=torch.load(previous/'import/worker/checkpoint-0427.pt',map_location='cpu',weights_only=False)
assert sha((previous/'import/worker/checkpoint-0427.pt').read_bytes())==protocol['paired_baseline_checkpoint_sha256']
assert all(t['start']==old['start'] and t['row_index']==old['row_index'] for t,old in zip(p['trace'],g8payload['trace']))
assert all(equal_tree(p[k],g8payload[k]) for k in ['sampler','rng','cuda_rng'])
assert (source/'nca/paced_generation.py').read_bytes()==(previous/'source/nca/paced_generation.py').read_bytes()
save('pairing.json',dict(initial_numerical_payload_equal_g8=True,all427_starts_and_rows_equal=True,final_rng_and_sampler_equal=True,original_pacing_source_equal=True,objective_changed=True))
model=RankedNCA().float();model.load_state_dict(p['model'],strict=True);model.eval()
baseline=PacedNCA().float();baseline.load_state_dict(g8payload['model'],strict=True);baseline.eval()
expected_runtime={'gpu_name':'Tesla T4','torch':'2.11.0+cu130','numpy':'2.1.3','python':'3.13.15','cuda_build':'13.0','cudnn':92700}
assert all(identity['runtime'][k]==v for k,v in expected_runtime.items())
''' +s[end:]
s=s.replace("scene=e['scene'];fields,domain,_=target_context(scene,config)\n assert", "scene=e['scene'];fields,domain,_=target_context(scene,config)\n evaluator=baseline if e['cohort']=='baseline_fresh' else model\n assert")
s=s.replace("with torch.no_grad():r=model.rollout","with torch.no_grad():r=evaluator.rollout")
s=s.replace("[('regression',33),('fresh_reserved',12)]","[('regression',45),('fresh_reserved',12),('baseline_fresh',12)]")
s=s.replace("accepted=all(gates.values())","accepted=all(v for k,v in gates.items() if not k.startswith('baseline_fresh'))")
Path('review_g9.py').write_text(s)
compile(s,'review_g9.py','exec')

