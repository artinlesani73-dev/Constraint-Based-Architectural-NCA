from pathlib import Path
s=Path(__file__).with_name('review_g5.py').read_text()
s=s.replace("PACKAGE=BASE/'G5-Destination-Guidance-2026-10-04'","PACKAGE=BASE/'G6-Paced-Growth-2026-10-04'").replace("OUT=BASE/'G5-Final-Review-2026-10-04'","OUT=BASE/'G6-Final-Review-2026-10-04'").replace("RUN='20261004T074101Z_d3a90a77c991'","RUN='20261004T081739Z_95ddbf523738'")
s=s.replace('NCA-G5-Destination-Package.zip','NCA-G6-Paced-Package.zip').replace('from nca.guided_generation import GuidedNCA','from nca.paced_generation import PacedNCA').replace('GuidedNCA()','PacedNCA()').replace('seed_generation_training_v5_destination','seed_generation_training_v6_paced')
s=s.replace("assert probe['cue_device_copy_exact'] and probe['destination_cue_version']=='opposite_x_interface_cube_distance_v1'","assert probe['paced_reference_cases']==8 and probe['pacing_horizon_constant']==64")
s=s.replace('budget_rejected_blocks','allowance_rejected_blocks')
needle="  counts=np.asarray(t['admission_counts']);assert counts.shape==(64,7) and (counts>=0).all()"
assert needle in s
s=s.replace(needle,needle+"\n  K=max(9,int(np.ceil((C-27)/63)));caps=np.asarray(t['step_ceilings'])\n  assert t['quota']==K and caps.shape==(64,) and np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))\n  assert (counts[:,0]+counts[:,6]<=caps).all()")
s=s.replace('full_training_rollouts_replayed=False))','full_training_rollouts_replayed=False,all_16384_effective_ceilings_verified=True))')
start=s.index("assert torch.equal(initial[0]['first.weight']")
end=s.index('\nmodel=PacedNCA()',start)
s=s[:start]+'''assert initial[0].keys()==initial[1].keys() and all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
save('pairing-with-g4.json',dict(dataset_hashes_equal=True,all256row_and_start_schedules_equal=True,final_firing_rng_equal=True,all_initial_parameters_equal=True,runtime_keys_equal=runtime_keys,extra_parameters=0,limitation='One seeded pacing intervention,not a multi-seed causal estimate. Rejection column now means effective allowance,not global cap alone.'))
''' +s[end:]
needle="  counts=r['admission_counts'].numpy();D,B,C=r['budget'].tolist();candidate=r['pre_admission_candidates'][-1].numpy()[0,0]"
assert needle in s
s=s.replace(needle,needle+"\n  K=int(r['quota']);caps=r['step_ceilings'].numpy()\n  assert K==max(9,int(np.ceil((C-27)/63))) and np.array_equal(caps,np.where(counts[:,0]==1,C,np.minimum(C,counts[:,0]+K)))\n  assert (counts[:,0]+counts[:,6]<=caps).all()")
s=s.replace('budget_rejected_events=int(counts[:,3].sum())','allowance_rejected_events=int(counts[:,3].sum()),quota=K,effective_allowance_hit_steps=int(((counts[:,0]+counts[:,6]==caps)&(caps<C)).sum())')
s=s.replace('admission_counts=counts,pre_admission_final_candidate=candidate','admission_counts=counts,step_ceilings=caps,quota=K,pre_admission_final_candidate=candidate')
compile(s,'review_g6.py','exec');Path(__file__).with_name('review_g6.py').write_text(s)
