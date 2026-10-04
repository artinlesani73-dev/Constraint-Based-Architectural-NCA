from pathlib import Path
s=Path(__file__).with_name('review_g4.py').read_text()
s=s.replace("PACKAGE=BASE/'G4-Block-Training-2026-10-03-v2'","PACKAGE=BASE/'G5-Destination-Guidance-2026-10-04'").replace("OUT=BASE/'G4-Final-Review-2026-10-04'","OUT=BASE/'G5-Final-Review-2026-10-04'").replace("RUN='20261004T065608Z_176639ac1bc5'","RUN='20261004T074101Z_d3a90a77c991'")
s=s.replace('NCA-G4-Block-Package.zip','NCA-G5-Destination-Package.zip').replace('from nca.block_generation import BlockNCA,full_origins,connected','from nca.guided_generation import GuidedNCA\nfrom nca.block_generation import full_origins,connected').replace("'seed_generation_training_v4_blocks'","'seed_generation_training_v5_destination'").replace('model=BlockNCA()','model=GuidedNCA()')
s=s.replace("order=TrainingOrder(27,1203)","assert probe['cue_device_copy_exact'] and probe['destination_cue_version']=='opposite_x_interface_cube_distance_v1'\norder=TrainingOrder(27,1203)")
needle='model=GuidedNCA().float();model.load_state_dict(p[\'model\'],strict=True);model.eval()'
assert needle in s
extra='''
# Verify pairing against the existing G4GPU run without another control job.
g4_dir=BASE/'G4-Final-Review-2026-10-04'
g4_identity=json.loads((g4_dir/'import/worker/identity.json').read_text())
g4=read_portable(g4_dir/'import/worker/checkpoint-0256.pt',g4_identity)
assert identity['ordered_training_rows']==g4_identity['ordered_training_rows']
assert [(t['row_index'],t['start']) for t in p['trace']]==[(t['row_index'],t['start']) for t in g4['trace']]
assert torch.equal(p['rng']['firing'],g4['rng']['firing'])
runtime_keys=['python','torch','numpy','gpu_name','cuda_build','cudnn','dtype','threads','deterministic','tf32_matmul','tf32_cudnn']
assert all(identity['runtime'][k]==g4_identity['runtime'][k] for k in runtime_keys)
g4archive=g4_dir/'20261004T065608Z_176639ac1bc5.zip'
g4receipt=json.loads(g4archive.with_suffix('.receipt.json').read_text());assert sha(g4archive.read_bytes())==g4receipt['sha256']
initial=[]
for path in [g4archive,DOWNLOAD/(RUN+'.zip')]:
 with zipfile.ZipFile(path) as z:
  em=json.loads(z.read('evidence-manifest.json'));raw=z.read('worker/checkpoint-0000.pt');assert sha(raw)==em['worker/checkpoint-0000.pt']
  initial.append(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)['model'])
assert torch.equal(initial[0]['first.weight'],initial[1]['first.weight'][:,:61])
assert torch.count_nonzero(initial[1]['first.weight'][:,61:])==0
assert all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0] if k!='first.weight')
save('pairing-with-g4.json',dict(dataset_hashes_equal=True,all256row_and_start_schedules_equal=True,final_firing_rng_equal=True,initial_core_parameters_equal=True,new_cue_weights_initially_zero=True,runtime_keys_equal=runtime_keys,final_cue_weight_absolute_sums=p['model']['first.weight'][:,61:].abs().sum((0,2,3,4)).tolist(),extra_parameters=128,limitation='One seeded feature intervention,not a multi-seed causal estimate;input shape may also affect numerical kernels.'))
'''
s=s.replace(needle,extra+'\n'+needle)
compile(s,'review_g5.py','exec');Path(__file__).with_name('review_g5.py').write_text(s)
