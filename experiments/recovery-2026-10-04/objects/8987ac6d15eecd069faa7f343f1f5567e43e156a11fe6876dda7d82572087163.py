from pathlib import Path
p=Path('review_g7.py').read_text().replace('G7-Vertical-Training-2026-10-04-v2','G8-Exposure-Training-2026-10-04').replace('G7-Final-Review-2026-10-04','G8-Final-Review-2026-10-04').replace('20261004T103602Z_1da578905a51','20261004T111257Z_f5eb0598cf17').replace('NCA-G7-Vertical-Package.zip','NCA-G8-Exposure-Package.zip').replace('seed_generation_training_v7_vertical','seed_generation_training_v8_exposure').replace('0256','0427').replace('256','427').replace('16384','27328')
p=p.replace('sha427','sha256').replace('G8-Final-Review-2026-10-04','G8-Final-Review-2026-10-04-v2')
a=p.index("assert sha((PREP/'split-manifest.json')");b=p.index("# Input construction checked",a)
p=p[:a]+'''for dependency in protocol['regression_sources']:
 assert sha(Path(dependency['path']).read_bytes())==dependency['sha256']
assert sha((PACKAGE/'fresh-split-manifest.json').read_bytes())==protocol['fresh_manifest_sha256']
for name in ['frozen-review.json','fresh-split-manifest.json','environment.json','package-receipt.json']:
 shutil.copyfile(PACKAGE/name,OUT/name)
oldsplit=json.loads(Path(protocol['regression_sources'][0]['path']).read_text())
g7split=json.loads(Path(protocol['regression_sources'][1]['path']).read_text())
newsplit=json.loads((PACKAGE/'fresh-split-manifest.json').read_text())
save('regression-splits.json',dict(g1=oldsplit,g7=g7split))
entries=[{**e,'cohort':'regression'} for e in oldsplit['entries'] if e['split'] in ['development','reserved']]+[{**e,'cohort':'regression'} for e in g7split['entries'] if e['split']=='reserved']+[{**e,'cohort':'fresh_reserved'} for e in newsplit['entries']]
assert len(entries)==15
config=json.loads((PACKAGE/'environment.json').read_text())['config']
''' +p[b:]
p=p.replace("oldsplit['entries']+newsplit['entries']","oldsplit['entries']+g7split['entries']+newsplit['entries']")
insert=p.index('model=PacedNCA().float();model.load_state_dict(p[\'model\']')
p=p[:insert]+'''from nca.generation_training import equal_tree
previous=BASE/'G7-Final-Review-2026-10-04'
g7payload=torch.load(previous/'import/worker/checkpoint-0256.pt',map_location='cpu',weights_only=False)
with zipfile.ZipFile(DOWNLOAD/(RUN+'.zip')) as z:g8prefix=torch.load(io.BytesIO(z.read('worker/checkpoint-0256.pt')),map_location='cpu',weights_only=False)
assert g7payload.keys()==g8prefix.keys()
prefix_checks={k:equal_tree(g7payload[k],g8prefix[k]) for k in g7payload if k!='identity'}
save('g7-prefix-verification.json',dict(update=256,identity_separate=True,checks=prefix_checks,all_numerical_equal=all(prefix_checks.values())))
print('G7 update256 prefix exact:',all(prefix_checks.values()),flush=True)
''' +p[insert:]
p=p.replace("[('regression',21),('fresh_reserved',12)]","[('regression',33),('fresh_reserved',12)]")
Path('review_g8.py').write_text(p)
