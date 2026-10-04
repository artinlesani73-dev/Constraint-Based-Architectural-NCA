from pathlib import Path
p=Path(__file__).parent
s=(p/'render_g11.py').read_text().replace('G11-R1-Prototype-2026-10-04-v2','G11-R3-Independent-Review-2026-10-04').replace('G11-R1','R3')
s=s.replace("scenes=json.loads((OUT/'scenes.json').read_text())","scenes={e['id']:e['scene'] for e in json.loads((OUT/'scene-index.json').read_text())['entries']}")
s=s.replace("cases=list(dict.fromkeys(x['case'] for x in r['observations']))","""# Render all fresh cases plus every regression case with a family failure in either model.
cases=list(dict.fromkeys(x['case'] for x in r['observations'] if x['cohort']=='fresh' or x['status']!='evaluated' or not x.get('score',{}).get('contract_pass',False)))
(OUT/'visual-selection.json').write_text(json.dumps(dict(cases=cases,rule='all fresh plus all family/certificate failures in either model'),indent=2),encoding='utf-8')""")
s=s.replace("with np.load(OUT/'cases'/case/f'{model}-{step}.npz') as a:field=a['field']","""path=OUT/'cases'/case/f'{model}-{step}.npz'
   if not path.exists():
    d.text((col*400+20,y+100),'No certified output',font=font(18),fill='#9a3c31')
    continue
   with np.load(path) as a:field=a['field']""")
s=s.replace("rec['procedural_voxels']","rec['planner_voxels']")
s=s.replace('TRAIN ONLY | G10 raw versus R3 hybrid | no smoothing','Frozen evaluation | G10 raw versus R3 hybrid | no smoothing')
(p/'render_g11_independent.py').write_text(s,encoding='utf-8')

