from pathlib import Path
r=Path('.')
for src,dst in [('nca/bulk_package.py','nca/curriculum_package.py'),('scripts/build_bulk_repair.py','scripts/build_curriculum_repair.py'),('scripts/colab_bulk_repair.py','scripts/colab_curriculum_repair.py')]:
 s=(r/src).read_text().replace('CGR2','CGR3').replace('cgr2','cgr3').replace('bulk_repair','curriculum_repair').replace('bulk_package','curriculum_package').replace('BULK_REPAIR_PROTOCOL','CURRICULUM_REPAIR_PROTOCOL').replace('BULK_REPAIR_SPEC','CGR3_CURRICULUM_SPEC').replace('bulk-runs','curriculum-runs').replace('bulk-attempt','curriculum-attempt').replace('NCA-CGR3-Connected','NCA-CGR3-Curriculum')
 if 'build_' in dst:s=s.replace("'nca/curriculum_repair.py',", "'nca/connected_repair.py','nca/repair_curriculum.py','nca/curriculum_repair.py',")
 if 'colab_' in dst:s=s.replace('np.savez_compressed(f,state=state)','np.savez_compressed(f,state=state,start=session.last_start)')
 (r/dst).write_text(s,encoding='utf-8')
s=(r/'docs/next-phase/BULK_REPAIR_PROTOCOL.md').read_text().replace('CGR2','CGR3').replace('Connected','Curriculum').replace('BULK_REPAIR_SPEC','CGR3_CURRICULUM_SPEC')
s+='\nCGR3 changes training starts only versus CGR1. Original-input evaluation is unchanged.\nVisit counters and each start hash are saved with checkpoint history; start arrays\nare exported per update. The eight-update rehearsal uses first visits (original\nstarts); the focused recovery test explicitly crosses an augmented second visit.\n'
(r/'docs/next-phase/CURRICULUM_REPAIR_PROTOCOL.md').write_text(s,encoding='utf-8')
