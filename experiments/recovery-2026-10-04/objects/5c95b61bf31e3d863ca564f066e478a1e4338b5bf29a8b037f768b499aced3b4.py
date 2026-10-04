from pathlib import Path
import json,shutil,hashlib
ROOT=Path('C:/Users/artin/Documents/Codex/outputs');P=ROOT/'G11-R3-Studio-2026-10-04';P.mkdir(exist_ok=False)
V=ROOT/'G11-R3-Variety-2026-10-04';R=ROOT/'G11-R3-Independent-Review-2026-10-04';OLD=ROOT/'G11-R3-Preview-2026-10-04'
shutil.copytree(V/'source',P/'source');shutil.copytree(V/'model',P/'model');shutil.copyfile(R/'config.json',P/'config.json')
scenes=[e['scene'] for e in json.loads((R/'scene-index.json').read_text())['entries']]+json.loads((V/'scenes.json').read_text());(P/'scenes.json').write_text(json.dumps(scenes,indent=2))
shutil.copyfile('r3_studio_server.py',P/'server.py');shutil.copyfile('r3_studio.js',P/'studio.js');shutil.copyfile(OLD/'data.js',P/'data.js')
s=(OLD/'index.html').read_text(encoding='utf-8').replace('research preview 01','local generation studio').replace('Pilot evaluation · 81 / 81 R3 passes','Experimental hybrid · CPU generation').replace('Saved evaluation outputs, not a live generator.','Generate new paired outputs on the listed sites, or inspect the saved evaluation cases. Local runs are exploratory and do not extend the independent evaluation score.')
s=s.replace("let out=c.outputs[`${model}-${step}`],r=out.record", "let out=c.outputs[`${model}-${step}`];if(out.error){$(`${prefix}Status`).textContent='No certified output';$(`${prefix}Stats`).textContent=out.error;let cv=$(`${prefix}Canvas`);cv.getContext('2d').clearRect(0,0,cv.width,cv.height);continue}let r=out.record")
s=s.replace("$('source').textContent='Evidence:","$('source').textContent=c.cohort==='generated'?`Local run ${c.id}; seed ${c.seed}. Exact inputs, trajectories, metrics and hashes are saved in the studio runs folder.`:'Evidence:")
s=s.replace('</script></html>','</script><script src="studio.js"></script></html>');(P/'index.html').write_text(s,encoding='utf-8')
files={p.relative_to(P).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and (p.parts[-2]=='model' or 'source' in p.parts or p.name in ['config.json','scenes.json'])}
(P/'identity.json').write_text(json.dumps(dict(version='R3-local-studio-v1',files=files,parent=str(V),semantics='same frozen R3 with optional firing seed; no training'),indent=2))
shutil.copyfile(__file__,P/'build.py');(P/'RESUME.md').write_text('Implementation prepared; verify one actual generation, persisted reload and failure handling before closure. Prior milestone: ../G11-R3-Variety-2026-10-04/RESUME.json. MG7 unchanged. Repository synchronization pending.\n')
print(P)
