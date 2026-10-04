from pathlib import Path
import hashlib,json,shutil

REPO=Path('C:/LAB/ai-aec-playground/PROJECTS/Constraint-Based-Architectural-NCA')
BASE=Path('C:/Users/artin/Documents/Codex/outputs')
DEST=REPO/'experiments/recovery-2026-10-04'
assert not DEST.exists(), 'Recovery import already exists; do not overwrite'
DEST.mkdir(parents=True)
TEXT={'.py','.js','.css','.html','.md','.json','.ipynb','.txt','.diff','.log'}
prefixes=('G1','G2','G3','G4','G5','G6','G7','G8','G9','Generation-','Reversible-','RGR1-','CGR3-cu130','CGR3-Curriculum-cu130','CGR3-Final-','CGR3-Transfer-')
folders=sorted(p for p in BASE.iterdir() if p.is_dir() and p.name.startswith(prefixes))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def private(p):return 'nca-next-phase-report' in p.name.lower()
def capture(p):
 digest=sha(p);n=p.stat().st_size
 row={'bytes':n,'sha256':digest}
 if p.suffix.lower() in TEXT and n<=512*1024:
  rel=Path('objects')/(digest+p.suffix.lower())
  target=DEST/rel;target.parent.mkdir(exist_ok=True)
  if not target.exists():shutil.copyfile(p,target)
  row['tracked_object']=rel.as_posix()
 else:row['storage']='original local artifact; not in Git'
 return row
index={'schema':1,'date':'2026-10-04','artifact_root':str(BASE),'snapshots':{},'tooling':{},'archives':{}}
for folder in folders:
 rows={}
 for p in sorted(folder.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or private(p):continue
  rows[p.relative_to(folder).as_posix()]=capture(p)
 index['snapshots'][folder.name]=rows
 # Keep narrative records immediately readable without the object index.
 for p in folder.glob('*.md'):
  if private(p):continue
  q=DEST/'reports'/folder.name/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
for p in sorted(Path.cwd().iterdir()):
 if p.is_file() and p.suffix in {'.py','.js','.css','.html'} and not private(p):index['tooling'][p.name]=capture(p)
for p in sorted(BASE.glob('*.zip')):
 if any(p.name.startswith(d.name) for d in folders):index['archives'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
(DEST/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
# Approved application source is also readable at its original relative paths.
studio=REPO/'experimental/r3-studio'
src=BASE/'G11-R3-Skins-2026-10-04'
for name,row in index['snapshots'][src.name].items():
 p=Path(name)
 if p.parts[0] in {'runs','drafts','implementation-history','model'}:continue
 if p.suffix not in {'.py','.js','.css','.html','.json'} or p.name in {'data.js','milestone-manifest.json'}:continue
 q=studio/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/p,q)
print(json.dumps({'snapshots':len(folders),'files':sum(len(x) for x in index['snapshots'].values()),'objects':len(list((DEST/'objects').iterdir())),'archives':len(index['archives'])}))
