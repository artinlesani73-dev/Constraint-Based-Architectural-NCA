"""Publication plots from retained logs only. No inference or parameter updates."""
from pathlib import Path
import argparse, csv, hashlib, json, math, zipfile, subprocess
import numpy as np
from reportlab.graphics.shapes import Drawing, String, Rect, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, Color

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=Path('C:/Users/artin/Documents/Codex/outputs')
RUNS={'G8':'G8-Final-Review-2026-10-04-v2','G9':'G9-Final-Review-2026-10-04','G10':'G10-Final-Review-2026-10-04'}
COLORS=['#247d70','#b9613c','#4c67a4']
POPPLER=Path('C:/Users/artin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')

def digest(raw): return hashlib.sha256(raw).hexdigest()
def avg(a, window=25):
    return [(i+window,float(v)) for i,v in enumerate(np.convolve(a,np.ones(window)/window,mode='valid'))]

def build(out):
    if (out/'figure-provenance.json').exists(): raise FileExistsError('Use a new output version; preserve completed evidence.')
    out.mkdir(parents=True,exist_ok=True); tmp=out/'render-source';tmp.mkdir(exist_ok=True)
    provenance={'training':{},'figures':{},'smoothing':'Trailing arithmetic mean of25 logged observations; raw values also shown. No smoothing across runs.','status':'Saved evidence only; no new experiment'}
    histories={}
    for method,folder in RUNS.items():
        base=ARCHIVE/folder; files=sorted((base/'import/worker').glob('update-*.json'))
        assert len(files)==427
        original=next(base.glob('*.zip')); rows=[]; hashes={}
        with zipfile.ZipFile(original) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            for p in files:
                raw=p.read_bytes(); assert raw==z.read('worker/'+p.name)
                row=json.loads(raw); rows.append(row); hashes[p.name]=digest(raw)
        assert [r['update'] for r in rows]==list(range(1,428))
        for r in rows:
            expected=r['frontier_loss']+.25*r['volume_loss']+r['band_loss']+r.get('ranking_loss',0)
            assert math.isclose(r['loss'],expected,rel_tol=2e-6,abs_tol=2e-6)
            assert all(math.isfinite(r[k]) for k in ('loss','frontier_loss','volume_loss','band_loss','pre_clip_gradient_norm'))
        histories[method]=rows
        provenance['training'][method]={'folder':folder,'archive_name':original.name,'verified_logs':427,'update_sha256':hashes}
    assert all([(r['row_index'],r['start']['sha256']) for r in histories[m]]==[(r['row_index'],r['start']['sha256']) for r in histories['G8']] for m in ('G9','G10'))
    oldpath=ROOT/'notebooks/model_c/v31_training_history.json'; old=json.loads(oldpath.read_text())
    assert [r['epoch'] for r in old]==list(range(500))
    provenance['phase1']={'source':oldpath.relative_to(ROOT).as_posix(),'sha256':digest(oldpath.read_bytes()),'observations':500,'caution':'Historical training proxies; not current binary acceptance.'}
    keys=['update','row_index','loss','frontier_loss','volume_loss','band_loss','ranking_loss','pre_clip_gradient_norm','base_loss']
    for m,rows in histories.items():
        with (out/f'{m}-training.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            for r in rows:
                result={k:r.get(k,0) for k in keys};result['base_loss']=r['frontier_loss']+.25*r['volume_loss']+r['band_loss'];writer.writerow(result)
    with (out/'phase1-training.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(old[0]));writer.writeheader();writer.writerows(old)

    def drawing(title,subtitle,w=1200,h=800):
        d=Drawing(w,h);d.add(Rect(0,0,w,h,fillColor=HexColor('#faf9f3'),strokeColor=None));d.add(String(42,h-42,title,fontName='Helvetica-Bold',fontSize=24,fillColor=HexColor('#1c373b')));d.add(String(42,h-66,subtitle,fontName='Helvetica',fontSize=12,fillColor=HexColor('#506667')));return d
    def save(d,name,caption):
        renderSVG.drawToFile(d,str(out/(name+'.svg')));renderPDF.drawToFile(d,str(tmp/(name+'.pdf')))
        subprocess.run([str(POPPLER),'-singlefile','-scale-to','2400','-png',str(tmp/(name+'.pdf')),str(out/name)],check=True,capture_output=True)
        provenance['figures'][name]={'caption':caption,'png_sha256':digest((out/(name+'.png')).read_bytes())}
    def plot(d,x,y,w,h,title,series,xmax,ymax=None,xlabel='Training update',ylabel='Loss'):
        d.add(String(x,y+h+18,title,fontName='Helvetica-Bold',fontSize=13,fillColor=HexColor('#1c373b')))
        lp=LinePlot();lp.x=x+45;lp.y=y+40;lp.width=w-65;lp.height=h-48
        data=[];styles=[]
        for vals,color,label in series:
            a=np.array(vals,dtype=float);assert np.isfinite(a).all()
            data.append(list(enumerate(a,start=1))); styles.append((color,True))
            data.append(avg(a));styles.append((color,False))
        lp.data=data;lp.joinedLines=1
        lp.xValueAxis.valueMin=1;lp.xValueAxis.valueMax=xmax;lp.xValueAxis.valueSteps=[1]+list(range(100,xmax+1,100));lp.xValueAxis.labels.fontSize=8
        lp.yValueAxis.valueMin=0;lp.yValueAxis.valueMax=ymax or max(max(v[0]) for v in series)*1.08;lp.yValueAxis.labels.fontSize=8;lp.yValueAxis.labelTextFormat='%.2g';lp.yValueAxis.visibleGrid=1;lp.yValueAxis.gridStrokeColor=HexColor('#dde4df')
        for i,(color,raw) in enumerate(styles):
            c=HexColor(color);lp.lines[i].strokeColor=Color(c.red,c.green,c.blue,alpha=.16) if raw else c;lp.lines[i].strokeWidth=.5 if raw else 2;lp.lines[i].symbol=None
        d.add(lp);d.add(String(x+w/2,y+8,xlabel,fontSize=9,textAnchor='middle',fillColor=HexColor('#506667')));d.add(String(x+45,y+h-9,ylabel,fontSize=8,fillColor=HexColor('#506667')))
        if len(series)>1:
            for i,(_,color,label) in enumerate(series):d.add(String(x+140+i*90,y+h-9,label,fontSize=8,fillColor=HexColor(color)))

    d=drawing('Phase 1: the original training record','500 logged epochs. Pale lines: raw values. Dark lines: trailing25-observation mean.')
    for i,(key,title,label) in enumerate([('total_loss','Weighted training objective','Loss'),('L_coverage','Unfilled corridor penalty','Historical loss'),('L_thickness','Historical thickness penalty','Historical loss'),('fill_ratio','Recorded occupancy fraction','Fraction')]):
        plot(d,40+(i%2)*580,80+(1-i//2)*320,540,260,title,[([r[key] for r in old],COLORS[0],'Phase 1')],500,xlabel='Logged epoch +1',ylabel=label)
    d.add(String(42,35,'Historical proxies and pavilion intent differ from current gross-volume checks. Do not compare total loss with G8-G10.',fontSize=11))
    save(d,'01-phase1-training','Original500-epoch history; raw values and trailing25-observation means. Historical losses are not current acceptance metrics.')

    d=drawing('Training the later candidates','427 updates per run. Matched rows and starting fields; generated trajectories differ. One training seed.')
    commonmax=max(r['loss'] for rows in histories.values() for r in rows)*1.04
    basemax=max(r['frontier_loss']+.25*r['volume_loss']+r['band_loss'] for rows in histories.values() for r in rows)*1.04
    for j,(m,rows) in enumerate(histories.items()):
        plot(d,35+j*390,420,365,255,m+' / recorded objective',[([r['loss'] for r in rows],COLORS[j],m)],427,commonmax)
        plot(d,35+j*390,100,365,255,m+' / shared base terms',[([r['frontier_loss']+.25*r['volume_loss']+r['band_loss'] for r in rows],COLORS[j],m)],427,basemax)
    d.add(String(42,51,'G8: base objective. G9/G10: base + access ranking. Top-row totals have different objectives; no winner is implied.',fontSize=11))
    d.add(String(42,32,'The same base expression is evaluated along each model\'s own training trajectory, not on a shared fixed validation batch.',fontSize=11))
    save(d,'02-candidate-training','G8/G9/G10 recorded objectives and shared base terms. Raw traces plus trailing25 means. These are training observations, not held-out loss curves.')

    d=drawing('What changed inside the loss','Pale lines: raw values. Dark lines: trailing25-observation mean. Dimensionless quantities; separate axes.')
    for i,(key,title) in enumerate([('frontier_loss','Frontier classification'),('volume_loss','Soft cube-volume discrepancy'),('band_loss','Global volume-band penalty'),('ranking_loss','Access-ranking auxiliary')]):
        selected=[(m,rows) for m,rows in histories.items() if key!='ranking_loss' or m!='G8']
        plot(d,40+(i%2)*580,80+(1-i//2)*320,540,260,title,[([r.get(key,0) for r in rows],COLORS[list(histories).index(m)],m) for m,rows in selected],427)
    d.add(String(42,35,'G8 has no ranking term. G9 and G10 use the same forward ranking formula but different gradient paths.',fontSize=11))
    save(d,'03-loss-components','Separate recorded frontier, soft-volume, band and ranking components. G8 does not have a ranking auxiliary; it is omitted in that panel.')

    d=drawing('Training progress is not acceptance','Contribution and prospective transfer studies: six feasible requests per method, reported separately.',1200,620)
    studies=[]
    for j,(fname,title,key) in enumerate([('R3-contribution-complete-2026-10-08.json','Contribution','feasible'),('R3-transfer-results-2026-10-08.json','Prospective transfer','certified')]):
        p=ROOT/'experiments/reports'/fname;data=json.loads(p.read_text());studies.append(data)
        provenance.setdefault('evaluation',{})[fname]=digest(p.read_bytes())
        x=60+j*580;d.add(String(x,505,title,fontSize=18,fontName='Helvetica-Bold'))
        for k,m in enumerate(('G10','R3','neutral')):
            row=data['summary'][m];count=row[key+'_accepted'];distance=row['mean_seed_jaccard_'+key]
            yy=420-k*100;d.add(String(x,yy+25,m,fontSize=13));d.add(Rect(x+85,yy,360,23,fillColor=HexColor('#e0e7e2'),strokeColor=None));d.add(Rect(x+85,yy,360*count/6,23,fillColor=HexColor(COLORS[k]),strokeColor=None));d.add(String(x+460,yy+5,f'{count}/6',fontSize=14,fontName='Helvetica-Bold'));d.add(String(x+85,yy-20,f'Mean seed Jaccard distance: {distance:.3f}',fontSize=11))
    d.add(String(60,110,'Blocked requests in each study: G10 two rejected outputs; R3 and neutral two refusals each. Refusals are not passes.',fontSize=12))
    d.add(String(60,80,'Neutral also passes all feasible requests. Learned scores change form and variation; necessity for these gates is unproven.',fontSize=12))
    d.add(String(60,50,'Small authored samples, two firing seeds per site. No independent training replication or architectural-quality metric.',fontSize=12))
    save(d,'04-evaluation-comparison','Acceptance on feasible requests, with blocked outcomes separate. Jaccard distance measures seed variation, not quality. Studies are not pooled.')

    d=drawing('The mathematics of the current approach','A compact reference for the article. Symbols describe the later hybrid system, not the original notebook.',1200,650)
    rows=[('01 / Local proposals','z, delta_h = f_theta( Phi([m, h]), Phi(c), r )','m: binary mass; h: seven hidden channels; c: fixed context; r: remaining budget signal.'),('02 / Hybrid growth','m_next = A( m, sigmoid(z), fire, domain, cap_t )','A: detached deterministic cube admission. R3 also establishes a procedural route witness.'),('03 / Base objective','L_base = L_frontier + 0.25 L_volume + L_band','Classification on fired eligible origins; soft cube-volume discrepancy; global volume-band penalty.'),('04 / Access ranking','L_rank = log(1 + mean_(a,b) exp(1 + z_b - z_a))','a: advancing teacher origin; b: other teacher origin. G10 detaches z_b in this auxiliary.'),('05 / Acceptance','all nine at64/128; |v - 0.24| <= 0.04; (N128 - N64)/N64 <= 0.05','v: occupied/domain fraction at each checked horizon. Refusal is reported separately.')]
    for i,(label,eq,desc) in enumerate(rows):
        yy=520-i*100;d.add(String(45,yy,label,fontSize=12,fillColor=HexColor('#247d70'),fontName='Helvetica-Bold'));d.add(String(45,yy-30,eq,fontName='Courier',fontSize=16));d.add(String(45,yy-53,desc,fontSize=11,fillColor=HexColor('#506667')))
    save(d,'05-mathematics','Current local proposal, detached admission, training objective, ranking supplement and independent acceptance gates. ASCII notation for portable embedding.')
    provenance['output_files']={p.name:digest(p.read_bytes()) for p in out.iterdir() if p.is_file()}
    (out/'figure-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'figures':len(provenance['figures']),'archive_matched_update_logs':1281,'matching_starts':427,'phase1_observations':500}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)

