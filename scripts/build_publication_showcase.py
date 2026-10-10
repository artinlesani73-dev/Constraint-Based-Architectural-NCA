"""Build an offline showcase and evidence-grounded technical PDF; never publish."""
from pathlib import Path
import argparse, hashlib, json, shutil, html
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
SHA = '8e4ac1f13ff57b2c644cd2bd6fd27c40b25eca7062ec729f002969171a122bb2'

def build(out, replace_draft=False):
    if (out / 'manifest.json').exists() and not replace_draft:
        raise FileExistsError('Use a new output directory; --replace-draft is only for an unfinished draft.')
    out.mkdir(parents=True, exist_ok=True)
    assets = out / 'assets'; assets.mkdir(exist_ok=True)
    study = ROOT / '.local-artifacts/r3-transfer-v1-2026-10-08'
    transfer_path = ROOT / 'experiments/reports/R3-transfer-results-2026-10-08.json'
    contribution_path = ROOT / 'experiments/reports/R3-contribution-complete-2026-10-08.json'
    transfer = json.loads(transfer_path.read_text())
    contribution = json.loads(contribution_path.read_text())
    protocol = json.loads((study / 'protocol.json').read_text())
    gallery = []
    for i, scene in enumerate(protocol['sites']):
        for seed in (2201, 2202):
            for method in ('G10', 'R3', 'neutral'):
                ob = next(o for o in transfer['observations'] if o['site'] == scene['scene_id'] and o['seed'] == seed and o['method'] == method)
                slot = study / f'slot-{i}-{seed}-{method}'
                frames = {}
                for step in (64, 128):
                    path = slot / f'{method}-{seed}-{step}.npz'
                    if path.exists():
                        with np.load(path, allow_pickle=False) as data:
                            field = data['field']
                            if field.ndim != 3: raise ValueError(f'Unexpected field shape: {field.shape}')
                            frames[str(step)] = np.flatnonzero(field > .5).tolist()
                gallery.append(dict(site=i, seed=seed, method=method, accepted=ob.get('accepted', False), frames=frames))
    public = dict(sites=[dict(id=s['scene_id'], n=s['grid_size'], buildings=s['buildings']) for s in protocol['sites']], slots=gallery)
    (assets / 'gallery.js').write_text('const EVIDENCE = ' + json.dumps(public, separators=(',', ':')) + ';', encoding='utf-8')
    for seed in (2201, 2202): shutil.copy2(study / f'comparison-{seed}.png', assets)
    evidence = dict(author='Artin Lesani Abdi', date='2026-10-10', checkpoint_sha256=SHA,
                    transfer=transfer['summary'], contribution=contribution['summary'], sources={})
    for p in (transfer_path, contribution_path, study / 'protocol.json', study / 'model/identity.json'):
        evidence['sources'][p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    (assets / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    pdfmetrics.registerFont(TTFont('Segoe', 'C:/Windows/Fonts/segoeui.ttf'))
    pdfmetrics.registerFont(TTFont('SegoeBold', 'C:/Windows/Fonts/segoeuib.ttf'))
    pdf = out / 'NCA-Technical-Report.pdf'
    c = canvas.Canvas(str(pdf), pagesize=(595, 842))
    c.setTitle('Constraint-Based Architectural NCA - Technical Report')
    c.setAuthor('Artin Lesani Abdi')
    page = 0
    def start(kicker, title):
        nonlocal page
        page += 1
        c.setFillColor(HexColor('#f7f6ef')); c.rect(0, 0, 595, 842, fill=1, stroke=0)
        c.setFillColor(HexColor('#217467')); c.setFont('SegoeBold', 10); c.drawString(45, 793, kicker.upper())
        c.setFillColor(HexColor('#172c30')); c.setFont('SegoeBold', 25); c.drawString(45, 750, title)
        c.setStrokeColor(HexColor('#c8d2cc')); c.line(45, 43, 550, 43)
        c.setFont('Segoe', 8); c.drawString(45, 28, 'Artin Lesani Abdi  /  10 October 2026  /  Research prototype')
        c.drawRightString(550, 28, f'{page:02d}')
    def text(s, y, size=11, bold=False, width=505):
        c.setFillColor(HexColor('#263d40')); c.setFont('SegoeBold' if bold else 'Segoe', size)
        words = s.split(); line = ''; lines = []
        for w in words:
            attempt = (line+' '+w).strip()
            if pdfmetrics.stringWidth(attempt, 'SegoeBold' if bold else 'Segoe', size) > width:
                lines.append(line); line = w
            else: line = attempt
        lines.append(line)
        for line in lines: c.drawString(45, y, line); y -= size*1.5
        return y-12
    def end(): c.showPage()
    start('Technical report / v1', 'Constraint-Based Architectural NCA')
    y=text('Context-conditioned growth of architectural mass', 704, 20, True)
    y=text('Building-scale results, learned contribution and a local research studio', y, 13)
    y=text('By Artin Lesani Abdi', y-8, 12, True)
    c.drawImage(str(assets/'comparison-2201.png'), 96, 160, width=403, height=400, preserveAspectRatio=True)
    text('Saved transfer results, seed 2201. All four authored cases and all three methods are retained, including rejected outputs and refusals.', 132, 9)
    text('Prepared for public review. This report is not peer reviewed and does not certify a building design.', 85, 9)
    end()
    start('01 / Intent and scope', 'Growth as a research instrument')
    y=705
    for title, body in [
        ('The design question', 'Can repeated local updates produce volumetric mass that responds to existing buildings, available space and facade interfaces, while meeting explicit geometric checks? The output is gross building volume. It is not solid construction material, a room layout or an inhabited building.'),
        ('Why cellular growth?', 'A neural cellular automaton applies a shared learned update rule repeatedly across a grid. Local interaction offers a way to study distributed form formation. In this project the learned rule operates inside a hybrid system: procedural admission, budgets and route planning also determine the result.'),
        ('What has been built', 'A frozen learned baseline (G10), an experimental hybrid (R3), contribution and prospective transfer comparisons, and a local-only studio for saved runs, generation, exports, drafts and versioned research notes.'),
        ('What the evidence supports', 'R3 achieved higher acceptance than G10 in these small authored studies. A neutral control using the same R3 planner and guards also passed every feasible request. Therefore the studies do not establish that learning is necessary for the tested acceptance gates. Learned scores affect form and seed variation.'),
        ('Scope boundary', 'The demonstrated domain is building-scale massing. No structural, regulatory, habitability, environmental-performance or architectural-quality validation is claimed. MG7 remains the production designation; R3 is an experimental research release.')]:
        y=text(title,y,13,True); y=text(body,y)
    end()
    start('02 / System', 'Three methods, one comparison')
    y=705
    for title, body in [
        ('G10: learned baseline with procedural admission', 'A learned local rule proposes additions. Detached deterministic CPU admission selects connected overlapping 3 x 3 x 3 voxel cubes, with legality and paced budget control. G10 is not a purely neural or unconstrained baseline.'),
        ('R3: planning plus learned growth', 'A procedural planner establishes a thick route witness and coverage; learned scores influence further growth within procedural guards and a budget. Acceptance is evaluated on saved geometry, rather than inferred from the planner alone.'),
        ('Neutral: R3 with the learned signal removed', 'The comparison keeps the R3 planner and guards, replaces occupancy logits with a constant +1 and hidden updates with zero. This removes learned gating and ranking together; it is not a ranking-only ablation.'),
        ('Frozen local update network', 'The checkpoint identity records architecture 61-64-8: 61 perceived/scalar inputs, a 64-channel intermediate layer and 8 output channels. The dynamic state has occupancy plus 7 hidden channels. These network dimensions should not be confused with older configuration defaults.'),
        ('Scale and interpretation', 'Voxels are isotropic at 0.8 m. A 32-cubed domain spans 25.6 m per edge; a 40-cubed domain spans 32 m. Increasing the number of cells here increases physical extent, not geometric resolution. A 3-cell cube spans 2.4 m.')]:
        y=text(title,y,13,True); y=text(body,y)
    y=text('Checkpoint SHA-256',y,10,True)
    y=text(SHA[:32]+' '+SHA[32:],y,9)
    end()
    start('03 / Evaluation', 'What passing actually means')
    y=text('Nine existing constraint families are retained. The current massing checks are geometric proxies, not building codes. Contract/context validity is also required.',705)
    metrics=[('Access','Raw and bulk mass connect the declared interfaces; disconnected occupied regions fail.'),('Coverage','Bulk mass occupies at least 8% of available domain cells in each of three X thirds.'),('Facade','Disallowed contact with existing buildings is at most 15% of occupied mass under the defined contact neighborhood.'),('Ground','No intersection with protected ground exclusions.'),('Legality','No occupied cells outside legal space.'),('Sparsity','Occupied fraction of the domain is between 8% and 40%.'),('Spill','No occupied cells outside the generation domain.'),('Support','Nonempty mass has no voxels classified as unsupported by the geometric support test; this is not mechanical analysis.'),('Thickness','At least 90% of mass belongs to the union of fully occupied 3 x 3 x 3 cubes; this is a bulk proxy, not wall thickness.')]
    for a,b in metrics: y=text(a+' - '+b,y,10)
    y=text('Study acceptance',y,13,True)
    y=text('All nine checks must pass at steps 64 and 128. The requested occupancy is 24%; absolute occupancy-fraction error must be at most 0.04. Relative growth from step 64 to 128 must be at most 0.05. A refused request is not accepted geometry.',y)
    text('The prospective transfer protocol froze four authored sites before generation: narrow offset, flanking blocks, raised obstacle, and sealed gap. Seeds 2201 and 2202 were used with each of three methods: 24 slots total, including blocked cases.',y)
    end()
    start('04 / Results', 'Acceptance and learned contribution')
    y=text('Counts below use the six feasible requests per method in each study. The two blocked requests per method are reported separately; they are not dropped or counted as passes.',705)
    for name, obj, key in [('Contribution study',contribution,'feasible'),('Prospective transfer',transfer,'certified')]:
        y=text(name,y,15,True)
        for m in ('G10','R3','neutral'):
            row=obj['summary'][m]
            y=text(f"{m}: {row[key+'_accepted']}/{row[key+'_requests']} feasible accepted; mean seed Jaccard distance {row['mean_seed_jaccard_'+key]:.3f}",y,11)
    y=text('Blocked cases',y,13,True)
    y=text('In each study, G10 produced two rejected outputs on the blocked site. R3 and neutral each refused both blocked requests. In transfer, all 24 slots completed: 20 outputs and 4 refusals, with no retries or timeouts.',y)
    y=text('Interpretation',y,13,True)
    y=text('Transfer acceptance improved from 2/6 for G10 to 6/6 for R3. Neutral also achieved 6/6, showing that procedural planning and guards can satisfy these gates without learned signals on this sample. R3 varied more across seeds than neutral (0.254 versus 0.037 mean Jaccard distance). Distance is 1 minus intersection-over-union of occupied voxel sets; more variation is not a quality score.',y)
    text('These are small authored samples, not population estimates. Two seeds are not independent training repetitions. Transfer scenes were frozen prospectively for this study, but are now exposed. No exhaustive training-history audit or broad out-of-distribution guarantee is claimed.',y)
    end()
    start('05 / Visual evidence', 'A second seed, the same protocol')
    c.drawImage(str(assets/'comparison-2202.png'),45,188,width=505,height=501.6,preserveAspectRatio=True)
    y=text('Seed 2202. Rows retain all four transfer scenes. Columns compare G10, R3 and neutral under the same site and seed.',158,10)
    text('Use the showcase replay to rotate actual saved voxel fields at steps 64 and 128. Geometry is experiment evidence, not an illustrative reconstruction. Refusals are displayed without substitute geometry.',y,10)
    end()
    start('06 / Reliability and limits', 'Evidence before scale')
    y=705
    for title, body in [
        ('Measured CPU resources', 'For R3 transfer outputs, rollout time ranged from 5.78 to 12.02 seconds. Peak worker working set ranged from 257.7 to 307.4 MiB. Rollout time excludes setup, planning and scoring; working set covers the whole worker through artifact writing. These are observed CPU measurements, not GPU forecasts.'),
        ('Failures remain part of the record', 'The earlier contribution run stopped after a CPU allocation failure; its cause was not established. Fifteen original outputs were preserved. A continuation supplied nine missing slots after capture/retention changes and exact-parity verification. Three verification-only outputs were excluded from the 24-slot analysis.'),
        ('Private research studio', 'The local studio preserves run identities, partial evidence, retry lineage, versioned drafts and notes. Geometry exports use metres and Z-up. A worker may outlive the server; ownership checks prevent concurrent conflicting use. Recovery preserves evidence and supports a new retry, not numerical checkpoint continuation.'),
        ('Remaining scientific uncertainty', 'Learned necessity, quality of form, broader generalization, long-horizon behavior and performance at larger domains remain open. A voxel OBJ is not guaranteed to be a manifold CAD model. A same-disk archive is not an off-device backup.'),
        ('Next phase: multiple sites', 'The agreed direction is context-conditioned growth of architectural mass across multiple sites. A 96 x 96 x 40 context representation exists at 0.8 m with three existing buildings and two opportunity regions. This is a schema/fixture milestone only: no generated neighborhood or validated multi-site model is presented. This work is deferred while preparing the building-scale showcase.')]:
        y=text(title,y,13,True); y=text(body,y)
    end()
    start('07 / Sources and reproducibility', 'A traceable research snapshot')
    y=text('Project repository',705,13,True)
    url='https://github.com/artinlesani73-dev/Constraint-Based-Architectural-NCA'
    y=text(url,y,9); c.linkURL(url,(45,y+12,550,y+35),relative=0)
    y=text('Primary internal evidence',y,13,True)
    for p in evidence['sources']:
        y=text(p,y,9)
        digest=evidence['sources'][p]; y=text('SHA-256: '+digest[:32]+' '+digest[32:],y,8)
    y=text('Study receipts, summaries and frozen protocol underpin the counts. The local artifact archive retains fields, manifests, configuration, model identity and per-slot measurements. The showcase includes a sanitized evidence index with source hashes; it does not distribute the private research history or checkpoint.',y,10)
    y=text('Related work',y,13,True)
    y=text('Mordvintsev et al. (2020), Growing Neural Cellular Automata. Introduces learned local update dynamics; an important conceptual foundation, not evidence that this project inherits regeneration capabilities.',y,10)
    y=text('https://distill.pub/2020/growing-ca/',y,9)
    y=text('Outdoor Scene Extrapolation with Hierarchical Generative Cellular Automata (2024), arXiv:2406.08292. Coarse-to-fine sparse generation and context planning inform future multi-site research. Its outdoor surface-completion task differs from gross building massing; no direct replication is claimed.',y,10)
    y=text('https://arxiv.org/abs/2406.08292',y,9)
    text('Release boundary: this is a locally prepared showcase and technical report. Hosting, public release and any new paid training are separate actions. The original private next-phase review remains excluded.',y,10)
    end(); c.save()
    template=(ROOT/'publication/showcase.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(template,encoding='utf-8')
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(dict(files=files,author='Artin Lesani Abdi',status='local review draft'),indent=2))
    print(json.dumps(dict(output=str(out),pdf_pages=page,slots=len(gallery),frames=sum(len(s['frames']) for s in gallery))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replace-draft',action='store_true')
    args=parser.parse_args(); build(args.output,args.replace_draft)
