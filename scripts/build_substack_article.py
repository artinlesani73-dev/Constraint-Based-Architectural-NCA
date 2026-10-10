"""Render the local Substack draft without third-party assets or account writes."""
from pathlib import Path
import argparse, html, re, shutil, json, hashlib, zipfile

ROOT=Path(__file__).resolve().parents[1]
EQUATIONS=[
 '(z<sub>t</sub>, Δh<sub>t</sub>) = f<sub>θ</sub>(Φ([m<sub>t</sub>, h<sub>t</sub>]), Φ(c), r<sub>t</sub>)',
 'm<sub>t+1</sub> = 𝒜(m<sub>t</sub>, σ(z<sub>t</sub>), F<sub>t</sub>, Ω, C<sub>t</sub>)',
 'ℒ<sub>base</sub> = ℒ<sub>frontier</sub> + 0.25 ℒ<sub>volume</sub> + ℒ<sub>band</sub>',
 'ℒ<sub>rank</sub> = log(1 + (1 / |A||B|) ∑<sub>a∈A</sub> ∑<sub>b∈B</sub> exp(1 + z<sub>b</sub> − z<sub>a</sub>))']

def inline(text):
    text=html.escape(text)
    text=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2">',text)
    text=re.sub(r'\[([^]]*)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    text=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',text)
    replacements={'m_t':'m<sub>t</sub>','h_t':'h<sub>t</sub>','r_t':'r<sub>t</sub>','z_t':'z<sub>t</sub>','F_t':'F<sub>t</sub>','C_t':'C<sub>t</sub>','z_b':'z<sub>b</sub>','\\Phi':'Φ','\\Omega':'Ω'}
    def math(m):
        s=m.group(1)
        for a,b in replacements.items():s=s.replace(a,b)
        return '<i>'+s+'</i>'
    return re.sub(r'\$([^$]+)\$',math,text)

def build(out):
    if (out/'publication-manifest.json').exists():
        raise FileExistsError('Preserve the completed kit; build into a new output version.')
    source=ROOT/'publication/substack-phase2.md';text=source.read_text(encoding='utf-8')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,out/'Substack-Phase-2.md')
    tex=re.findall(r'\$\$\s*(.*?)\s*\$\$',text,re.S)
    (out/'equations.tex').write_text('\n\n'.join('\\[\n'+t+'\n\\]' for t in tex),encoding='utf-8')
    index=iter(EQUATIONS)
    body=[]
    for chunk in text.split('\n\n'):
        if chunk.startswith('$$'): body.append('<div class="equation">'+next(index)+'</div>')
        elif chunk.startswith('# '):body.append('<h1>'+inline(chunk[2:])+'</h1>')
        elif chunk.startswith('## '):body.append('<h2>'+inline(chunk[3:])+'</h2>')
        elif chunk.startswith('|'):
            lines=chunk.splitlines();body.append('<div class="table"><table>')
            for i,line in enumerate(lines):
                if i==1:continue
                tag='th' if i==0 else 'td';body.append('<tr>'+''.join(f'<{tag}>'+inline(c.strip())+f'</{tag}>' for c in line.strip('|').split('|'))+'</tr>')
            body.append('</table></div>')
        elif chunk.startswith('- '):body.append('<ul>'+''.join('<li>'+inline(x[2:])+'</li>' for x in chunk.splitlines())+'</ul>')
        else:body.append('<p>'+inline(chunk)+'</p>')
    css='''body{margin:0;background:#faf9f4;color:#25383a;font:19px/1.8 Georgia,serif}header{padding:16px 6%;background:#17272c;color:#c8ddd4;font:13px system-ui}main{max-width:800px;margin:65px auto;padding:0 24px}h1{font:600 clamp(36px,5vw,60px)/1.12 system-ui;letter-spacing:-.045em}h2{font:550 29px/1.25 system-ui;margin-top:65px}a{color:#247d70;text-underline-offset:4px}img{display:block;width:calc(100% + 120px);max-width:calc(100vw - 35px);margin:35px 0 20px -60px;border:1px solid #e1e6df}p:has(>em:only-child){font-size:15px;color:#5e6b68;line-height:1.6}.equation{overflow-x:auto;padding:25px;background:#edf0e8;font:20px/1.8 Georgia;margin:30px 0}.table{overflow-x:auto}table{border-collapse:collapse;font:15px/1.5 system-ui}td,th{padding:15px;border-bottom:1px solid #ccd6cc;text-align:left}footer{margin-top:70px;padding:25px 0;border-top:1px solid #ccd6cc;font:14px system-ui}@media(max-width:940px){img{width:100%;margin-left:0}body{font-size:18px}}'''
    (out/'Substack-Phase-2.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Growing architectural volume | Artin Lesani Abdi</title><style>'+css+'</style><header>LOCAL SUBSTACK DRAFT · ARTIN LESANI ABDI · NOT PUBLISHED</header><main>'+''.join(body)+'<footer><a href="NCA-Technical-Report.pdf" download>Download the companion technical report</a></footer></main></html>',encoding='utf-8')
    shutil.copy2(Path('C:/Users/artin/Documents/Codex/outputs/NCA-Showcase-2026-10-10/dist/NCA-Technical-Report.pdf'),out/'NCA-Technical-Report.pdf')
    guide='''SUBSTACK PUBLICATION KIT

Author: Artin Lesani Abdi. Local draft only; nothing uploaded or published.

1. Open Substack-Phase-2.html to review the complete article and figures.
2. Use Substack-Phase-2.md as the editable text source. Upload the five PNGs from figures-v2 in their numbered positions, retaining captions. SVGs are vector masters. Equations.tex contains editable LaTeX source; the mathematics PNG is a portable image fallback. The HTML displays equivalent equations using text and subscripts.
3. Attach NCA-Technical-Report.pdf through your chosen publication workflow. This copy is byte-identical to the reviewed showcase report; that report has not been silently revised.
4. Review title, voice and claims, then publish manually or authorize a specific publishing action. Repository links are public; the showcase itself does not yet have a hosted URL. Do not paste local file paths into a public post.

Evidence: figures-v2/figure-provenance.json records hashes for all1281 archive-matched later update logs, the original500-observation history and evaluation summaries. CSVs reproduce plotted observations. Smoothing is a trailing25-observation mean. Training loss is not a current acceptance metric. R3 reuses G10 weights; neutral has no training run. There is no validation-loss time series in this kit: the supplied curves are training observations, while the geometric evaluations use saved final-checkpoint outputs.

The original private next-phase review, checkpoints, runtime identities and private studio history are excluded. No external fonts, scripts, telemetry or login required to view this draft. Earlier figure authoring versions remain local and are excluded from the publication ZIP.
'''
    (out/'START-HERE.txt').write_text(guide,encoding='utf-8')
    payload=[out/'Substack-Phase-2.md',out/'Substack-Phase-2.html',out/'equations.tex',out/'START-HERE.txt',out/'NCA-Technical-Report.pdf']+list((out/'figures-v2').glob('*.png'))+list((out/'figures-v2').glob('*.svg'))+list((out/'figures-v2').glob('*.csv'))+[out/'figures-v2/figure-provenance.json']
    manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in payload}
    (out/'publication-manifest.json').write_text(json.dumps(manifest,indent=2));payload.append(out/'publication-manifest.json')
    with zipfile.ZipFile(out/'NCA-Substack-Publication-Kit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in payload:z.write(p,p.relative_to(out))
    print(json.dumps({'output':str(out),'article_words':len(text.split()),'figure_count':5,'equations':len(tex),'zip_bytes':(out/'NCA-Substack-Publication-Kit.zip').stat().st_size}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
