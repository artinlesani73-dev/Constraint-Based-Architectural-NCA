 'use strict';
const $=id=>document.getElementById(id);
let data,study,a,b,view='iso';
const labels={aligned:'Aligned connections',wide_gap:'Wider gap',offset_interfaces:'Offset connections',blocked_gap:'Blocked gap',partial_obstruction:'Partial obstruction'};
const families=['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'];
function row(parent,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}parent.append(tr);}
function candidateName(r){return `${Math.round(r.request_fraction*100)}% volume · seed ${r.seed}`;}
function chooseScene(){const previous=$('candidate').value;$('candidate').replaceChildren();for(const r of data.cases.filter(r=>r.scene_case===$('scene').value)){const o=document.createElement('option');o.value=r.case;o.textContent=candidateName(r);$('candidate').append(o);}if([...$('candidate').options].some(o=>o.value===previous))$('candidate').value=previous;refresh();}
function badge(id,pass){$(id).textContent=pass?'Pilot checks met':'Pilot checks unmet';$(id).className='badge '+(pass?'pass':'fail');}
function refresh(){
 b=data.cases.find(r=>r.case===$('candidate').value);const context=data.contexts.find(c=>c.case===b.scene_case);study={scene:context.scene};
 const original=b.original;const oldSet=new Set(original.occupied_zyx.map(c=>c.join(',')));b={...b,added_zyx:b.occupied_zyx.filter(c=>!oldSet.has(c.join(',')))};a={...original,added_zyx:[]};
 $('title-b').textContent=candidateName(b);badge('pass-a',a.targets.contract_pass);badge('pass-b',b.targets.contract_pass);
 const failed=families.filter(f=>!b.targets.family_pass[f]);$('status').textContent=failed.length?`Selected result: unmet checks — ${failed.join(', ')}.`:'Selected result: all nine pilot checks met. Architectural quality is not established.';
 $('caption-b').textContent=`${b.added_count} added cell${b.added_count===1?'':'s'} · ${b.removed_count} removed cell${b.removed_count===1?'':'s'} compared with the previous MG2 result. ${b.method==='contact'?'Fresh procedural generation from the same scene and seed.':'Saved direct-refinement boundary; no clipping or repair.'}`;
 $('families').replaceChildren();for(const f of families)row($('families'),[f,a.targets.family_pass[f]?'Met':'Unmet',b.targets.family_pass[f]?'Met':'Unmet']);
 const pct=x=>(100*x).toFixed(3)+'%';$('metrics').replaceChildren();for(const [label,get]of [
  ['Occupied cells',t=>t.occupied_voxels],['Building volume (m³)',t=>t.gross_volume_m3.toFixed(2)],
  ['Volume / fixed region',t=>pct(t.volume_fraction)],['Deviation from requested volume',t=>((t.volume_fraction-b.request_fraction)*100).toFixed(3)+' percentage points'],
  ['Façade contact / occupied volume',t=>pct(t.facade_contact_fraction)],['Volume meeting local scale',t=>pct(t.bulk_fraction)],
  ['Illegal / outside-region cells',t=>t.illegal_voxels+' / '+t.outside_domain_voxels]])row($('metrics'),[label,get(a.targets),get(b.targets)]);
 $('timing').textContent=`Generator time: ${b.seconds.toFixed(3)} s, including contact-context setup. Evaluation is additional.`;
 $('soft').textContent=`${b.rejected_proposals} proposal checks deferred (including rechecks) when their resulting contact ratio exceeded the limit. Construction: ${b.generation.status.replaceAll('_',' ')}. Requested ${b.generation.requested_voxels} cells; saved ${b.targets.occupied_voxels}. The final block can overshoot the stopping target. A failed route is not a general proof that a site is impossible.`;
 $('context-note').textContent=`Fixed region: ${b.targets.domain_voxels} cells. Same request, seed and evaluation for both panels. Every failure is retained.`;
 $('provenance').textContent=`Saved study ${data.run_id} · MG3_v1. Five contexts, three requested volumes, three seeds. Development examples, not held-out generalization. Earlier MG1, MD1 and MG2 data remains unchanged.`;draw();
}
function draw(){if(!a)return;render($('canvas-a'),a);render($('canvas-b'),b);$('slice-control').hidden=view==='iso';$('cut').disabled=view!=='iso';$('slice-value').textContent=$('slice').value;$('added-legend').hidden=!$('added').checked;$('occupied-legend').textContent=$('added').checked?'Previous / retained cells':'All occupied cells';const s=Number($('slice').value);$('view-note').textContent=view==='iso'?($('cut').checked?'Cutaway: foreground Y ≥ 16 hidden. Full fields evaluated.':'Complete geometry. Full fields evaluated.'):`${view==='xz'?'Vertical Y':'Horizontal Z'} slice ${s} · cell center ${((s+.5)*.8).toFixed(1)} m`;}
function render(canvas,result){
    const ctx=canvas.getContext('2d'),r=canvas.getBoundingClientRect(),w=r.width,h=r.height,dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const slice=Number($('slice').value),scale=Math.min(w/(view==='iso'?49:35),h/(view==='iso'?55:36));
    const p=(x,y,z)=>view==='iso'?[w/2+(x-y)*.866*scale,h*.72+(x+y-32)*.5*scale-z*scale]:view==='xy'?[w/2+(x-16)*scale,h/2+(y-16)*scale]:[w/2+(x-16)*scale,h*.88-z*scale];
    const faces=[];
    function box(x,y,z,dx,dy,dz,colors,alpha=1,mask=null){
        if(view==='iso'&&$('cut').checked){dy=Math.min(y+dy,16)-y;if(dy<=0)return;}
        if(view==='xz'&&!(y<=slice&&slice<y+dy)||view==='xy'&&!(z<=slice&&slice<z+dz))return;
        const open=(ox,oy,oz)=>!mask?.has(`${z+oz},${y+oy},${x+ox}`);
        if(view!=='xz'&&(view==='xy'||open(0,0,dz)))faces.push({pts:[[x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[0],alpha,d:x+y+z+dz});
        if(view!=='xy'&&(view==='xz'||open(0,dy,0)))faces.push({pts:[[x,y+dy,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[1],alpha,d:x+y+dy+z});
        if(view==='iso'&&open(dx,0,0))faces.push({pts:[[x+dx,y,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x+dx,y,z+dz]],color:colors[2],alpha,d:x+dx+y+z});
    }
    function cells(coords,colors,alpha,commonMask=null){const shown=view==='iso'&&$('cut').checked?coords.filter(c=>c[1]<16):coords;const mask=new Set(shown.map(c=>c.join(',')));for(const[z,y,x]of shown)box(x,y,z,1,1,1,colors,alpha,commonMask||mask);}
    function line(a,b){ctx.beginPath();ctx.moveTo(...p(...a));ctx.lineTo(...p(...b));ctx.strokeStyle='#dce2d5';ctx.lineWidth=.5;ctx.stroke();}
    if(view==='xz'){for(let z=0;z<=32;z+=2)line([0,slice,z],[32,slice,z]);}
    else for(let i=0;i<=32;i+=2){line([i,0,0],[i,32,0]);line([0,i,0],[32,i,0]);}
    if($('context').checked)for(const c of study.scene.buildings)box(c.x[0],c.y[0],c.z[0],c.x[1]-c.x[0],c.y[1]-c.y[0],c.z[1]-c.z[0],['#cbd3c2','#b1bdaa','#c1cbb8'],.35);
    const shown=view==='iso'&&$('cut').checked?result.occupied_zyx.filter(c=>c[1]<16):result.occupied_zyx;
    const mask=new Set(shown.map(c=>c.join(',')));
    const additions=new Set(result.added_zyx.map(c=>c.join(',')));
    if($('added').checked){
        cells(result.occupied_zyx.filter(c=>!additions.has(c.join(','))),['#72a07e','#285e4f','#438369'],1,mask);
        cells(result.added_zyx,['#d4b47f','#a37847','#bd955f'],1,mask);
    }else cells(result.occupied_zyx,['#72a07e','#285e4f','#438369'],1,mask);
    faces.sort((a,b)=>view==='xy'?a.pts[0][2]-b.pts[0][2]:view==='xz'?a.pts[0][1]-b.pts[0][1]:a.d-b.d);
    for(const f of faces){ctx.beginPath();f.pts.forEach((q,i)=>i?ctx.lineTo(...p(...q)):ctx.moveTo(...p(...q)));ctx.closePath();ctx.globalAlpha=f.alpha;ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle=f.color;ctx.lineWidth=.4;ctx.stroke();}ctx.globalAlpha=1;
    ctx.font='11px Segoe UI';ctx.fillStyle='#6a7e6e';ctx.fillText('0.8 m / cell',18,h-17);
}




$('scene').onchange=chooseScene;$('candidate').onchange=refresh;['cut','added','context'].forEach(id=>$(id).onchange=draw);$('slice').oninput=draw;
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw();});
new ResizeObserver(draw).observe($('canvas-b').parentElement);
(async()=>{try{const response=await fetch('study.json');if(!response.ok)throw new Error('Saved study unavailable');data=await response.json();if(data.version!=='MG3_view_v1')throw new Error('Unsupported study');for(const c of data.contexts){const o=document.createElement('option');o.value=c.case;o.textContent=labels[c.case]||c.case;$('scene').append(o);}$('scene').value='partial_obstruction';chooseScene();$('candidate').value='partial_obstruction__v24__s2';refresh();}catch(error){$('status').textContent='Could not load saved comparison: '+error.message;}})();
