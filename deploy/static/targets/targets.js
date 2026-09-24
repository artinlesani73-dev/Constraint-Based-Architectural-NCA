'use strict';
const $=id=>document.getElementById(id);
let data,study,a,b,view='iso';
const scenes={aligned:'Aligned connections',wide_gap:'Wider gap',offset_interfaces:'Offset connections',blocked_gap:'Blocked gap'};
const names={compact_mass:'Compact mass',articulated_mass:'Articulated mass',empty:'Empty field',thin_sheet:'Thin sheet',fragmented:'Separated fragments',thin_neck:'One-voxel neck',unsupported:'Unsupported mass',context_collision:'Context collision',ground_intrusion:'Ground intrusion',detached_satellite:'Detached fragment',outside_domain:'Outside study region',excessive_fill:'Excessive filling'};
const notes={compact_mass:'A direct filled-volume control; no cavities required.',articulated_mass:'Overlapping volumes produce a stepped mass with substantial connections.',empty:'An empty field cannot pass merely by avoiding collisions.',thin_sheet:'A wide outline lacks substantial depth.',fragmented:'Removing a complete cross-section disconnects the mass.',thin_neck:'A single-voxel link connects the raw field but not its substantial regions.',unsupported:'The mass has no geometric attachment to the declared support boundary.',context_collision:'The proposal contains occupied cells inside existing context.',ground_intrusion:'A retained occupied cell intrudes into protected ground.',detached_satellite:'A small detached piece must not be hidden by scoring only the main component.',outside_domain:'All original cells are retained, including those outside the declared distribution region.',excessive_fill:'Filling the entire available region exceeds the pilot volume budget.'};
const families=['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'];
function row(parent,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}parent.append(tr);}
function drawable(record){const bulk=new Set(record.bulk_zyx.map(c=>c.join(',')));return {...record,added_zyx:record.occupied_zyx.filter(c=>!bulk.has(c.join(',')))};}
function refresh(){
 const scene=$('scene').value,candidate=$('candidate').value;
 const context=data.contexts.find(c=>c.case===scene);study={scene:context.scene};
 a=drawable(data.cases.find(c=>c.scene_case===scene&&c.case==='compact_mass'));b=drawable(data.cases.find(c=>c.scene_case===scene&&c.case===candidate));
 $('title-b').textContent=names[candidate];$('caption-b').textContent=notes[candidate];
 for(const [id,r]of [['a',a],['b',b]]){const pass=r.targets.contract_pass;$('pass-'+id).textContent=pass?'Pilot checks met':'Pilot checks unmet';$('pass-'+id).className='badge '+(pass?'pass':'fail');}
 const failed=families.filter(f=>!b.targets.family_pass[f]);
 $('status').textContent=b.targets.contract_pass?'Candidate meets all nine pilot checks. This is a constructed control, not learned output.':`Candidate does not meet: ${failed.join(', ')}.${b.targets.context_necessary_checks_pass?'':' The context also fails the necessary connection check.'}`;
 $('families').replaceChildren();$('legacy').replaceChildren();
 for(const f of families){row($('families'),[f,...[a,b].map(c=>c.targets.family_pass[f]?'Met':'Unmet')]);row($('legacy'),[f,...[a,b].map(c=>c.legacy.families[f].toFixed(5))]);}
 $('metrics').replaceChildren();for(const [label,f]of [['Occupied cells',r=>r.occupied_voxels],['Building volume (m³)',r=>r.gross_volume_m3.toFixed(2)],['Volume / fixed region',r=>(r.volume_fraction*100).toFixed(2)+'%'],['Volume meeting local scale',r=>(r.bulk_fraction*100).toFixed(2)+'%'],['Unconnected occupied cells',r=>r.unreached_occupied_voxels],['Substantial volume in X thirds',r=>r.thirds.map(t=>t.fraction===null?'N/A':(100*t.fraction).toFixed(1)+'%').join(' / ')],['Illegal occupied cells',r=>r.illegal_voxels],['Outside region (cells)',r=>r.outside_domain_voxels]])row($('metrics'),[label,f(a.targets),f(b.targets)]);
 $('context-note').textContent=`Fixed region: ${context.domain.domain_voxels} cells. Necessary substantial-connection check: ${b.targets.context_necessary_checks_pass?'met; not a proof of full feasibility':'unmet'}.`;
 $('provenance').textContent=`Saved study ${data.run_id} · ${data.version}. 48 base controls and 432 archived sensitivity evaluations. No loss replacement, model promotion or training.`;draw();
}
function draw(){if(!a)return;render($('canvas-a'),a);render($('canvas-b'),b);$('slice-control').hidden=view==='iso';$('cut').disabled=view!=='iso';$('slice-value').textContent=$('slice').value;$('added-legend').hidden=!$('added').checked;$('occupied-legend').textContent=$('added').checked?'Volume meeting the local scale':'All occupied cells';const s=Number($('slice').value);$('view-note').textContent=view==='iso'?($('cut').checked?'Cutaway: foreground Y ≥ 16 hidden. Full fields are evaluated.':'Complete geometry. Full fields are evaluated.'):`${view==='xz'?'Vertical Y':'Horizontal Z'} slice ${s} · cell center ${((s+.5)*.8).toFixed(1)} m`;}
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


['scene','candidate'].forEach(id=>$(id).onchange=refresh);['cut','added','context'].forEach(id=>$(id).onchange=draw);$('slice').oninput=draw;
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw();});
new ResizeObserver(draw).observe($('canvas-b').parentElement);
(async()=>{try{const response=await fetch('study.json');if(!response.ok)throw new Error('Saved study unavailable');data=await response.json();if(data.version!=='massing_targets_audit_v1')throw new Error('Unsupported study');
for(const [id,entries]of [['scene',Object.entries(scenes)],['candidate',Object.entries(names)]])for(const [value,label]of entries){const option=document.createElement('option');option.value=value;option.textContent=label;$(id).append(option);}
$('candidate').value='articulated_mass';refresh();}catch(error){$('status').textContent='Could not load target audit: '+error.message;}})();
