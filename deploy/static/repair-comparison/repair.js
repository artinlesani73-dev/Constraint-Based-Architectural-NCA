'use strict';
const $=id=>document.getElementById(id);let data,study,selected,view='iso';
const families=['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'];
function row(parent,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}parent.append(tr);}
function refresh(){selected=data.cases[Number($('example').value)];study={scene:selected.scene};const d=selected.diagnosis;$('diagnosis').textContent=d.unmet.length?`Unmet: ${d.unmet.join(', ')}. ${d.raw_unreached} disconnected cells; ${d.unreached_excess} outside the target. Both interfaces remain connected through the main bulk. Unsupported cells: ${d.unsupported_voxels}. Bulk fraction: ${(100*d.bulk_fraction).toFixed(2)}% (minimum 90%).`:'NR4 meets all nine checks on this example. Shape preservation is reported separately below.';$('status').textContent=selected.label+' · Same camera and scale in every panel.';
selected.results.forEach((r,i)=>{$('badge-'+i).textContent=r.metrics.targets.contract_pass?'Nine checks met':'Checks unmet';$('badge-'+i).className='badge '+(r.metrics.targets.contract_pass?'pass':'fail');$('caption-'+i).textContent=`${r.occupied_zyx.length} occupied cells · ${r.added_zyx.length} excess · ${r.missing_zyx.length} missing relative to target`;});
$('metrics').replaceChildren();for(const [label,get] of [['Overlap with target',m=>(100*m.iou).toFixed(2)+'%'],['Volume (m³)',m=>m.targets.gross_volume_m3.toFixed(1)],['Requested-volume error (cells)',m=>m.request_error_cells],['Surviving input cells removed',m=>m.surviving_cells_removed],['Missing input cells recovered',m=>m.recovered_cells]])row($('metrics'),[label,...selected.results.map(r=>get(r.metrics))]);
$('families').replaceChildren();for(const f of families)row($('families'),[f,...selected.results.map(r=>r.metrics.targets.family_pass[f]?'Met':'Unmet')]);
$('provenance').textContent=`NR3 run ${data.run_id}; NR4 run ${data.nr4_run_id} · checkpoint 256 · seed 1201 · 32 rollout steps · firing seed 2101 · CPU-scored validation. NR3 SHA256: ${data.checkpoint_sha256}; NR4 SHA256: ${data.nr4_checkpoint_sha256}. Reference target is a procedural teacher, not an architectural ground truth.`;draw();}
function draw(){if(!selected)return;$('volume-legend').textContent=$('added').checked?'Volume matching the target':'All occupied volume';selected.results.forEach((r,i)=>render($('canvas-'+i),r));$('slice-control').hidden=view==='iso';$('cut').disabled=view!=='iso';$('slice-value').textContent=$('slice').value;$('view-note').textContent=view==='iso'?($('cut').checked?'Foreground Y ≥ 16 hidden; metrics use full geometry.':'Complete geometry; excess highlight can be toggled.'):`${view==='xz'?'Y':'Z'} slice ${$('slice').value}; metrics use full geometry.`;}
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
        cells(result.added_zyx,['#d59b72','#9f4e32','#b56a43'],1,mask);
    }else cells(result.occupied_zyx,['#72a07e','#285e4f','#438369'],1,mask);
    if($('missing').checked)cells(result.missing_zyx,['#b4d1e0','#7399b0','#91b6cc'],.38);
    if($('isolated').checked && result.isolated_zyx)cells(result.isolated_zyx,['#f597cf','#9f236a','#bb438a'],1);
    faces.sort((a,b)=>view==='xy'?a.pts[0][2]-b.pts[0][2]:view==='xz'?a.pts[0][1]-b.pts[0][1]:a.d-b.d);
    for(const f of faces){ctx.beginPath();f.pts.forEach((q,i)=>i?ctx.lineTo(...p(...q)):ctx.moveTo(...p(...q)));ctx.closePath();ctx.globalAlpha=f.alpha;ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle=f.color;ctx.lineWidth=.4;ctx.stroke();}ctx.globalAlpha=1;
    ctx.font='11px Segoe UI';ctx.fillStyle='#6a7e6e';ctx.fillText('0.8 m / cell',18,h-17);
}





$('example').onchange=refresh;['added','missing','context','cut','isolated'].forEach(id=>$(id).onchange=draw);$('slice').oninput=draw;
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw();});
new ResizeObserver(draw).observe($('canvas-0').parentElement);
function populate(){const previous=$('example').value;$('example').replaceChildren();data.cases.forEach((r,i)=>{if($('failures').checked&&!r.diagnosis.unmet.length)return;const o=document.createElement('option');o.value=i;o.textContent=r.label;$('example').append(o);});if([...$('example').options].some(o=>o.value===previous))$('example').value=previous;refresh();}
$('failures').onchange=populate;
(async()=>{try{const response=await fetch('study.json');if(!response.ok)throw Error('Evidence unavailable');data=await response.json();if(data.version!=='NR34_comparison_v1'||data.cases.length!==27)throw Error('Incomplete review');populate();}catch(e){$('status').textContent='Could not load comparison: '+e.message;}})();
