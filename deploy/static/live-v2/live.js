 'use strict';
const $=id=>document.getElementById(id);
const labels={aligned:'Aligned connections',wide_gap:'Wider gap',offset_interfaces:'Offset connections',blocked_gap:'Blocked gap',partial_obstruction:'Partial obstruction'};
const families=['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'];
let presets=[],selected=null,pair=null,view='iso',polling=false,autoJob=null,knownRecords='';
const pct=n=>(100*n).toFixed(3)+'%';
function versionName(r){return r.version==='studio_mass_v1'?'Original':'Incremental';}
function name(r){const q=r.mass_request;return `${(labels[q.scene_case]||q.scene_case)} · ${Math.round(q.request_fraction*100)}% · seed ${q.seed} · ${versionName(r)}`;}
function row(parent,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td);}parent.append(tr);}
async function api(path,body){const response=await fetch('/api/mass-v2'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Invalid request. Check the site, volume and integer seed.');return data;}
function status(text){$('status').textContent=text;}
function onError(e){status(e.message);}
function show(record){const preset=presets.find(p=>p.case===record.mass_request.scene_case);if(preset){$('site').value=preset.case;preview();if(preset.requests.includes(record.mass_request.request_fraction))$('volume').value=record.mass_request.request_fraction;if(preset.seeds.includes(record.mass_request.seed))$('seed').value=record.mass_request.seed;}selected=record;$('evidence').hidden=false;$('record-label').textContent='SAVED BUILDING MASS';$('result-title').textContent=name(record);const t=record.targets;const badge=$('result-badge');badge.textContent=t.contract_pass?'All nine checks met':'Pilot checks unmet';badge.className='badge '+(t.contract_pass?'pass':'fail');$('gross').textContent=t.gross_volume_m3.toFixed(2)+' m³';$('cells').textContent=t.occupied_voxels+' / '+record.generation.requested_voxels;$('contact').textContent=pct(t.facade_contact_fraction);$('bulk').textContent=pct(t.bulk_fraction);$('families').replaceChildren();for(const f of families)row($('families'),[f,t.family_pass[f]?'Met':'Unmet']);$('termination').textContent='Generation: '+record.generation.status.replaceAll('_',' ');$('request-note').textContent=`${pct(t.volume_fraction)} of the fixed ${t.domain_voxels}-cell region. Requested ${Math.round(record.mass_request.request_fraction*100)}%. Difference: ${record.generation.target_error_voxels} cells. The final block can overshoot the request.`;$('identity').textContent=`Saved ${record.created_at} · ${record.id}. Generator ${record.method}; evaluation ${t.version}. Full source and growth decisions are retained.`;status(t.contract_pass?'Saved locally. All nine pilot checks met.':'Saved locally, including the failed checks. Review the result before using it.');draw();}
async function openRecord(id){show(await api('/records/'+encodeURIComponent(id)));}
function preview(){
 const p=presets.find(p=>p.case===$('site').value);if(!p)return;
 for(const [id,values] of [['volume',p.requests],['seed',p.seeds]]){
   const prior=Number($(id).value);$(id).replaceChildren();
   for(const value of values){const o=document.createElement('option');o.value=value;o.textContent=id==='volume'?Math.round(value*100)+'% of the fixed region':String(value);$(id).append(o);}
   if(values.includes(prior))$(id).value=prior;
 }
 $('site-note').textContent=p.domain_voxels+' cells in the fixed generation region. Evaluated development site.';
 $('scale-note').textContent=`${p.size} × ${p.size} × ${p.size} cells\n${(p.size*p.scene.voxel_size_m).toFixed(1)} m world span\n0.8 m per cell · 2.4 m growth blocks`;
 selected=null;$('evidence').hidden=true;$('record-label').textContent='SITE PREVIEW';$('result-badge').textContent='No generated result';$('result-badge').className='badge';$('result-title').textContent=p.label;status('Site preview ready. Generate to create a saved result.');draw();
}
function draw(){
 if(!presets.length)return;const p=presets.find(p=>p.case===$('site').value);
 const result=selected||{scene:p.scene,occupied_zyx:[]};const n=result.scene.grid_size;
 $('slice').max=n-1;if(Number($('slice').value)>=n)$('slice').value=n-1;
 render($('canvas'),result);if(pair){render($('canvas-a'),pair.a);render($('canvas-b'),pair.b);}
 $('slice-control').hidden=view==='iso';$('cut').disabled=view!=='iso';$('slice-value').textContent=$('slice').value;
 const s=Number($('slice').value);
 $('view-note').textContent=view==='iso'?($('cut').checked?`Cutaway hides the foreground half of each site. Full fields are evaluated.`:`Complete geometry · ${n}³ cells · ${(n*result.scene.voxel_size_m).toFixed(1)} m world span.`):`${view==='xz'?'Vertical Y':'Horizontal Z'} slice ${s} · cell center ${((s+.5)*result.scene.voxel_size_m).toFixed(1)} m. Comparisons use the same cell index, clamped per site.`;
}
async function refresh(){if(polling)return;polling=true;try{const [j,l]=await Promise.all([api('/jobs'),api('/records')]);const issues=[...j.integrity_issues,...l.integrity_issues];$('issues').hidden=!issues.length;$('issues').textContent='Some local records need inspection: '+issues.join(', ');$('jobs').replaceChildren();for(const job of j.jobs){const el=document.createElement('div');el.className='job';const info=document.createElement('span');info.textContent=`${name(job)} — ${job.stage}`;el.append(info);if(job.detail.reason){const note=document.createElement('small');note.textContent=job.detail.reason;el.append(note);}if(['queued','running'].includes(job.state)){const btn=document.createElement('button');btn.textContent='Cancel';btn.onclick=async()=>{try{await api('/jobs/'+job.id+'/cancel',{});status('Attempt stopped. Its history is retained.');await refresh();}catch(e){onError(e);}};el.append(btn);}if(['cancelled','interrupted','failed'].includes(job.state)){const btn=document.createElement('button');btn.textContent='Retry same request';btn.onclick=async()=>{try{const next=await api('/jobs/'+job.id+'/retry',{});autoJob=next.id;status('Linked retry queued with the same site, volume and seed.');await refresh();}catch(e){onError(e);}};el.append(btn);}if(job.state==='completed'){const btn=document.createElement('button');btn.textContent='View result';btn.onclick=()=>openRecord(job.id).catch(onError);el.append(btn);} $('jobs').append(el);}
const signature=JSON.stringify(l.records.map(r=>r.id));if(signature!==knownRecords){knownRecords=signature;$('library').replaceChildren();for(const r of l.records){const b=document.createElement('button');b.className='card';const title=document.createElement('strong');title.textContent=(labels[r.mass_request.scene_case]||r.mass_request.scene_case)+' · '+versionName(r);const detail=document.createElement('span');detail.textContent=`${Math.round(r.mass_request.request_fraction*100)}% · seed ${r.mass_request.seed} · ${r.targets.gross_volume_m3.toFixed(2)} m³`;const verdict=document.createElement('span');verdict.className='badge '+(r.targets.contract_pass?'pass':'fail');verdict.textContent=r.targets.contract_pass?'All nine checks met':'Pilot checks unmet';const stamp=document.createElement('span');stamp.className='small muted';stamp.textContent=r.created_at+' · '+r.id;b.append(title,detail,verdict,stamp);b.onclick=()=>openRecord(r.id).catch(onError);$('library').append(b);}if(!l.records.length)$('library').textContent='No generated volumes yet. Your first result will be saved here.';for(const id of ['compare-a','compare-b']){const old=$(id).value;$(id).replaceChildren();for(const r of l.records){const o=document.createElement('option');o.value=r.id;o.textContent=name(r)+' · '+r.id;$(id).append(o);}if(l.records.some(r=>r.id===old))$(id).value=old;}if(l.records.length>1&&$('compare-a').value===$('compare-b').value)$('compare-b').selectedIndex=1;}
$('compare').disabled=l.records.length<2;if(autoJob){const job=j.jobs.find(j=>j.id===autoJob);if(job?.state==='completed'){const id=autoJob;autoJob=null;await openRecord(id);}else if(job&&['failed','cancelled','interrupted'].includes(job.state)){autoJob=null;status('Attempt '+job.state+'. History retained; a retry preserves the request.');}}
}catch(e){onError(e);}finally{polling=false;}}
$('generate').onclick=async()=>{const seed=Number($('seed').value);if(!$('seed').value||!Number.isSafeInteger(seed)||seed<0||seed>2147483647){status('Enter an integer seed between 0 and 2147483647.');return;}$('generate').disabled=true;try{const job=await api('/jobs',{scene_case:$('site').value,request_fraction:Number($('volume').value),seed});autoJob=job.id;status('Generation queued. You can leave this page; the local job history is retained.');await refresh();}catch(e){onError(e);}finally{$('generate').disabled=false;}};
$('compare').onclick=async()=>{try{pair=await api('/compare?a='+encodeURIComponent($('compare-a').value)+'&b='+encodeURIComponent($('compare-b').value));$('comparison').hidden=false;$('caption-a').textContent='A / '+name(pair.a);$('caption-b').textContent='B / '+name(pair.b);$('compare-note').textContent=`${pair.note} ${pair.same_context?pair.added_voxels+' added cells · '+pair.removed_voxels+' removed cells.':''}`;$('compare-metrics').replaceChildren();for(const f of families)row($('compare-metrics'),[f,pair.a.targets.family_pass[f]?'Met':'Unmet',pair.b.targets.family_pass[f]?'Met':'Unmet']);row($('compare-metrics'),['Building volume (m³)',pair.a.targets.gross_volume_m3.toFixed(2),pair.b.targets.gross_volume_m3.toFixed(2)]);row($('compare-metrics'),['Façade contact',pct(pair.a.targets.facade_contact_fraction),pct(pair.b.targets.facade_contact_fraction)]);draw();}catch(e){onError(e);}};
$('export').onclick=async()=>{if(!selected)return;try{const response=await fetch('/api/mass-v2/records/'+selected.id+'/export');if(!response.ok)throw new Error('Export failed; the saved record remains available.');const raw=await response.text();const url=URL.createObjectURL(new Blob([raw],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='NCA-mass-'+selected.id+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);status('Export prepared with geometry, growth decisions and retained source bytes.');}catch(e){onError(e);}};
$('import').onchange=async()=>{const file=$('import').files[0];if(!file)return;try{if(file.size>20000000)throw new Error('Export exceeds the 20 MB import limit.');status('Verifying source hashes and replaying the imported generation…');const response=await fetch('/api/mass-v2/import',{method:'POST',headers:{'Content-Type':'application/json'},body:await file.text()});const result=await response.json();if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:'Import could not be verified.');autoJob=result.id;status('Import replay queued. You can cancel it below; a successful replay creates a new saved attempt.');await refresh();}catch(e){onError(e);}finally{$('import').value='';}};
$('site').onchange=preview;$('refresh').onclick=refresh;['cut','context'].forEach(id=>$(id).onchange=draw);$('slice').oninput=draw;document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw();});new ResizeObserver(draw).observe($('canvas').parentElement);
(async()=>{try{const data=await api('/presets');presets=data.contexts;for(const p of presets){const o=document.createElement('option');o.value=p.case;labels[p.case]=p.label;o.textContent=p.label;$('site').append(o);}$('site').value='partial_obstruction';preview();$('generate').disabled=false;status('Ready to generate a building volume.');await refresh();setInterval(refresh,2000);}catch(e){onError(e);}})();

function render(canvas,result){
    const ctx=canvas.getContext('2d'),r=canvas.getBoundingClientRect(),w=r.width,h=r.height,dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const n=result.scene.grid_size,half=n/2,slice=Math.min(Number($('slice').value),n-1),scale=Math.min(w/(view==='iso'?n*1.53:n+3),h/(view==='iso'?n*1.72:n+4));
    const p=(x,y,z)=>view==='iso'?[w/2+(x-y)*.866*scale,h*.72+(x+y-n)*.5*scale-z*scale]:view==='xy'?[w/2+(x-half)*scale,h/2+(y-half)*scale]:[w/2+(x-half)*scale,h*.88-z*scale];
    const faces=[];
    function box(x,y,z,dx,dy,dz,colors,alpha=1,mask=null){
        if(view==='iso'&&$('cut').checked){dy=Math.min(y+dy,half)-y;if(dy<=0)return;}
        if(view==='xz'&&!(y<=slice&&slice<y+dy)||view==='xy'&&!(z<=slice&&slice<z+dz))return;
        const open=(ox,oy,oz)=>!mask?.has(`${z+oz},${y+oy},${x+ox}`);
        if(view!=='xz'&&(view==='xy'||open(0,0,dz)))faces.push({pts:[[x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[0],alpha,d:x+y+z+dz});
        if(view!=='xy'&&(view==='xz'||open(0,dy,0)))faces.push({pts:[[x,y+dy,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[1],alpha,d:x+y+dy+z});
        if(view==='iso'&&open(dx,0,0))faces.push({pts:[[x+dx,y,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x+dx,y,z+dz]],color:colors[2],alpha,d:x+dx+y+z});
    }
    function cells(coords,colors,alpha,commonMask=null){const shown=view==='iso'&&$('cut').checked?coords.filter(c=>c[1]<half):coords;const mask=new Set(shown.map(c=>c.join(',')));for(const[z,y,x]of shown)box(x,y,z,1,1,1,colors,alpha,commonMask||mask);}
    function line(a,b){ctx.beginPath();ctx.moveTo(...p(...a));ctx.lineTo(...p(...b));ctx.strokeStyle='#dce2d5';ctx.lineWidth=.5;ctx.stroke();}
    if(view==='xz'){for(let z=0;z<=n;z+=2)line([0,slice,z],[n,slice,z]);}
    else for(let i=0;i<=n;i+=2){line([i,0,0],[i,n,0]);line([0,i,0],[n,i,0]);}
    if($('context').checked)for(const c of result.scene.buildings)box(c.x[0],c.y[0],c.z[0],c.x[1]-c.x[0],c.y[1]-c.y[0],c.z[1]-c.z[0],['#cbd3c2','#b1bdaa','#c1cbb8'],.35);
    const shown=view==='iso'&&$('cut').checked?result.occupied_zyx.filter(c=>c[1]<half):result.occupied_zyx;
    const mask=new Set(shown.map(c=>c.join(',')));
    cells(result.occupied_zyx,['#72a07e','#285e4f','#438369'],1,mask);
    faces.sort((a,b)=>view==='xy'?a.pts[0][2]-b.pts[0][2]:view==='xz'?a.pts[0][1]-b.pts[0][1]:a.d-b.d);
    for(const f of faces){ctx.beginPath();f.pts.forEach((q,i)=>i?ctx.lineTo(...p(...q)):ctx.moveTo(...p(...q)));ctx.closePath();ctx.globalAlpha=f.alpha;ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle=f.color;ctx.lineWidth=.4;ctx.stroke();}ctx.globalAlpha=1;
    ctx.font='11px Segoe UI';ctx.fillStyle='#6a7e6e';ctx.fillText(`${n}³ · ${result.scene.voxel_size_m} m / cell`,18,h-17);
}




