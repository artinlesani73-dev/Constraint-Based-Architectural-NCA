'use strict';
const $=id=>document.getElementById(id);
let study,view='iso',a,b;
const names={empty:'Empty field',w1_scaffold:'W1 connection scaffold',sp1_platform:'SP1 flat platform',slab_512:'Flat slab',solid_512:'Solid block',open_ends_512:'Open-ended shell',side_aperture_487:'Shell with side opening',closed_shell_610:'Closed shell',extent_decoy_8:'Eight scattered corners',detached_plates:'Two detached plates',separated_blocks:'Two separated blocks'};
const captions={empty:'No occupied source cells.',w1_scaffold:'Historical procedural connection scaffold; not trained NCA output.',sp1_platform:'Historical platform control, one voxel deep.',slab_512:'A shallow slab, two voxels deep.',solid_512:'A compact block, already filled.',open_ends_512:'A hollow shell with two open ends.',side_aperture_487:'An open-ended shell with an additional side opening.',closed_shell_610:'A shell with a completely enclosed cavity.',extent_decoy_8:'Eight disconnected corner cells with large outer extents.',detached_plates:'Two separate plates with a deliberate vertical gap.',separated_blocks:'Two solid blocks separated by a horizontal gap.'};
const methods={identity:'Unchanged geometry',sealed_cavities:'Enclosed cavities only',axis_span:'Vertical gap completion'};
const explanations={identity:'Keep every original cell. This is the comparison baseline; changing the meaning of a voxel does not change its geometry.',sealed_cavities:'Fill empty regions fully enclosed by the original form. Openings to the grid exterior remain open. Existing buildings do not supply enclosure.',axis_span:'On each vertical line, fill empty gaps up to 6.4 m between consecutive occupied cells. Apply once, inside the study region and outside existing buildings. This is an explicit geometric rule, not an inferred interior.'};
function row(parent,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td);}parent.append(tr);}
function refresh(){
    const name=$('source').value,method=$('method').value;
    a=study.cases.find(c=>c.case===name&&c.method==='identity');b=study.cases.find(c=>c.case===name&&c.method===method);
    $('title-a').textContent=names[name];$('title-b').textContent=methods[method];
    $('count-a').textContent=a.massing.occupied_voxels+' cells';$('count-b').textContent=b.massing.occupied_voxels+' cells';
    $('caption-a').textContent=captions[name];$('caption-b').textContent=`${b.operation.added_voxels} cells added · ${b.operation.rejected_additions} proposed additions blocked. Interiors remain unspecified.`;
    $('status').textContent=`${a.massing.occupied_voxels} original + ${b.operation.added_voxels} added = ${b.massing.occupied_voxels} occupied cells. ${b.operation.added_voxels===0?'This operation leaves the geometry unchanged.':'Both the original and derived fields are preserved.'}`;
    $('operation-title').textContent=methods[method];$('operation-description').textContent=explanations[method];
    let note='This is one geometric control. Its result does not establish a general rule for building design.';
    if(['empty','sp1_platform','slab_512','w1_scaffold'].includes(name))note='Filling cannot invent missing vertical extent. A shallow source can remain shallow even when every available gap is filled.';
    if(name==='solid_512')note='A filled block is a legitimate massing representation; it need not contain modeled rooms. It still needs evaluation for placement, distribution and the project’s constraints.';
    if(['open_ends_512','side_aperture_487'].includes(name))note=method==='sealed_cavities'?'An open form has no fully enclosed cavity to fill. Choosing a building mass from this source requires an additional explicit interpretation.':'Vertical completion fills this shell, including its openings. That is a deliberate rule, not proof that all these cells were intended as interior.';
    if(name==='closed_shell_610')note='The enclosed cavity can be converted into building volume without increasing the outer extents. This is the clearest cavity-filling control.';
    if(['detached_plates','separated_blocks'].includes(name))note='Counterexample: vertical completion joins these separate parts into the same block as the shell. The rule cannot distinguish an intended gap from a missing interior.';
    if(name==='extent_decoy_8')note='Large outer extents do not establish substantial mass. Vertical completion can produce isolated posts from corner pairs without generating a coherent building volume.';
    $('case-note').textContent=note;
    $('metrics').replaceChildren();
    const rows=[['Occupied cells',r=>r.occupied_voxels],['Occupied volume (m³)',r=>r.gross_volume_m3.toFixed(2)],['Extents X × Y × Z (m)',r=>[...r.extent_m_zyx].reverse().map(v=>v.toFixed(1)).join(' × ')],['Connected components',r=>r.components_6],['Inside study region',r=>(r.domain_occupancy_fraction*100).toFixed(2)+'%'],['Outside study region (cells)',r=>r.outside_domain_voxels],['Context collisions (cells)',r=>r.context_collision_voxels]];
    for(const[label,f]of rows)row($('metrics'),[label,f(a.massing),f(b.massing)]);
    $('domain-description').textContent=`The fixed study region contains ${study.domain.domain_voxels.toLocaleString()} cells (${study.domain.domain_volume_m3.toFixed(2)} m³), compared with ${a.legacy.envelope_voxels.toLocaleString()} cells in the historical route envelope.`;
    $('legacy').replaceChildren();const legacy=[a.legacy,b.legacy];
    row($('legacy'),['Old occupied-cell ratio',...legacy.map(r=>(r.material_ratio*100).toFixed(2)+'%')]);
    row($('legacy'),['Old 3–12% budget',...legacy.map(r=>r.in_budget?'Met':'Unmet')]);
    row($('legacy'),['Occupied interface connectivity',...legacy.map(r=>r.connectivity.all_connected?'Connected':'Disconnected')]);
    for(const f of ['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'])row($('legacy'),[f,...legacy.map(r=>r.families[f].toFixed(5))]);
    $('provenance').textContent=`Saved study ${study.run_id} · ${study.version} · ${study.cases.length} original/operation records. Source VA1 ${study.recipe.source_run}. No new training or accepted massing objective.`;
    draw();
}
function draw(){
    if(!a)return;render($('canvas-a'),a);render($('canvas-b'),b);
    $('slice-control').hidden=view==='iso';$('cut').disabled=view!=='iso';$('slice-value').textContent=$('slice').value;$('added-legend').hidden=!$('added').checked;
    $('occupied-legend').textContent=$('added').checked?'Original occupied cells':'All occupied cells';
    const s=Number($('slice').value);
    $('view-note').textContent=view==='iso'?($('cut').checked?'Cutaway: foreground Y ≥ 16 hidden. Metrics use full fields.':'Complete form. Metrics use full fields.'):`${view==='xz'?'Vertical slice at Y':'Horizontal slice at Z'} = ${s} · cell-center coordinate ${((s+.5)*study.scene.voxel_size_m).toFixed(1)} m`;
}
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

['source','method'].forEach(id=>$(id).onchange=refresh);
['cut','added','context'].forEach(id=>$(id).onchange=draw);$('slice').oninput=draw;
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw();});
new ResizeObserver(draw).observe($('canvas-b').parentElement);
(async()=>{try{
    const response=await fetch('study.json');if(!response.ok)throw new Error('Saved study unavailable');study=await response.json();
    if(study.version!=='massing_audit_v1'||study.scene.grid_size!==32||study.scene.voxel_size_m!==.8)throw new Error('Unsupported saved study geometry');
    for(const c of study.cases.filter(c=>c.method==='identity')){const option=document.createElement('option');option.value=c.case;option.textContent=names[c.case];$('source').append(option);}
    $('source').value='open_ends_512';refresh();
}catch(error){$('status').textContent='Could not load the comparison: '+error.message;}})();
