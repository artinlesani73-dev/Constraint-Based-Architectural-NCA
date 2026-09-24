'use strict';
const $=id=>document.getElementById(id);
let study,view='iso',a,b;
const names={empty:'Empty field',w1_scaffold:'W1 connection scaffold',sp1_platform:'SP1 flat platform',slab_512:'Flat slab · 512 cells',solid_512:'Solid block · 512 cells',open_ends_512:'Hollow form · 512 cells',side_aperture_487:'Open volumetric form',closed_shell_610:'Closed hollow shell',extent_decoy_8:'Eight scattered corners'};
const descriptions={empty:'A control: empty site area must not count as form-created space.',w1_scaffold:'Historical connection control. Generated procedurally, not by trained NCA.',sp1_platform:'Historical platform diagnostic. Its brief is superseded as the design target.',slab_512:'Same material count as the block and hollow form, arranged in a shallow slab.',solid_512:'Three-dimensional extent, with its interior completely filled.',open_ends_512:'Thin material around a void; its open ends meet the solid context buildings.',side_aperture_487:'A side aperture connects the internal void to the surrounding free space.',closed_shell_610:'A topologically sealed cavity. Enclosure is a probe, not a required design goal.',extent_decoy_8:'Same outer extents as the hollow form, with eight disconnected cells.'};
function row(parent,values){const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});parent.append(tr);}
function refresh(){
    a=study.cases.find(c=>c.case===$('a').value);b=study.cases.find(c=>c.case===$('b').value);
    for(const[key,c]of [['a',a],['b',b]]){$('title-'+key).textContent=names[c.case];$('caption-'+key).textContent=descriptions[c.case];}
    $('status').textContent=a.volume.material_voxels===b.volume.material_voxels?`Equal material count: ${a.volume.material_voxels} cells each. Inspect how their arrangement changes the void.`:'Different material counts. Inspect both amount and arrangement; this is not an equal-mass comparison.';
    $('metrics').replaceChildren();const v=[a.volume,b.volume];
    const measures=[['Material cells',r=>r.material_voxels],['Material volume (m³)',r=>r.material_volume_m3.toFixed(2)],['Extents X × Y × Z (m)',r=>[...r.extent_m_zyx].reverse().map(n=>n.toFixed(1)).join(' × ')],['Material components',r=>r.material_components_6],['2-axis bracketed empty cells',r=>r.bracketed_2_axes_voxels],['3-axis bracketed empty cells',r=>r.bracketed_3_axes_voxels],['Sealed by proposed material',r=>r.sealed_by_form_voxels],['Sealed including context',r=>r.sealed_with_context_voxels],['Bracketed cells open to exterior',r=>r.bracketed_exterior_connected_voxels],['Centers fitting 2.4 m free cube',r=>r.free_cube_centers_in_bracketed_void['3']],['Centers fitting 4.0 m free cube',r=>r.free_cube_centers_in_bracketed_void['5']]];
    for(const[name,f]of measures)row($('metrics'),[name,...v.map(f)]);
    $('legacy').replaceChildren();const l=[a.legacy,b.legacy];
    row($('legacy'),['Material / fixed envelope',...l.map(r=>(100*r.material_ratio).toFixed(2)+'%')]);
    row($('legacy'),['Original 3–12% budget',...l.map(r=>r.in_budget?'Met':'Unmet')]);
    row($('legacy'),['Material-access connectivity',...l.map(r=>r.connectivity.all_connected?'Connected':'Disconnected')]);
    for(const f of ['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'])row($('legacy'),[f,...l.map(r=>r.families[f].toFixed(5))]);
    $('provenance').textContent=`Saved audit ${study.run_id} · ${study.version}. Nine analytical fields, one context. No new training or generalized design-quality claim.`;draw();
}
function draw(){if(!a)return;render($('canvas-a'),a);render($('canvas-b'),b);$('slice-control').hidden=view==='iso';$('cut').disabled=view!=='iso';$('slice-value').textContent=$('slice').value;const s=Number($('slice').value);$('view-note').textContent=view==='iso'?($('cut').checked?'Cutaway: foreground Y ≥ 16 hidden. Metrics use the full field.':'Full proposed form. Metrics use the full field.'):`${view==='xz'?'Vertical slice at Y':'Horizontal slice at Z'} = ${s} cells · cell-center coordinate ${((s+.5)*.8).toFixed(1)} m`;
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
    function cells(coords,colors,alpha){const shown=view==='iso'&&$('cut').checked?coords.filter(c=>c[1]<16):coords;const mask=new Set(shown.map(c=>c.join(',')));for(const[z,y,x]of shown)box(x,y,z,1,1,1,colors,alpha,mask);}
    function line(a,b){ctx.beginPath();ctx.moveTo(...p(...a));ctx.lineTo(...p(...b));ctx.strokeStyle='#dce2d5';ctx.lineWidth=.5;ctx.stroke();}
    if(view==='xz'){for(let z=0;z<=32;z+=2)line([0,slice,z],[32,slice,z]);}
    else for(let i=0;i<=32;i+=2){line([i,0,0],[i,32,0]);line([0,i,0],[32,i,0]);}
    if($('context').checked)for(const c of study.scene.buildings)box(c.x[0],c.y[0],c.z[0],c.x[1]-c.x[0],c.y[1]-c.y[0],c.z[1]-c.z[0],['#cbd3c2','#b1bdaa','#c1cbb8'],.35);
    cells(result.material_zyx,['#72a07e','#285e4f','#438369'],1);
    if($('void').checked)cells(result.void_zyx,['#a1d7e8','#69b4d0','#8ecbdd'],.37);
    faces.sort((a,b)=>view==='xy'?a.pts[0][2]-b.pts[0][2]:view==='xz'?a.pts[0][1]-b.pts[0][1]:a.d-b.d);
    for(const f of faces){ctx.beginPath();f.pts.forEach((q,i)=>i?ctx.lineTo(...p(...q)):ctx.moveTo(...p(...q)));ctx.closePath();ctx.globalAlpha=f.alpha;ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle=f.color;ctx.lineWidth=.4;ctx.stroke();}ctx.globalAlpha=1;
    ctx.font='11px Segoe UI';ctx.fillStyle='#6a7e6e';ctx.fillText('0.8 m / cell',18,h-17);
}
['a','b'].forEach(id=>$(id).onchange=refresh);['cut','void','context'].forEach(id=>$(id).onchange=draw);$('slice').oninput=draw;
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));draw();});
new ResizeObserver(draw).observe($('canvas-b').parentElement);
(async()=>{try{const response=await fetch('study.json');if(!response.ok)throw new Error('Saved audit unavailable');study=await response.json();for(const id of ['a','b'])for(const c of study.cases){const option=document.createElement('option');option.value=c.case;option.textContent=names[c.case];$(id).append(option);}$('a').value='sp1_platform';$('b').value='side_aperture_487';refresh();}catch(error){$('status').textContent='Could not load the audit: '+error.message;}})();
