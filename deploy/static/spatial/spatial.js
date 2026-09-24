'use strict';
const $=id=>document.getElementById(id);
let study,selected,view='iso';
const families=['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'];
const names={'aligned':'01 · Platform and shared landing','narrow-deck':'02 · Too narrow','missing-floor':'03 · Missing floor','low-headroom':'04 · Not enough headroom','blocked-span':'05 · Blocked span','split-levels':'06 · Different approach levels'};
function row(parent,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td);}parent.append(tr);}
function refresh(){
    selected=study.cases.find(c=>c.case===$('case').value);
    const a=selected.baseline.spatial,b=selected.spatial,al=selected.baseline.legacy,bl=selected.legacy;
    $('description').textContent=selected.scene.description;
    for(const [id,r]of [['baseline-gate',a],['candidate-gate',b]]){$(id).textContent=r.spatial_gate?'Spatial brief met':'Spatial brief unmet';$(id).className='badge '+(r.spatial_gate?'pass':'fail');}
    $('candidate-caption').textContent=selected.construction.status==='unsupported_layout'?'No deck generated: this method does not connect different approach levels.':b.spatial_gate?'The green floor carries a clear passage and a wider landing. Blue represents empty volume.':'This candidate is retained with its failed checks. It is not an accepted design.';
    $('verdict').textContent=b.spatial_gate?'A usable surface, under this brief.':!b.layout_supported?'This layout needs another method.':'Connected material is not enough.';
    const reasons=[];
    if(!b.layout_supported)reasons.push(b.layout_reason);
    else{
        if(!b.approach_connected)reasons.push('No continuous route for the required 2.4 m square footprint.');
        if(b.floor_cells_with_insufficient_clearance)reasons.push(`${b.floor_cells_with_insufficient_clearance} floor cells have less than 2.4 m clearance.`);
        if(!b.landing_accessible)reasons.push('The full landing is not clear and reachable.');
        if(b.material_context_collision_voxels)reasons.push(`${b.material_context_collision_voxels} proposed cells intersect existing context.`);
    }
    $('explanation').textContent=reasons.join(' ')||'The floor, required clearance, approach connection and landing satisfy the declared geometric checks. The old occupied-entrance objective still asks a different question.';
    $('facts').replaceChildren();
    for(const [value,label]of [[b.floor_area_m2.toFixed(2)+' m²','Proposed floor footprint'],[b.clear_surface_area_m2.toFixed(2)+' m²','Floor with clear height'],[String(b.material_voxels),'Material voxels']]){const d=document.createElement('div'),strong=document.createElement('b'),s=document.createElement('span');strong.textContent=value;s.textContent=label;d.append(strong,s);$('facts').append(d);}
    $('checks').replaceChildren();const yes=v=>v?'Yes':'No';
    for(const values of [
        ['Clear, width-qualified approach route',yes(a.approach_connected),yes(b.approach_connected)],
        ['Full landing clear and reachable',yes(a.landing_accessible),yes(b.landing_accessible)],
        ['Spatial brief met',yes(a.spatial_gate),yes(b.spatial_gate)],
        ['Old material-access connected',yes(al.connectivity.all_connected),yes(bl.connectivity.all_connected)],
        ['Old 3–12% material budget',yes(al.in_budget),yes(bl.in_budget)],
        ['Material / old fixed envelope',(al.material_ratio*100).toFixed(2)+'%',(bl.material_ratio*100).toFixed(2)+'%'],
        ['Old joint budget + connectivity',yes(al.joint_budget_connectivity),yes(bl.joint_budget_connectivity)]])row($('checks'),values);
    $('families').replaceChildren();for(const f of families)row($('families'),[f,al.families[f].toFixed(5),bl.families[f].toFixed(5)]);
    $('provenance').textContent=`Saved run ${study.run_id} · ${study.version}. Six hand-designed development examples; no learned model. Scene ${selected.scene_hash.slice(0,12)}.`;
    draw();
}
function draw(){if(!selected)return;render($('baseline'),selected.baseline);render($('candidate'),selected);}
function render(canvas,result){
    const ctx=canvas.getContext('2d'),rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const scale=Math.min(w/(view==='iso'?53:37),h/(view==='iso'?58:38));
    const point=(x,y,z)=>view==='iso'?[w*.5+(x-y)*.866*scale,h*.72+(x+y-32)*.5*scale-z*scale]:view==='plan'?[w*.5+(x-16)*scale,h*.5+(y-16)*scale]:[w*.5+(x-16)*scale,h*.87-z*scale];
    const faces=[];
    function box(x,y,z,dx,dy,dz,colors,alpha=1,mask=null){
        if(view==='section'&&!(y<=15&&15<y+dy))return;
        const available=(ox,oy,oz)=>!mask?.has(`${z+oz},${y+oy},${x+ox}`);
        if(view!=='section'&&available(0,0,dz))faces.push({points:[[x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[0],alpha,depth:x+y+z+dz});
        if(view!=='plan'&&(view==='section'||available(0,dy,0)))faces.push({points:[[x,y+dy,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[1],alpha,depth:x+y+dy+z});
        if(view==='iso'&&available(dx,0,0))faces.push({points:[[x+dx,y,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x+dx,y,z+dz]],color:colors[2],alpha,depth:x+dx+y+z});
    }
    function cells(coords,colors,alpha){const mask=new Set(coords.map(c=>c.join(',')));for(const[z,y,x]of coords)box(x,y,z,1,1,1,colors,alpha,mask);}
    function line(a,b,color){ctx.beginPath();ctx.moveTo(...point(...a));ctx.lineTo(...point(...b));ctx.strokeStyle=color;ctx.lineWidth=.6;ctx.stroke();}
    if(view==='section'){for(let z=0;z<=28;z+=2)line([0,15,z],[32,15,z],'#e2e4da');}
    else for(let i=0;i<=32;i+=2){line([i,0,0],[i,32,0],'#dee2d6');line([0,i,0],[32,i,0],'#dee2d6');}
    if($('context').checked)for(const b of selected.scene.buildings)box(b.x[0],b.y[0],b.z[0],b.x[1]-b.x[0],b.y[1]-b.y[0],b.z[1]-b.z[0],['#ccd1c1','#b0baaa','#c1c8b8'],.42);
    cells(result.material_zyx,['#6c9d79','#285e4f','#3e7962'],1);
    for(const[z,y,x]of result.surface_zyx||[])box(x,y,z,1,1,.035,['#add7a6','#add7a6','#add7a6'],.9);
    if($('air').checked)cells(result.clearance_zyx||[],['#a2d4e4','#67b0cb','#8fc8dd'],.24);
    for(const e of selected.scene.entrances)box(e.x,e.y,e.z,e.extent,e.extent,e.extent,['#ddb375','#b78448','#c99858'],.3);
    faces.sort((a,b)=>view==='plan'?a.points[0][2]-b.points[0][2]:view==='section'?a.points[0][1]-b.points[0][1]:a.depth-b.depth);
    for(const f of faces){ctx.beginPath();f.points.forEach((p,i)=>i?ctx.lineTo(...point(...p)):ctx.moveTo(...point(...p)));ctx.closePath();ctx.globalAlpha=f.alpha;ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle=f.color;ctx.lineWidth=.35;ctx.stroke();}ctx.globalAlpha=1;
    ctx.font='11px Segoe UI';ctx.fillStyle='#697a6d';
    if(view==='section'){
        for(const[z,label]of [[8,'floor +6.4 m'],[11,'clearance +8.8 m']]){const p=point(10,15,z);ctx.fillText(label,p[0],p[1]-6);}
    }else {const p=point(28,28,0);ctx.fillText('0.8 m / cell',p[0]-50,p[1]+18);}
}
$('case').onchange=refresh;['air','context'].forEach(id=>$(id).onchange=draw);
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('view-note').textContent=view==='section'?'Section through Y = 15 cells (12 m) · Z is up':view==='plan'?'Plan · looking down Z':'Parallel projection';draw();});
new ResizeObserver(draw).observe($('candidate').parentElement);
(async()=>{try{const response=await fetch('study.json');if(!response.ok)throw new Error('Saved study unavailable');study=await response.json();for(const c of study.cases){const option=document.createElement('option');option.value=c.case;option.textContent=names[c.case]||c.case;$('case').append(option);}refresh();}catch(error){$('description').textContent='Could not load the saved study: '+error.message;}})();
