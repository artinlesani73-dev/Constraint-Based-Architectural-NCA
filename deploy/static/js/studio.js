'use strict';
const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
let scene = null, draft = null, result = null, presets = [], saved = [];
let dirty = false, busy = false, view = 'iso';
let comparison = null, watchedJob = null;
const jobStates = new Map();
const families = ['access','coverage','facade','ground','legality','sparsity','spill','support','thickness'];
const canvas = $('canvas'), ctx = canvas.getContext('2d');
function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
}
function status(text, error = false) { $('status').textContent = text; $('status').className = error ? 'error' : ''; }
async function api(path, body) {
    const response = await fetch('/api/studio/' + path, body === undefined ? {} : {
        method:'POST', headers:{'Content-Type':'application/json'}, body:typeof body==='string'?body:JSON.stringify(body)});
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail));
    return value;
}
function lock(value) {
    busy = value;
    document.querySelectorAll('button,select,input').forEach(node => node.disabled = value);
    $('construct').disabled = value || dirty || !scene;
    $('export').disabled = value || dirty || !result;
    $('compare').disabled = value || saved.filter(r=>r.kind==='result').length < 2;
}
function markDirty() {
    dirty = true; result = null; lock(false); evidence(); draw();
    $('study-name').textContent = 'UNAPPLIED EDITS';
    $('drawing-note').textContent = 'Last applied scene · edited coordinates are pending validation';
    status('Edits are pending. Apply to validate, then save the scene or build an alternative.');
}
function editor() {
    $('editor').replaceChildren();
    for (const [group, items] of [['buildings',draft.buildings],['entrances',draft.entrances]]) {
        items.forEach((item,index) => {
            const detail = el('details'), title = el('summary',item.id.replaceAll('_',' '));
            detail.append(title);
            for (const axis of ['x','y','z']) {
                const row = el('label',undefined,'coordinate-row'); row.append(el('span',axis.toUpperCase()));
                const values = group === 'buildings' ? item[axis] : [item[axis]];
                values.forEach((value,bound) => {
                    const input = el('input'); input.type='number'; input.min='0'; input.max='32'; input.step='1'; input.value=value;
                    input.setAttribute('aria-label',`${item.id} ${axis} ${group === 'buildings' ? (bound ? 'end' : 'start') : 'position'}`);
                    input.addEventListener('input',() => {
                        const number = input.valueAsNumber;
                        if (group === 'buildings') {
                            draft[group][index][axis][bound] = number;
                            if (axis === 'x' && item.side) item.gap_facing_x = item.x[item.side === 'left' ? 1 : 0];
                        } else draft[group][index][axis] = number;
                        markDirty();
                    });
                    row.append(input);
                }); detail.append(row);
            }
            if (group === 'entrances') {
                const select = el('select'); select.setAttribute('aria-label',`${item.id} kind`);
                ['ground','facade'].forEach(kind => {const option=el('option',kind);option.value=kind;select.append(option);});
                select.value=item.kind; select.onchange=()=>{item.kind=select.value;markDirty();}; detail.append(select);
            }
            $('editor').append(detail);
        });
    }
}
function selectScene(value, record = null) {
    scene=clone(value); draft=clone(value); dirty=false; result=record;
    $('scene-name').textContent=scene.scene_id.replace(/^ref-\d+-/,'').replaceAll('-',' ');
    $('description').textContent=scene.description;
    $('study-name').textContent=record?.kind === 'result' ? 'SAVED PROCEDURAL STUDY' : 'SCENE PREVIEW';
    $('drawing-note').textContent=record?.kind === 'result' ? 'Procedural connection scaffold · not an inhabitable space' : 'Existing context · no generated material';
    $('presets').value=presets.some(s=>s.scene_id===scene.scene_id) ? scene.scene_id : '';
    editor(); evidence(); lock(busy); draw(); renderLibrary();
}
function stat(label,value,good) {
    const row=el('div',undefined,'stat'); row.append(el('span',label),el('b',value,good===undefined?'':good?'good':'bad')); $('summary').append(row);
}
function evidence() {
    const d=result?.diagnostics;
    $('summary').replaceChildren(); $('families').replaceChildren();
    $('result-title').textContent=d?'Read the result.':'A scene, before a solution.';
    $('method').textContent=d?'SCAFFOLD · NOT TRAINED NCA':'PREVIEW ONLY';
    $('evidence-note').textContent=d ? (d.joint_budget_connectivity ? 'Connectivity and material budget met. Review all nine proxy penalties below.' : 'This result has unmet checks. It is retained for comparison.') : 'Build an alternative to evaluate its geometry. A useful image is only the start.';
    stat('Material connectivity',d?(d.connectivity.all_connected?'Connected':'Disconnected'):'Not evaluated',d?.connectivity.all_connected);
    stat('Material / envelope',d?`${(d.material_ratio*100).toFixed(2)}%`:'—',d?.in_budget);
    stat('Budget target','3–12%');
    stat('Material voxels',d?d.material_voxels.toLocaleString():'—');
    stat('Routing context',d?(d.context_valid?'Compatible':'Incompatible'):'—',d?.context_valid);
    for (const name of families) { const row=el('div'); row.append(el('dt',name),el('dd',d?d.families[name].toFixed(5):'—'));$('families').append(row); }
    $('provenance').textContent=result ? `Saved locally · ${result.id}\nScene ${result.scene_hash.slice(0,16)}\n${d ? `Construction: ${result.construction_status}. ${result.elapsed_seconds.toFixed(2)} s. ` : ''}Full provenance is included in the export.` : 'Scene and result records stay on this computer. Export creates a portable JSON copy.';
}
async function refreshLibrary() {
    const data=await api('records'); saved=data.records; renderLibrary();
    $('integrity-alert').hidden=!data.integrity_issues.length;
    $('integrity-alert').textContent=`${data.integrity_issues.length} incomplete or damaged record(s) retained on disk: ${data.integrity_issues.join(', ')}. Inspect the local archive.`;
}
function renderLibrary() {
    $('records').replaceChildren(); $('record-count').textContent=`${saved.length} LOCAL RECORDS`;
    if(!saved.length) $('records').append(el('p','Every completed study is saved here automatically.','muted'));
    saved.forEach((record,index)=>{
        const card=el('button',undefined,'card'+(result?.id===record.id?' active':''));
        card.append(el('span',record.kind==='result'?'PROCEDURAL STUDY':record.kind.toUpperCase()),el('strong',record.scene.scene_id.replace(/^ref-\d+-/,'')),el('span',new Date(record.created_at).toLocaleString()),el('span',record.id.slice(-12)));
        card.disabled=busy;
        card.onclick=async()=>{
            if(dirty && !confirm('Discard unapplied edits and open this saved study?')) return;
            lock(true);
            try { const loaded=await api('records/'+record.id);selectScene(loaded.scene,loaded);status(loaded.kind==='failure'?'Retained failure: '+loaded.error:'Opened the saved scene and its original evidence.',loaded.kind==='failure'); }
            catch(error){status(error.message,true);} finally {lock(false);}
        }; $('records').append(card);
    });
    for (const id of ['compare-a','compare-b']) {
        const previous=$(id).value; $(id).replaceChildren();
        saved.filter(r=>r.kind==='result').forEach(r=>{
            const option=el('option',`${r.scene.scene_id.replace(/^ref-\d+-/,'')} · ${r.scene_hash.slice(0,7)} · ${r.id.slice(-6)}`);
            option.value=r.id; $(id).append(option);
        });
        if(saved.some(r=>r.id===previous&&r.kind==='result')) $(id).value=previous;
        else if(id==='compare-b'&&$(id).options.length>1) $(id).selectedIndex=1;
    }
    $('compare').disabled=busy||saved.filter(r=>r.kind==='result').length<2;
}
async function apply() {
    const record=await api('records?plan=false',{scene:draft}); selectScene(record.scene,record); await refreshLibrary(); status('Scene validated and saved locally. Build an alternative to evaluate it.');
}
$('apply').onclick=async()=>{lock(true);try{await apply();}catch(error){status(error.message,true);}finally{lock(false);}};
async function save(plan) {
    lock(true);status(plan?'Constructing and evaluating all nine families…':'Saving a scene snapshot…');
    try {
        if(dirty) await apply();
        if(plan) {
            const job=await api('jobs',{scene}); watchedJob=job.id;
            status(`Scaffold job ${job.id.slice(-12)} queued. You can keep editing or cancel it below.`);
            await refreshJobs(); return;
        }
        const record=await api('records?plan=false',{scene});
        selectScene(record.scene,record);await refreshLibrary();
        status(plan?`Saved alternative ${record.id.slice(-12)}. ${record.diagnostics.joint_budget_connectivity?'Connectivity and budget met.':'Some checks are unmet; the result is retained.'}`:'Scene snapshot saved locally.');
    }catch(error){try{await refreshLibrary();}catch{} status(error.message,true);}finally{lock(false);}
}
$('construct').onclick=()=>save(true); $('save-scene').onclick=()=>save(false);
$('presets').onchange=()=>{
    if(dirty&&!confirm('Discard unapplied edits and load another scene?')) {$('presets').value=scene.scene_id;return;}
    selectScene(presets.find(s=>s.scene_id===$('presets').value));status('Reference scene loaded. Edits create a new study; the reference file stays unchanged.');
};
$('export').onclick=async()=>{
    lock(true);
    try {
        const response=await fetch('/api/studio/records/'+result.id+'/export');
        if(!response.ok)throw new Error('Export failed; the saved record is still available locally.');
        // Preserve number encodings (for example 0.0) covered by the checksum.
        const portable=await response.text();
        const url=URL.createObjectURL(new Blob([portable],{type:'application/json'}));
        const link=el('a');link.href=url;link.download=`NCA-Studio-${result.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
        status('Portable record exported with its integrity checksum.');
    }catch(error){status(error.message,true);}finally{lock(false);}
};
$('import').onclick=()=>$('import-file').click();
$('import-file').onchange=async()=>{
    const file=$('import-file').files[0]; if(!file)return;
    if(dirty&&!confirm('Discard unapplied edits and open the imported record?')) {$('import-file').value='';return;}
    lock(true);status('Checking the imported record and its geometry…');
    try {
        if(file.size>2_000_000)throw new Error('Import exceeds the 2 MB limit.');
        const imported=await api('import',await file.text());
        selectScene(imported.record.scene,imported.record);await refreshLibrary();
        status(imported.duplicate?'Verified record already exists; opened its local copy.':'Imported a verified copy. Original provenance is retained in the record.');
    }catch(error){status('Import rejected: '+error.message,true);}finally{$('import-file').value='';lock(false);}
};

function stable(value) {
    if(Array.isArray(value))return '['+value.map(stable).join(',')+']';
    if(value&&typeof value==='object')return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+stable(value[k])).join(',')+'}';
    return JSON.stringify(value);
}
async function refreshJobs() {
    const data=await api('jobs');$('jobs').replaceChildren();
    $('job-alert').hidden=!data.integrity_issues.length;
    $('job-alert').textContent='Job history needs inspection: '+data.integrity_issues.join(', ');
    if(!data.jobs.length)$('jobs').append(el('p','No submitted jobs.','muted'));
    let libraryChanged=false;
    for(const job of data.jobs) {
        const row=el('div',undefined,'job'), text=el('div');
        text.append(el('strong',job.scene.scene_id.replace(/^ref-\d+-/,'')),el('small',`${job.id.slice(-12)} · ${job.stage}${job.parent_job?' · retry of '+job.parent_job.slice(-12):''}`));
        row.append(text,el('span',job.state,'job-state '+job.state));
        if(['queued','running'].includes(job.state)||['failed','cancelled','interrupted'].includes(job.state)) {
            const action=['queued','running'].includes(job.state)?'cancel':'retry';
            const button=el('button',action==='cancel'?'Cancel':'Retry');button.setAttribute('aria-label',`${action} job ${job.id.slice(-12)}`);button.disabled=busy;
            button.onclick=async()=>{
                button.disabled=true;
                try{const response=await api('jobs/'+job.id+'/'+action,{});if(action==='retry')watchedJob=response.id;await refreshJobs();status(action==='cancel'?`Job is ${response.state}.`:'Retry saved as a new linked job.');}
                catch(error){status(error.message,true);button.disabled=false;}
            };row.append(button);
        }
        $('jobs').append(row);
        if(jobStates.get(job.id)!==job.state&&job.state==='completed') {
            libraryChanged=true;
            if(job.id===watchedJob) {
                const record=await api('records/'+job.detail.result_id);
                if(!dirty&&!busy&&stable(scene)===stable(job.scene))selectScene(record.scene,record);
                status(`Scaffold ${job.id.slice(-12)} completed and saved. ${record.diagnostics.joint_budget_connectivity?'Connectivity and budget met.':'Unmet checks are retained.'}`);
                watchedJob=null;
            }
        }
        jobStates.set(job.id,job.state);
    }
    if(libraryChanged)await refreshLibrary();
}
async function pollJobs(){try{await refreshJobs();}catch(error){$('job-alert').hidden=false;$('job-alert').textContent='Job status unavailable. Reconnect before retrying a submission: '+error.message;}finally{setTimeout(pollJobs,1200);}}

$('compare').onclick=async()=>{
    lock(true);
    try {
        comparison=await api('compare?a='+encodeURIComponent($('compare-a').value)+'&b='+encodeURIComponent($('compare-b').value));
        $('compare-content').hidden=false;
        $('compare-note').textContent=comparison.same_scene?'Same scene revision. Compare the saved outcomes below.':'Different scene revisions: descriptive comparison, not a controlled model comparison.';
        for(const key of ['a','b'])$('caption-'+key).textContent=`${key.toUpperCase()} · ${comparison[key].scene.scene_id} · ${comparison[key].id.slice(-6)}`;
        $('geometry-delta').textContent=`B relative to A: ${comparison.added_voxels} added · ${comparison.removed_voxels} removed · ${comparison.shared_voxels} shared material voxels. Views and layers follow the main viewport controls.`;
        $('scene-diff').textContent=comparison.scene_changes.length?comparison.scene_changes.map(c=>`${c.group} / ${c.id}\nA: ${JSON.stringify(c.before)}\nB: ${JSON.stringify(c.after)}`).join('\n\n'):'No building or entrance changes.';
        const a=comparison.a.diagnostics,b=comparison.b.diagnostics;$('compare-metrics').replaceChildren();
        const rows=[['Material connectivity',a.connectivity.all_connected?'Connected':'Disconnected',b.connectivity.all_connected?'Connected':'Disconnected','—'],['Joint budget/connectivity',a.joint_budget_connectivity?'Met':'Unmet',b.joint_budget_connectivity?'Met':'Unmet','—'],['Material / envelope',`${(a.material_ratio*100).toFixed(2)}%`,`${(b.material_ratio*100).toFixed(2)}%`,`${((b.material_ratio-a.material_ratio)*100).toFixed(2)} pp`],['Envelope voxels',a.envelope_voxels,b.envelope_voxels,b.envelope_voxels-a.envelope_voxels],...families.map(k=>[k,a.families[k].toFixed(5),b.families[k].toFixed(5),(b.families[k]-a.families[k]).toFixed(5)])];
        for(const values of rows){const row=el('tr');values.forEach(v=>row.append(el('td',String(v))));$('compare-metrics').append(row);}
        draw();status('Comparison loaded. Proxy values do not establish an inhabitable space.');
    }catch(error){status(error.message,true);}finally{lock(false);}
};
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
    view=button.dataset.view;
    document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',b===button?'true':'false');});
    $('view-label').textContent=view==='iso'?'PARALLEL PROJECTION':view==='plan'?'LOOKING DOWN Z':'LOOKING ALONG Y';draw();
});
['buildings','material','entrances','guide'].forEach(id=>$(id).onchange=draw);

// Lightweight orthographic surface renderer. Geometry comes from saved voxel
// coordinates, never an illustrative substitute. One canvas, no remote assets.
function draw() {
    drawGeometry(canvas,scene,result,view);
    if(comparison)for(const key of ['a','b'])drawGeometry($('canvas-'+key),comparison[key].scene,comparison[key],view);
}
function drawGeometry(canvas,scene,result,view) {
    const ctx=canvas.getContext('2d');
    const rect=canvas.getBoundingClientRect(), w=rect.width,h=rect.height,dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    if(!scene)return;
    const scale=Math.min(w/(view==='iso'?62:42),h/(view==='iso'?57:43));
    const project=(x,y,z)=>view==='iso'?[w/2+(x-y)*.866*scale,h*.66+(x+y-32)*.5*scale-z*scale]:view==='plan'?[w/2+(x-16)*scale,h*.57+(y-16)*scale]:[w/2+(x-16)*scale,h*.87-z*scale];
    const line=(a,b,color)=>{ctx.beginPath();ctx.moveTo(...project(...a));ctx.lineTo(...project(...b));ctx.strokeStyle=color;ctx.lineWidth=.65;ctx.stroke();};
    for(let i=0;i<=32;i+=2){line([i,0,0],[i,32,0],'#d6ddd0');line([0,i,0],[32,i,0],'#d6ddd0');}
    const faces=[];
    const add=(x,y,z,dx,dy,dz,colors,alpha=1,mask=null)=>{
        const available=(ox,oy,oz)=>!mask?.has(`${z+oz},${y+oy},${x+ox}`);
        if(view!=='elevation'&&available(0,0,dz))faces.push({points:[[x,y,z+dz],[x+dx,y,z+dz],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[0],alpha,depth:x+y+z+dz});
        if(view!=='plan'&&available(0,dy,0))faces.push({points:[[x,y+dy,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x,y+dy,z+dz]],color:colors[1],alpha,depth:x+y+dy+z});
        if(view==='iso'&&available(dx,0,0))faces.push({points:[[x+dx,y,z],[x+dx,y+dy,z],[x+dx,y+dy,z+dz],[x+dx,y,z+dz]],color:colors[2],alpha,depth:x+dx+y+z});
    };
    if($('buildings').checked)scene.buildings.forEach(b=>add(b.x[0],b.y[0],b.z[0],b.x[1]-b.x[0],b.y[1]-b.y[0],b.z[1]-b.z[0],['#d9ddcf','#b7c0b1','#c4cbbb'],.68));
    if($('material').checked&&result?.material_zyx){const mask=new Set(result.material_zyx.map(c=>c.join(',')));result.material_zyx.forEach(([z,y,x])=>add(x,y,z,1,1,1,['#5f9c83','#275b4d','#367361'],1,mask));}
    if($('entrances').checked)scene.entrances.forEach(e=>add(e.x,e.y,e.z,e.extent,e.extent,e.extent,['#dbb476','#ac7546','#c49155']));
    if($('guide').checked&&result?.guide_zyx)result.guide_zyx.forEach(([z,y,x])=>add(x+.3,y+.3,z+.3,.4,.4,.4,['#f4d8a0','#c59c51','#e2bc72']));
    faces.sort((a,b)=>view==='plan'?a.points[0][2]-b.points[0][2]:view==='elevation'?a.points[0][1]-b.points[0][1]:a.depth-b.depth);
    for(const face of faces){ctx.beginPath();face.points.forEach((point,i)=>{const p=project(...point);i?ctx.lineTo(...p):ctx.moveTo(...p);});ctx.closePath();ctx.globalAlpha=face.alpha;ctx.fillStyle=face.color;ctx.fill();ctx.strokeStyle=face.color;ctx.lineWidth=.45;ctx.stroke();}
    ctx.globalAlpha=1;ctx.font='9px Segoe UI';ctx.fillStyle='#778675';
    for(const [label,p]of [['X',[33,0,0]],['Y',[0,33,0]]]){const q=project(...p);ctx.fillText(label,...q);}
}
new ResizeObserver(draw).observe(canvas.parentElement);
(async()=>{try{const data=await api('scenes');presets=data.scenes;presets.forEach(s=>{const option=el('option',s.scene_id.replace(/^ref-/,'').replaceAll('-',' '));option.value=s.scene_id;$('presets').append(option);});selectScene(presets.find(s=>s.scene_id.includes('elevated'))||presets[0]);await refreshLibrary();status('Workspace ready. This planner builds connection scaffolds, not inhabitable spaces.');}catch(error){status('Workspace could not load: '+error.message,true);}finally{pollJobs();}})();
