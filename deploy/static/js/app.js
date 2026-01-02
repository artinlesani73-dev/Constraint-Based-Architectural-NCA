// Three.js Setup
let scene, camera, renderer, controls, transformControls;
const viewer = document.getElementById('viewer');
const grid_size = 32;
let abortController = null;
let isGenerating = false;
let helperGroup = null;
let helperMeshes = [];
let selectedHelper = null;
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    camera = new THREE.PerspectiveCamera(75, viewer.clientWidth / viewer.clientHeight, 0.1, 1000);
    camera.position.set(40, 40, 40);
    camera.lookAt(16, 16, 16);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    viewer.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(16, 0, 16);
    controls.update();

    transformControls = new THREE.TransformControls(camera, renderer.domElement);
    transformControls.addEventListener('dragging-changed', (event) => {
        controls.enabled = !event.value;
    });
    transformControls.addEventListener('mouseUp', () => {
        if (selectedHelper) {
            applyHelperTransform(selectedHelper);
        }
    });
    transformControls.addEventListener('objectChange', () => {
        if (selectedHelper) {
            snapHelperMesh(selectedHelper);
        }
    });
    scene.add(transformControls);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(50, 100, 50);
    scene.add(directionalLight);

    // Grid helper for ground
    const gridHelper = new THREE.GridHelper(32, 32, 0x334155, 0x1e293b);
    gridHelper.position.set(16, 0, 16);
    scene.add(gridHelper);

    helperGroup = new THREE.Group();
    scene.add(helperGroup);

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Voxel Rendering
let voxelMeshes = [];
let lastRender = null;
const styleSettings = {
    structure: { color: '#6366f1', opacity: 1.0 },
    existing: { color: '#64748b', opacity: 0.8 },
    access: { color: '#ef4444', opacity: 0.9 },
    anchors: { color: '#22c55e', opacity: 0.2 },
};
const visibilitySettings = {
    structure: true,
    existing: true,
    access: true,
    anchors: true,
};

function clearVoxels() {
    voxelMeshes.forEach(m => scene.remove(m));
    voxelMeshes = [];
}

function renderVoxels(structure, existing, access, anchors) {
    clearVoxels();
    const boxGeo = new THREE.BoxGeometry(0.95, 0.95, 0.95);

    // Materials
    const existingMat = new THREE.MeshPhongMaterial({
        color: styleSettings.existing.color,
        transparent: styleSettings.existing.opacity < 1,
        opacity: styleSettings.existing.opacity
    });
    const structuralMat = new THREE.MeshPhongMaterial({
        color: styleSettings.structure.color,
        transparent: styleSettings.structure.opacity < 1,
        opacity: styleSettings.structure.opacity
    });
    const accessMat = new THREE.MeshPhongMaterial({
        color: styleSettings.access.color,
        transparent: styleSettings.access.opacity < 1,
        opacity: styleSettings.access.opacity
    });
    const anchorMat = new THREE.MeshPhongMaterial({
        color: styleSettings.anchors.color,
        transparent: styleSettings.anchors.opacity < 1,
        opacity: styleSettings.anchors.opacity
    });

    for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_size; y++) {
            for (let x = 0; x < grid_size; x++) {
                // Access Points (Red) - Priority Render
                if (visibilitySettings.access && access && access[z][y][x] > 0.5) {
                    const mesh = new THREE.Mesh(boxGeo, accessMat);
                    mesh.position.set(x + 0.5, z + 0.5, y + 0.5);
                    scene.add(mesh);
                    voxelMeshes.push(mesh);
                }
                // Existing
                else if (visibilitySettings.existing && existing && existing[z][y][x] > 0.5) {
                    const mesh = new THREE.Mesh(boxGeo, existingMat);
                    mesh.position.set(x + 0.5, z + 0.5, y + 0.5);
                    scene.add(mesh);
                    voxelMeshes.push(mesh);
                }
                // Generated Structure
                else if (visibilitySettings.structure && structure && structure[z][y][x] > 0.5) {
                    const mesh = new THREE.Mesh(boxGeo, structuralMat);
                    mesh.position.set(x + 0.5, z + 0.5, y + 0.5);
                    scene.add(mesh);
                    voxelMeshes.push(mesh);
                }
                // Anchor Zones (Subtle Green)
                else if (visibilitySettings.anchors && anchors && anchors[z][y][x] > 0.5) {
                    const mesh = new THREE.Mesh(boxGeo, anchorMat);
                    mesh.position.set(x + 0.5, z + 0.5, y + 0.5);
                    scene.add(mesh);
                    voxelMeshes.push(mesh);
                }
            }
        }
    }
    document.getElementById('voxel-count').textContent = `Voxels: ${voxelMeshes.length}`;
    lastRender = { structure, existing, access, anchors };
}

function clearHelpers() {
    if (!helperGroup) return;
    helperGroup.clear();
    helperMeshes = [];
}

function addBuildingHelper(b, index) {
    const dx = b.x[1] - b.x[0];
    const dy = b.y[1] - b.y[0];
    const dz = b.z[1] - b.z[0];
    const geometry = new THREE.BoxGeometry(dx, dz, dy);
    const edges = new THREE.EdgesGeometry(geometry);
    const material = new THREE.LineBasicMaterial({
        color: 0x93c5fd,
        transparent: true,
        opacity: 0.7
    });
    const line = new THREE.LineSegments(edges, material);
    line.position.set(b.x[0] + dx / 2, b.z[0] + dz / 2, b.y[0] + dy / 2);
    line.userData = { type: 'building', index, baseSize: { dx, dy, dz } };
    helperGroup.add(line);
    helperMeshes.push(line);
}

function addAccessHelper(ap, index) {
    const geometry = new THREE.BoxGeometry(2, 2, 2);
    const material = new THREE.MeshBasicMaterial({
        color: styleSettings.access.color,
        transparent: true,
        opacity: styleSettings.access.opacity
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(ap.x + 1, ap.z + 1, ap.y + 1);
    mesh.userData = { type: 'access', index, baseSize: { dx: 2, dy: 2, dz: 2 } };
    helperGroup.add(mesh);
    helperMeshes.push(mesh);
}

function rebuildHelpers() {
    clearHelpers();
    buildings.forEach((b, i) => addBuildingHelper(b, i));
    accessPoints.forEach((ap, i) => addAccessHelper(ap, i));
}

function onPointerDown(event) {
    if (isGenerating) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(helperMeshes, false);
    if (intersects.length > 0) {
        selectedHelper = intersects[0].object;
        transformControls.attach(selectedHelper);
    }
}

function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
}

function snap(val) {
    return Math.round(val);
}

function snapHalf(val) {
    return Math.round(val * 2) / 2;
}

function snapHelperMesh(mesh) {
    if (!mesh || !mesh.userData) return;
    const type = mesh.userData.type;
    const mode = transformControls ? transformControls.getMode() : 'translate';

    if (type === 'access' && mode === 'translate') {
        const x = clamp(snap(mesh.position.x - 1), 0, grid_size - 2);
        const y = clamp(snap(mesh.position.z - 1), 0, grid_size - 2);
        const z = clamp(snap(mesh.position.y - 1), 0, grid_size - 2);
        mesh.position.set(x + 1, z + 1, y + 1);
        return;
    }

    if (type === 'building') {
        const base = mesh.userData.baseSize;
        if (mode === 'translate') {
            const cx = snapHalf(mesh.position.x);
            const cy = snapHalf(mesh.position.z);
            mesh.position.set(cx, mesh.position.y, cy);
        }
        if (mode === 'scale') {
            const scaled = new THREE.Vector3(
                base.dx * mesh.scale.x,
                base.dz * mesh.scale.y,
                base.dy * mesh.scale.z
            );
            const dx = clamp(snap(scaled.x), 2, grid_size);
            const dz = clamp(snap(scaled.y), 2, grid_size);
            const dy = clamp(snap(scaled.z), 2, grid_size);
            mesh.scale.set(dx / base.dx, dz / base.dz, dy / base.dy);
        }
    }
}

function applyHelperTransform(mesh) {
    if (!mesh || !mesh.userData) return;
    const type = mesh.userData.type;
    const index = mesh.userData.index;

    if (type === 'access') {
        const x = clamp(snap(mesh.position.x - 1), 0, grid_size - 2);
        const y = clamp(snap(mesh.position.z - 1), 0, grid_size - 2);
        const z = clamp(snap(mesh.position.y - 1), 0, grid_size - 2);
        accessPoints[index] = { ...accessPoints[index], x, y, z };
        mesh.position.set(x + 1, z + 1, y + 1);
        updateLists();
        return;
    }

    if (type === 'building') {
        const base = mesh.userData.baseSize;
        const scaled = new THREE.Vector3(
            base.dx * mesh.scale.x,
            base.dz * mesh.scale.y,
            base.dy * mesh.scale.z
        );
        const dx = clamp(snap(scaled.x), 2, grid_size);
        const dz = clamp(snap(scaled.y), 2, grid_size);
        const dy = clamp(snap(scaled.z), 2, grid_size);

        const centerX = mesh.position.x;
        const centerY = mesh.position.z;

        let x0 = clamp(snap(centerX - dx / 2), 0, grid_size - 1);
        let y0 = clamp(snap(centerY - dy / 2), 0, grid_size - 1);
        let x1 = clamp(x0 + dx, 1, grid_size);
        let y1 = clamp(y0 + dy, 1, grid_size);

        if (x1 - x0 < 2) x1 = clamp(x0 + 2, 2, grid_size);
        if (y1 - y0 < 2) y1 = clamp(y0 + 2, 2, grid_size);

        const z0 = 0;
        const z1 = clamp(dz, 1, grid_size);

        const side = (x0 + x1) / 2 < grid_size / 2 ? 'left' : 'right';
        const gap_facing_x = side === 'left' ? x1 : x0;

        buildings[index] = {
            ...buildings[index],
            x: [x0, x1],
            y: [y0, y1],
            z: [z0, z1],
            side,
            gap_facing_x
        };
        updateLists();
    }
}

function updateHelperStyles() {
    helperMeshes.forEach((mesh) => {
        if (!mesh.userData) return;
        if (mesh.userData.type === 'access' && mesh.material) {
            mesh.material.color.set(styleSettings.access.color);
            mesh.material.opacity = styleSettings.access.opacity;
            mesh.material.transparent = styleSettings.access.opacity < 1;
        }
    });
}

function bindStyleControls() {
    const pairs = [
        ['structure', 'color-structure', 'opacity-structure', 'opacity-structure-val'],
        ['existing', 'color-existing', 'opacity-existing', 'opacity-existing-val'],
        ['access', 'color-access', 'opacity-access', 'opacity-access-val'],
        ['anchors', 'color-anchors', 'opacity-anchors', 'opacity-anchors-val'],
    ];

    pairs.forEach(([key, colorId, opacityId, opacityValId]) => {
        const colorInput = document.getElementById(colorId);
        const opacityInput = document.getElementById(opacityId);
        const opacityVal = document.getElementById(opacityValId);

        colorInput.addEventListener('input', () => {
            styleSettings[key].color = colorInput.value;
            updateHelperStyles();
            if (lastRender) {
                renderVoxels(lastRender.structure, lastRender.existing, lastRender.access, lastRender.anchors);
            } else {
                updatePreview();
            }
        });

        opacityInput.addEventListener('input', () => {
            styleSettings[key].opacity = parseFloat(opacityInput.value);
            opacityVal.textContent = opacityInput.value;
            updateHelperStyles();
            if (lastRender) {
                renderVoxels(lastRender.structure, lastRender.existing, lastRender.access, lastRender.anchors);
            } else {
                updatePreview();
            }
        });
    });
}

// UI Management
let defaultBuildings = [
    { x: [1, 9], y: [0, 27], z: [0, 14], gap_facing_x: 9, side: "left" },
    { x: [23, 32], y: [0, 19], z: [0, 14], gap_facing_x: 23, side: "right" }
];

let defaultAccess = [
    { x: 12, y: 23, z: 0, type: "ground" },
    { x: 9, y: 1, z: 11, type: "elevated" }
];

let buildings = JSON.parse(JSON.stringify(defaultBuildings));
let accessPoints = JSON.parse(JSON.stringify(defaultAccess));

function setInputLock(locked) {
    isGenerating = locked;
    const inputs = document.querySelectorAll('input, button:not(#stop-btn)');
    inputs.forEach(el => {
        if (el.id !== 'stop-btn') el.disabled = locked;
    });
    document.getElementById('stop-btn').disabled = !locked;
    if (locked) {
        transformControls.detach();
        selectedHelper = null;
    }
}

async function updatePreview() {
    if (isGenerating) return;
    try {
        const response = await fetch('/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ buildings, access_points: accessPoints })
        });
        const data = await response.json();
        renderVoxels(null, data.existing, data.access, data.anchors);
    } catch (err) {
        console.error("Preview error:", err);
    }
}

function updateLists(skipPreview = false) {
    const bList = document.getElementById('buildings-list');
    bList.innerHTML = '';
    buildings.forEach((b, i) => {
        const div = document.createElement('div');
        div.className = 'building-item';
        const width = b.x[1] - b.x[0];
        const depth = b.y[1] - b.y[0];
        const height = b.z[1] - b.z[0];
        div.innerHTML = `
            <button class="delete" onclick="removeBuilding(${i})">×</button>
            <div class="item-header">
                <span class="item-title">Building ${i + 1}</span>
            </div>
            <div class="size-controls">
                <div class="size-row">
                    <label>X Pos</label>
                    <input type="range" min="0" max="${grid_size - 2}" value="${b.x[0]}" data-building="${i}" data-prop="x0">
                    <span>${b.x[0]}</span>
                </div>
                <div class="size-row">
                    <label>Width</label>
                    <input type="range" min="2" max="${grid_size}" value="${width}" data-building="${i}" data-prop="width">
                    <span>${width}</span>
                </div>
                <div class="size-row">
                    <label>Y Pos</label>
                    <input type="range" min="0" max="${grid_size - 2}" value="${b.y[0]}" data-building="${i}" data-prop="y0">
                    <span>${b.y[0]}</span>
                </div>
                <div class="size-row">
                    <label>Depth</label>
                    <input type="range" min="2" max="${grid_size}" value="${depth}" data-building="${i}" data-prop="depth">
                    <span>${depth}</span>
                </div>
                <div class="size-row">
                    <label>Height</label>
                    <input type="range" min="1" max="${grid_size}" value="${height}" data-building="${i}" data-prop="height">
                    <span>${height}</span>
                </div>
            </div>
        `;
        bList.appendChild(div);

        // Bind slider events
        div.querySelectorAll('input[type="range"]').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const idx = parseInt(e.target.dataset.building);
                const prop = e.target.dataset.prop;
                const val = parseInt(e.target.value);
                e.target.nextElementSibling.textContent = val;
                updateBuildingProperty(idx, prop, val);
            });
        });
    });

    const aList = document.getElementById('access-list');
    aList.innerHTML = '';
    accessPoints.forEach((ap, i) => {
        const div = document.createElement('div');
        div.className = 'access-item';
        div.innerHTML = `
            <button class="delete" onclick="removeAccess(${i})">×</button>
            <div class="item-fields">
                <div>Pos: ${ap.x}, ${ap.y}, ${ap.z}</div>
                <div>Type: ${ap.type}</div>
            </div>
        `;
        aList.appendChild(div);
    });
    rebuildHelpers();
    if (!skipPreview) {
        updatePreview();
    }
}

function updateBuildingProperty(index, prop, value) {
    const b = buildings[index];
    if (!b) return;

    switch (prop) {
        case 'x0':
            const oldWidth = b.x[1] - b.x[0];
            b.x[0] = Math.min(value, grid_size - 2);
            b.x[1] = Math.min(b.x[0] + oldWidth, grid_size);
            break;
        case 'width':
            b.x[1] = Math.min(b.x[0] + value, grid_size);
            break;
        case 'y0':
            const oldDepth = b.y[1] - b.y[0];
            b.y[0] = Math.min(value, grid_size - 2);
            b.y[1] = Math.min(b.y[0] + oldDepth, grid_size);
            break;
        case 'depth':
            b.y[1] = Math.min(b.y[0] + value, grid_size);
            break;
        case 'height':
            b.z[1] = Math.min(value, grid_size);
            break;
    }

    // Update side and gap_facing_x
    const centerX = (b.x[0] + b.x[1]) / 2;
    b.side = centerX < grid_size / 2 ? 'left' : 'right';
    b.gap_facing_x = b.side === 'left' ? b.x[1] : b.x[0];

    rebuildHelpers();
    updatePreview();
}

window.removeBuilding = (i) => { buildings.splice(i, 1); updateLists(); };
window.removeAccess = (i) => { accessPoints.splice(i, 1); updateLists(); };

document.getElementById('add-building').onclick = () => {
    buildings.push({ x: [0, 4], y: [0, 4], z: [0, 8] });
    updateLists();
};

document.getElementById('add-access').onclick = () => {
    accessPoints.push({ x: 16, y: 16, z: 0, type: "ground" });
    updateLists();
};

// Collapsible sections
window.toggleSection = function(header) {
    const section = header.parentElement;
    const icon = header.querySelector('.collapse-icon');
    section.classList.toggle('collapsed');
    icon.textContent = section.classList.contains('collapsed') ? '+' : '-';
};

// Slider bindings
const sliderBindings = [
    ['steps', 'steps-val'],
    ['threshold', 'threshold-val'],
    ['noise-std', 'noise-std-val'],
    ['fire-rate', 'fire-rate-val'],
    ['corridor-seed', 'corridor-seed-val'],
    ['corridor-width', 'corridor-width-val'],
    ['vertical-envelope', 'vertical-envelope-val'],
    ['update-scale', 'update-scale-val'],
];

sliderBindings.forEach(([sliderId, valId]) => {
    const slider = document.getElementById(sliderId);
    const valSpan = document.getElementById(valId);
    if (slider && valSpan) {
        slider.oninput = (e) => {
            valSpan.textContent = e.target.value;
        };
    }
});

document.getElementById('stop-btn').onclick = () => {
    if (abortController) {
        abortController.abort();
        console.log("Generation stopped.");
    }
};

document.getElementById('reset-btn').onclick = () => {
    buildings = JSON.parse(JSON.stringify(defaultBuildings));
    accessPoints = JSON.parse(JSON.stringify(defaultAccess));
    document.getElementById('seed').value = 42;
    document.getElementById('steps').value = 50;
    document.getElementById('steps-val').textContent = 50;
    updateLists();
};

document.getElementById('generate-btn').onclick = async () => {
    const genBtn = document.getElementById('generate-btn');
    genBtn.textContent = 'Generating...';
    setInputLock(true);

    abortController = new AbortController();
    const startTime = performance.now();

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                buildings,
                access_points: accessPoints,
                steps: parseInt(document.getElementById('steps').value),
                seed: parseInt(document.getElementById('seed').value),
                noise_std: parseFloat(document.getElementById('noise-std').value),
                corridor_seed_scale: parseFloat(document.getElementById('corridor-seed').value),
                fire_rate: parseFloat(document.getElementById('fire-rate').value),
                corridor_width: parseInt(document.getElementById('corridor-width').value),
                vertical_envelope: parseInt(document.getElementById('vertical-envelope').value),
                threshold: parseFloat(document.getElementById('threshold').value),
                update_scale: parseFloat(document.getElementById('update-scale').value)
            }),
            signal: abortController.signal
        });

        const data = await response.json();
        if (data.detail) throw new Error(data.detail);

        renderVoxels(data.structure, data.existing, data.access, data.anchors);
        document.getElementById('gen-time').textContent = `Generation: ${Math.round(performance.now() - startTime)} ms`;
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log("Generation was aborted by the user.");
            updatePreview(); // Show constraints again after abort
        } else {
            alert("Error generating design: " + err.message);
        }
    } finally {
        setInputLock(false);
        genBtn.textContent = 'Generate Design';
        abortController = null;
    }
};

function bindVisibilityControls() {
    const toggles = [
        ['structure', 'show-structure'],
        ['existing', 'show-existing'],
        ['access', 'show-access'],
        ['anchors', 'show-anchors'],
    ];

    toggles.forEach(([key, id]) => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', () => {
                visibilitySettings[key] = checkbox.checked;
                if (lastRender) {
                    renderVoxels(lastRender.structure, lastRender.existing, lastRender.access, lastRender.anchors);
                } else {
                    updatePreview();
                }
            });
        }
    });
}

// Start
initThree();
updateLists();
bindStyleControls();
bindVisibilityControls();
window.onresize = () => {
    camera.aspect = viewer.clientWidth / viewer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
};

window.addEventListener('keydown', (event) => {
    if (!transformControls || isGenerating) return;
    if (event.key.toLowerCase() === 't') {
        transformControls.setMode('translate');
    }
    if (event.key.toLowerCase() === 's') {
        transformControls.setMode('scale');
    }
});
