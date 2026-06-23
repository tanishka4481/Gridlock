// App State
let activeTab = 'triage';
let incidentQueue = [];
let routePoints = {
    origin: { lat: 12.9716, lng: 77.5946, marker: null },
    destination: { lat: 12.9130, lng: 77.6390, marker: null }
};
let routePickerMode = 'origin'; // 'origin' or 'destination'
let stationPools = {};
let activeRouteLayers = {
    dijkstra: null,
    astar: null,
    tomtom: null,
    incidentDecay: null
};

// Map Initialization
const map = L.map('map', {
    zoomControl: false // Position manually below
}).setView([12.9716, 77.5946], 12);

L.control.zoom({
    position: 'topright'
}).addTo(map);

// Minimalist CartoDB Positron base tile layer
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// Heatmap Layer holder
let heatmapLayer = null;
let clickedMarker = null;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    fetchMetadata();
    fetchHeatmapData();
    initForms();
    initRoutePicker();
    initStationControls();
});

// Tab Transitions Setup
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const sections = document.querySelectorAll('.panel-section');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            btn.classList.add('active');
            activeTab = btn.getAttribute('data-tab');
            document.getElementById(`panel-${activeTab}`).classList.add('active');
            
            // Map click mode resets
            removeClickedMarker();
            
            // Toggle panels depending on tab states
            if (activeTab === 'routing') {
                showRouteMarkers();
            } else {
                hideRouteMarkers();
            }
        });
    });
}

// Fetch form selectors lists from server
async function fetchMetadata() {
    try {
        const response = await fetch('/api/meta_data');
        const data = await response.json();
        
        // Populate selectors
        populateSelect('event-cause', data.categories.event_cause);
        populateSelect('event-type', data.categories.event_type);
        populateSelect('priority', data.categories.priority);
        populateSelect('veh-type', data.categories.veh_type);
        populateSelect('road-closure', data.categories.requires_road_closure);
        populateSelect('police-station', data.categories.police_station);
        populateSelect('corridor', data.categories.corridor);
        
        stationPools = data.station_pools;
        renderStationConfig();
    } catch (err) {
        console.error("Failed to fetch initial parameters: ", err);
    }
}

function populateSelect(id, values) {
    const select = document.getElementById(id);
    if (!select) return;
    select.innerHTML = '';
    values.forEach(val => {
        const opt = document.createElement('option');
        opt.value = val;
        opt.textContent = val.replace(/_/g, ' ').toUpperCase();
        select.appendChild(opt);
    });
}

// Fetch historical heatmap overlays
async function fetchHeatmapData() {
    try {
        const response = await fetch('/api/historical_heatmap');
        const points = await response.json();
        
        if (points && points.length > 0) {
            const heatArray = points.map(p => [p.lat, p.lng, 0.5]);
            heatmapLayer = L.heatLayer(heatArray, {
                radius: 12,
                blur: 8,
                maxZoom: 15,
                max: 1.0,
                gradient: {0.4: '#CCCCCC', 0.65: '#FFCC00', 1.0: '#FF5E00'}
            }).addTo(map);
        }
    } catch (err) {
        console.error("Failed to load historical heatmap points: ", err);
    }
}

// Map Click Interactions
map.on('click', (e) => {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;
    
    if (activeTab === 'triage') {
        document.getElementById('latitude').value = lat.toFixed(6);
        document.getElementById('longitude').value = lng.toFixed(6);
        
        // Place click marker
        removeClickedMarker();
        clickedMarker = L.marker([lat, lng], {
            icon: L.divIcon({
                className: 'custom-div-icon',
                html: "<div style='background-color:#FF5E00; width:12px; height:12px; border-radius:50%; border:2px solid white;'></div>",
                iconSize: [12, 12],
                iconAnchor: [6, 6]
            })
        }).addTo(map);
    } 
    else if (activeTab === 'routing') {
        updateRouteCoordinate(routePickerMode, lat, lng);
    }
});

function removeClickedMarker() {
    if (clickedMarker) {
        map.removeLayer(clickedMarker);
        clickedMarker = null;
    }
}

// Form Handlers (Queue Operations)
function initForms() {
    const btnAddQueue = document.getElementById('btn-add-queue');
    const btnClearQueue = document.getElementById('btn-clear-queue');
    const btnRunOpt = document.getElementById('btn-run-optimization');
    
    btnAddQueue.addEventListener('click', () => {
        const latVal = document.getElementById('latitude').value;
        const lonVal = document.getElementById('longitude').value;
        
        if (!latVal || !lonVal) {
            alert("Please specify coordinates by clicking on the map first.");
            return;
        }
        
        const dateVal = document.getElementById('incident-date').value || new Date().toISOString().split('T')[0];
        const timeVal = document.getElementById('incident-time').value || '12:00';
        
        const payload = {
            id: `Q_${String(incidentQueue.length + 1).padStart(3, '0')}`,
            start_datetime: `${dateVal} ${timeVal}:00+00`,
            latitude: parseFloat(latVal),
            longitude: parseFloat(lonVal),
            requires_road_closure: document.getElementById('road-closure').value,
            priority: document.getElementById('priority').value,
            veh_type: document.getElementById('veh-type').value,
            corridor: document.getElementById('corridor').value,
            event_cause: document.getElementById('event-cause').value,
            event_type: document.getElementById('event-type').value,
            zone: 'admin zone',
            police_station: document.getElementById('police-station').value,
            junction: document.getElementById('junction').value
        };
        
        incidentQueue.push(payload);
        updateQueueTable();
        removeClickedMarker();
        
        // Reset forms coords
        document.getElementById('latitude').value = '';
        document.getElementById('longitude').value = '';
    });
    
    btnClearQueue.addEventListener('click', () => {
        incidentQueue = [];
        updateQueueTable();
        document.getElementById('allocation-container').innerHTML = `<div class="empty-state">Submit queue and execute Knapsack Solver to show optimized station resource deployment tables.</div>`;
    });
    
    btnRunOpt.addEventListener('click', async () => {
        if (incidentQueue.length === 0) {
            alert("Incident queue is empty. Add events first.");
            return;
        }
        
        try {
            const response = await fetch('/api/optimize_queue', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(incidentQueue)
            });
            const results = await response.json();
            renderAllocations(results);
        } catch (err) {
            console.error("Solver error: ", err);
            alert("Allocation solver failed. Check server connection.");
        }
    });
}

function updateQueueTable() {
    const tbody = document.getElementById('queue-tbody');
    const badge = document.getElementById('queue-count');
    badge.textContent = `${incidentQueue.length} Events`;
    
    if (incidentQueue.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No incidents queued. Click map to log events.</td></tr>`;
        updateRouteIncidentDropdown();
        return;
    }
    
    tbody.innerHTML = '';
    incidentQueue.forEach((inc, idx) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-family: 'IBM Plex Mono'; font-weight: 500;">${inc.id}</td>
            <td>${inc.junction}</td>
            <td>${inc.event_cause.replace(/_/g, ' ').toUpperCase()}</td>
            <td>${inc.priority.toUpperCase()}</td>
            <td>${inc.police_station.toUpperCase()}</td>
            <td><button class="action-btn" onclick="removeQueueItem(${idx})">Remove</button></td>
        `;
        tbody.appendChild(tr);
    });
    
    updateRouteIncidentDropdown();
}

window.removeQueueItem = function(index) {
    incidentQueue.splice(index, 1);
    // Recalculate IDs
    incidentQueue.forEach((inc, i) => {
        inc.id = `Q_${String(i + 1).padStart(3, '0')}`;
    });
    updateQueueTable();
};

function renderAllocations(data) {
    const container = document.getElementById('allocation-container');
    if (!data || data.length === 0) {
        container.innerHTML = `<div class="empty-state">No allocation logs returned.</div>`;
        return;
    }
    
    container.innerHTML = '';
    
    // Group results by police station
    const stationsMap = {};
    data.forEach(item => {
        // Resolve station matching the incident ID
        const matched = incidentQueue.find(i => i.id === item.id);
        const st = matched ? matched.police_station : 'unknown';
        if (!stationsMap[st]) stationsMap[st] = [];
        stationsMap[st].push(item);
    });
    
    Object.keys(stationsMap).forEach(station => {
        const group = stationsMap[station];
        const card = document.createElement('div');
        card.className = 'allocation-card';
        
        let hasDeficit = group.some(i => i.status.toLowerCase().includes('under'));
        if (hasDeficit) {
            card.style.borderColor = 'var(--primary-orange)';
        }
        
        let html = `<h4>Station: ${station.toUpperCase()}</h4>`;
        group.forEach(item => {
            const statusClass = item.status.toLowerCase().includes('under') ? 'under-res' : 'managed';
            html += `
                <div class="alloc-row">
                    <span class="label">ID</span>
                    <span style="font-weight: 600;">${item.id} (${item.impact.toUpperCase()})</span>
                </div>
                <div class="alloc-row">
                    <span class="label">Cops (Alloc/Req)</span>
                    <span>${item.allocated_cops} / ${item.demanded_cops}</span>
                </div>
                <div class="alloc-row">
                    <span class="label">Barricades</span>
                    <span>${item.allocated_barricades} / ${item.demanded_barricades}</span>
                </div>
                <div class="alloc-row" style="margin-bottom: 8px;">
                    <span class="label">Status</span>
                    <span class="alloc-status ${statusClass}">${item.status.toUpperCase()}</span>
                </div>
            `;
        });
        
        card.innerHTML = html;
        container.appendChild(card);
    });
}

// Route Picker Logic
function initRoutePicker() {
    const buttons = document.querySelectorAll('.picker-btn');
    const btnRunRoute = document.getElementById('btn-run-route');
    const selectLink = document.getElementById('route-incident-select');
    const btnCloseDiag = document.getElementById('btn-close-diagnostics');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            routePickerMode = btn.getAttribute('data-mode');
        });
    });
    
    selectLink.addEventListener('change', () => {
        const val = selectLink.value;
        if (val === 'none') return;
        
        const matched = incidentQueue.find(i => i.id === val);
        if (!matched) return;
        
        // Auto lock origin to incident location
        updateRouteCoordinate('origin', matched.latitude, matched.longitude);
        
        // Match station coordinates
        const stationCoords = {
            'peenya': [13.0358, 77.5140],
            'hsr layout': [12.9130, 77.6390],
            'wilson garden': [12.9460, 77.5920],
            'sadashivanagar': [13.0070, 77.5800],
            'cubbon park': [12.9740, 77.6010],
            'kengeri': [12.9180, 77.4840],
            'hebbala': [13.0360, 77.5930]
        };
        const dest = stationCoords[matched.police_station] || [12.9716, 77.5946];
        updateRouteCoordinate('destination', dest[0], dest[1]);
    });
    
    // Sliders
    const slider = document.getElementById('sim-hour');
    const display = document.getElementById('sim-hour-val');
    slider.addEventListener('input', () => {
        display.textContent = `${String(slider.value).padStart(2, '0')}:00`;
    });
    
    btnRunRoute.addEventListener('click', calculateRoutes);
    
    btnCloseDiag.addEventListener('click', () => {
        document.getElementById('diagnostics-panel').style.display = 'none';
        clearPathLayers();
    });
}

function updateRouteCoordinate(mode, lat, lng) {
    routePoints[mode].lat = lat;
    routePoints[mode].lng = lng;
    document.getElementById(`txt-${mode}`).textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
    
    // Redraw markers
    if (routePoints[mode].marker) {
        map.removeLayer(routePoints[mode].marker);
    }
    
    const color = mode === 'origin' ? 'green' : 'red';
    routePoints[mode].marker = L.marker([lat, lng], {
        draggable: true,
        icon: L.divIcon({
            className: 'custom-div-icon',
            html: `<div style='background-color:${color}; width:16px; height:16px; border-radius:50%; border:3px solid white; box-shadow:0 2px 5px rgba(0,0,0,0.3);'></div>`,
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        })
    }).addTo(map);
    
    routePoints[mode].marker.on('dragend', (e) => {
        const position = e.target.getLatLng();
        updateRouteCoordinate(mode, position.lat, position.lng);
    });
}

function showRouteMarkers() {
    if (!routePoints.origin.marker) updateRouteCoordinate('origin', routePoints.origin.lat, routePoints.origin.lng);
    if (!routePoints.destination.marker) updateRouteCoordinate('destination', routePoints.destination.lat, routePoints.destination.lng);
}

function hideRouteMarkers() {
    if (routePoints.origin.marker) {
        map.removeLayer(routePoints.origin.marker);
        routePoints.origin.marker = null;
    }
    if (routePoints.destination.marker) {
        map.removeLayer(routePoints.destination.marker);
        routePoints.destination.marker = null;
    }
    clearPathLayers();
}

function updateRouteIncidentDropdown() {
    const select = document.getElementById('route-incident-select');
    select.innerHTML = '<option value="none">Manual Coordinate Simulation Mode</option>';
    
    incidentQueue.forEach(inc => {
        const opt = document.createElement('option');
        opt.value = inc.id;
        opt.textContent = `[${inc.id}] ${inc.junction} - ${inc.police_station.toUpperCase()}`;
        select.appendChild(opt);
    });
}

async function calculateRoutes() {
    const payload = {
        orig_lat: routePoints.origin.lat,
        orig_lon: routePoints.origin.lng,
        dest_lat: routePoints.destination.lat,
        dest_lon: routePoints.destination.lng,
        sim_hour: parseInt(document.getElementById('sim-hour').value),
        sim_day: parseInt(document.getElementById('sim-day').value)
    };
    
    // Check if synced to an incident coordinate link
    const selectLink = document.getElementById('route-incident-select');
    if (selectLink.value !== 'none') {
        const matched = incidentQueue.find(i => i.id === selectLink.value);
        if (matched) {
            payload.active_incident = matched;
        }
    }
    
    try {
        const response = await fetch('/api/find_route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        
        if (result.error) {
            alert(result.error);
            return;
        }
        
        drawRoutes(result.dijkstra_route, result.astar_route, result.tomtom_route, payload.active_incident);
        renderMetricsTable(result.metrics_table);
        
        // Show diagnostics panel
        document.getElementById('diagnostics-panel').style.display = 'flex';
    } catch (err) {
        console.error("Routing calculation failed: ", err);
        alert("Dynamic routing path calculation failed. Graph boundaries may be out of bounds.");
    }
}

function clearPathLayers() {
    if (activeRouteLayers.dijkstra) map.removeLayer(activeRouteLayers.dijkstra);
    if (activeRouteLayers.astar) map.removeLayer(activeRouteLayers.astar);
    if (activeRouteLayers.tomtom) map.removeLayer(activeRouteLayers.tomtom);
    if (activeRouteLayers.incidentDecay) map.removeLayer(activeRouteLayers.incidentDecay);
}

function drawRoutes(dijkstraCoords, astarCoords, tomtomCoords, activeIncident) {
    clearPathLayers();
    
    // Draw Dijkstra (Solid Blue)
    activeRouteLayers.dijkstra = L.polyline(dijkstraCoords, {
        color: '#007AFF',
        weight: 6,
        opacity: 0.85
    }).addTo(map);
    
    // Draw A* (Dashed Green/Blue)
    activeRouteLayers.astar = L.polyline(astarCoords, {
        color: '#34C759',
        weight: 4,
        opacity: 0.85,
        dashArray: '8, 8'
    }).addTo(map);
    
    // Draw TomTom (Solid Orange)
    if (tomtomCoords && tomtomCoords.length > 0) {
        activeRouteLayers.tomtom = L.polyline(tomtomCoords, {
            color: '#FF9500',
            weight: 5,
            opacity: 0.9
        }).addTo(map);
    }
    
    // Zoom map bounds
    const bounds = L.latLngBounds([routePoints.origin, routePoints.destination]);
    
    // Draw incident decay radius overlay if locked
    if (activeIncident) {
        const lat = parseFloat(activeIncident.latitude);
        const lng = parseFloat(activeIncident.longitude);
        
        // Resolve radius based on cause
        const cause = String(activeIncident.event_cause).toLowerCase();
        const isClosure = String(activeIncident.requires_road_closure).toUpperCase() === 'TRUE';
        let radius = 200.0;
        if (cause.includes('public_event')) radius = 800.0;
        else if (isClosure) radius = 500.0;
        else if (cause.includes('accident')) radius = 300.0;
        
        activeRouteLayers.incidentDecay = L.circle([lat, lng], {
            radius: radius,
            color: '#FF5E00',
            fillColor: '#FF5E00',
            fillOpacity: 0.15,
            weight: 1,
            dashArray: '4, 4'
        }).addTo(map);
        
        bounds.extend([lat, lng]);
    }
    
    map.fitBounds(bounds, { padding: [50, 50] });
}

function renderMetricsTable(tableData) {
    const tbody = document.getElementById('metrics-tbody');
    tbody.innerHTML = '';
    tableData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-family: 'Outfit'; font-weight: 500;">${row.report}</td>
            <td>${row.dijkstra}</td>
            <td>${row.astar}</td>
            <td>${row.tomtom || 'N/A'}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Station supply config adjustments
function initStationControls() {
    const btnSave = document.getElementById('btn-save-config');
    const btnReset = document.getElementById('btn-reset-config');
    
    btnSave.addEventListener('click', async () => {
        const updatedPools = {};
        Object.keys(stationPools).forEach(st => {
            const copsInput = document.getElementById(`config-${st}-cops`);
            const barInput = document.getElementById(`config-${st}-barricades`);
            
            if (copsInput && barInput) {
                updatedPools[st] = {
                    cops: parseInt(copsInput.value),
                    barricades: parseInt(barInput.value)
                };
            }
        });
        
        try {
            const response = await fetch('/api/station_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updatedPools)
            });
            const data = await response.json();
            stationPools = data.pools;
            alert("Station supply configurations saved successfully in memory.");
        } catch (err) {
            console.error("Config save error: ", err);
            alert("Failed to apply configurations.");
        }
    });
    
    btnReset.addEventListener('click', async () => {
        const defaults = {
            'peenya': {'cops': 15, 'barricades': 20},
            'hsr layout': {'cops': 20, 'barricades': 25},
            'wilson garden': {'cops': 12, 'barricades': 15},
            'sadashivanagar': {'cops': 10, 'barricades': 12},
            'cubbon park': {'cops': 25, 'barricades': 30},
            'kengeri': {'cops': 12, 'barricades': 15},
            'hebbala': {'cops': 18, 'barricades': 22},
            'unknown': {'cops': 8, 'barricades': 10}
        };
        
        try {
            const response = await fetch('/api/station_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(defaults)
            });
            const data = await response.json();
            stationPools = data.pools;
            renderStationConfig();
            alert("Defaults restored.");
        } catch (err) {
            console.error("Config reset error: ", err);
        }
    });
}

function renderStationConfig() {
    const container = document.getElementById('station-config-container');
    if (!container) return;
    container.innerHTML = '';
    
    // Sort stations alphabetically
    const sortedStations = Object.keys(stationPools).sort();
    
    sortedStations.forEach(st => {
        const item = stationPools[st];
        const row = document.createElement('div');
        row.className = 'config-item';
        row.innerHTML = `
            <h4>${st.replace(/_/g, ' ').toUpperCase()} Pool</h4>
            <div class="form-row" style="margin-top: 5px;">
                <div class="form-col">
                    <label>Cops</label>
                    <input type="number" id="config-${st}-cops" value="${item.cops}" min="0" max="100">
                </div>
                <div class="form-col">
                    <label>Barricades</label>
                    <input type="number" id="config-${st}-barricades" value="${item.barricades}" min="0" max="100">
                </div>
            </div>
        `;
        container.appendChild(row);
    });
}
