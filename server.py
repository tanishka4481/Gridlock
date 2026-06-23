import os
import sys
import datetime
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory


# Resolve paths relative to the script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tomtom import TomTomSuite
import recommendation_engine
from recommendation_engine import run_integrated_traffic_pipeline, process_single_incident, TrafficResourceOptimizer
from digital_twin import (load_bengaluru_graph_with_priors, load_empirical_traffic_priors,
                           compute_dynamic_network_states, compare_scenarios, DigitalTwin)
import osmnx as ox

app = Flask(__name__, static_folder='static', static_url_path='')

# Load reference datasets
historical_df = pd.DataFrame()
theme_path = os.path.join(BASE_DIR, "theme2.csv")
if os.path.exists(theme_path):
    dtype_dict = {
        'id': 'str', 'event_type': 'str', 'requires_road_closure': 'str', 'priority': 'str',
        'corridor': 'str', 'zone': 'str', 'police_station': 'str',
        'veh_type': 'str', 'junction': 'str', 'event_cause': 'str',
        'latitude': 'float64', 'longitude': 'float64'
    }
    historical_df = pd.read_csv(theme_path, dtype=dtype_dict)
    historical_df = historical_df.dropna(subset=['latitude', 'longitude', 'start_datetime'])
    historical_df['start_datetime'] = pd.to_datetime(historical_df['start_datetime'], format='ISO8601', utc=True, errors='coerce')

# Station supply state
STATION_SUPPLY_POOLS = {
    'peenya': {'cops': 15, 'barricades': 20},
    'hsr layout': {'cops': 20, 'barricades': 25},
    'wilson garden': {'cops': 12, 'barricades': 15},
    'sadashivanagar': {'cops': 10, 'barricades': 12},
    'cubbon park': {'cops': 25, 'barricades': 30},
    'kengeri': {'cops': 12, 'barricades': 15},
    'hebbala': {'cops': 18, 'barricades': 22},
    'unknown': {'cops': 8, 'barricades': 10}
}
recommendation_engine.STATION_SUPPLY_POOLS = STATION_SUPPLY_POOLS

# Load physical network graph & temporal priors
print("[API] Loading OSM Bengaluru graph map & traffic logs priors matrix...")
G_base = load_bengaluru_graph_with_priors(cache_filename=os.path.join(BASE_DIR, "bengaluru_network.graphml"))
mined_traffic_matrix, fallback_speed, junctions_registry = load_empirical_traffic_priors(os.path.join(BASE_DIR, "road_network_priors.csv"))
print("[API] Network graph loaded successfully.")

# Initialize TomTom live decision support engines
print("[API] Initializing TomTom decision-support modules...")
tomtom_api = TomTomSuite()
twin_engine = DigitalTwin()
optimizer = TrafficResourceOptimizer()

# Expose UI Categories dynamically matching historical records
def get_categorical_bounds(col_name, defaults):
    if not historical_df.empty and col_name in historical_df.columns:
        vals = historical_df[col_name].dropna().astype(str).str.strip().unique().tolist()
        return sorted([v for v in vals if v.lower() != 'unknown'])
    return defaults

CATEGORY_LISTS = {
    'event_cause': get_categorical_bounds('event_cause', ['vehicle_breakdown', 'accident', 'tree_fall', 'public_event', 'others']),
    'event_type': ['unplanned', 'planned'],
    'priority': ['High', 'Low'],
    'veh_type': get_categorical_bounds('veh_type', ['heavy_vehicle', 'bmtc_bus', 'private_bus', 'lcv', 'car', 'unknown']),
    'requires_road_closure': ['TRUE', 'FALSE'],
    'police_station': get_categorical_bounds('police_station', ['peenya', 'hsr layout', 'wilson garden', 'sadashivanagar', 'cubbon park', 'kengeri', 'hebbala']),
    'corridor': get_categorical_bounds('corridor', ['tumkur road', 'orr east 1', 'cbd 2', 'non-corridor'])
}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/meta_data', methods=['GET'])
def get_metadata():
    return jsonify({
        'categories': CATEGORY_LISTS,
        'junctions': junctions_registry,
        'station_pools': STATION_SUPPLY_POOLS,
        'diagnostics': {
            'mined_segments_count': len(mined_traffic_matrix),
            'fallback_speed_kph': fallback_speed,
            'graph_nodes_count': len(G_base.nodes),
            'graph_edges_count': len(G_base.edges)
        }
    })

@app.route('/api/historical_heatmap', methods=['GET'])
def get_heatmap_points():
    if historical_df.empty:
        return jsonify([])
    # Sample subset to prevent front-end browser lag
    sample_size = min(3000, len(historical_df))
    subset = historical_df.sample(n=sample_size)[['latitude', 'longitude']].dropna()
    coords = [{"lat": float(r['latitude']), "lng": float(r['longitude'])} for _, r in subset.iterrows()]
    return jsonify(coords)

@app.route('/api/predict_incident', methods=['POST'])
def predict_single():
    data = request.json
    try:
        # Build mock record
        incident_row = {
            'id': data.get('id', 'INC_01'),
            'start_datetime': data.get('start_datetime', datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+00')),
            'latitude': float(data['latitude']),
            'longitude': float(data['longitude']),
            'requires_road_closure': data.get('requires_road_closure', 'FALSE'),
            'priority': data.get('priority', 'Low'),
            'veh_type': data.get('veh_type', 'car'),
            'corridor': data.get('corridor', 'non-corridor'),
            'event_cause': data.get('event_cause', 'others'),
            'event_type': data.get('event_type', 'unplanned'),
            'zone': data.get('zone', 'unknown'),
            'police_station': data.get('police_station', 'unknown'),
            'junction': data.get('junction', 'unknown')
        }
        res = process_single_incident(incident_row, historical_df)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

STATION_COORDINATES = {
    'peenya': [13.0358, 77.5140],
    'hsr layout': [12.9130, 77.6390],
    'wilson garden': [12.9460, 77.5920],
    'sadashivanagar': [13.0070, 77.5800],
    'cubbon park': [12.9740, 77.6010],
    'kengeri': [12.9180, 77.4840],
    'hebbala': [13.0360, 77.5930]
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@app.route('/api/optimize_queue', methods=['POST'])
def optimize_queue():
    data = request.json  # List of incident payloads
    if not data or len(data) == 0:
        return jsonify([])
    try:
        # 1. Parse incidents and build incidents_list
        incidents_list = []
        incident_locs = []
        for idx, item in enumerate(data):
            inc_id = item.get('id', f"Q_{str(idx+1).rjust(3, '0')}")
            lat = float(item['latitude'])
            lon = float(item['longitude'])
            incident_locs.append([lat, lon])
            
            # Fetch live traffic flow from TomTom at incident location
            traffic_flow = tomtom_api.get_live_traffic_flow(lat, lon)
            
            # Predict static impact via prediction pipeline model
            from recommendation_engine import predict_incident_impact
            predicted_impact = predict_incident_impact(item)
            static_prob = 0.85 if predicted_impact == "High" else 0.25
            
            # Compute dynamic urgency score using TomTom live traffic data
            urgency_score, congestion_ratio = optimizer.compute_dynamic_urgency_uplift(static_prob, traffic_flow)
            
            # Calculate demanded personnel and equipment dynamically based on live traffic
            closure_active = str(item.get('requires_road_closure', 'FALSE')).upper().strip() == 'TRUE'
            cause = item.get('event_cause', 'others')
            demands = optimizer.calculate_rule_demand(cause, congestion_ratio, closure_active)
            
            incidents_list.append({
                "id": inc_id,
                "latitude": lat,
                "longitude": lon,
                "urgency_score": urgency_score,
                "congestion_ratio": congestion_ratio,
                "demanded_cops": demands["demanded_cops"],
                "demanded_barricades": demands["demanded_barricades"],
                "impact": predicted_impact
            })
            
        # 2. Build stations_list and retrieve many-to-many travel times
        station_names = list(STATION_SUPPLY_POOLS.keys())
        station_names = [st for st in station_names if st != 'unknown']
        
        origins = []
        for name in station_names:
            coords = STATION_COORDINATES.get(name, [12.9716, 77.5946])
            origins.append(coords)
            
        destinations = incident_locs
        
        # Calculate sync matrix travel times using TomTom Matrix v2 Sync
        matrix_res = []
        try:
            matrix_res = tomtom_api.calculate_matrix_v2(origins, destinations)
        except Exception as me:
            print(f"[Warning] Matrix calculation failed: {me}")
            
        # Map matrix results to stations list ETA mapping dictionary
        eta_map = {}  # maps (station_name, incident_id) -> travel_time_mins
        for cell in matrix_res:
            o_idx = cell.get('originIndex')
            d_idx = cell.get('destinationIndex')
            if o_idx is not None and d_idx is not None and 'routeSummary' in cell:
                station_name = station_names[o_idx]
                incident_id = incidents_list[d_idx]['id']
                travel_time_sec = cell['routeSummary']['travelTimeInSeconds']
                eta_map[(station_name, incident_id)] = travel_time_sec / 60.0
                
        # Fill in missing values using Haversine calculation fallback
        for st_name in station_names:
            st_coords = STATION_COORDINATES.get(st_name, [12.9716, 77.5946])
            for inc in incidents_list:
                inc_id = inc['id']
                if (st_name, inc_id) not in eta_map:
                    dist_km = haversine_distance(st_coords[0], st_coords[1], inc['latitude'], inc['longitude'])
                    speed_kph = fallback_speed if fallback_speed > 0 else 30.0
                    eta_map[(st_name, inc_id)] = (dist_km / speed_kph) * 60.0

        # Construct final stations list formatting with ETA dictionaries
        stations_list = []
        for st_name in station_names:
            pool = STATION_SUPPLY_POOLS.get(st_name, {'cops': 10, 'barricades': 10})
            
            # Map ETAs specifically for each incident in the active queue
            inc_eta_dict = {}
            for inc in incidents_list:
                inc_eta_dict[inc['id']] = eta_map.get((st_name, inc['id']), 10.0)
                
            stations_list.append({
                "station_name": st_name,
                "available_cops": pool.get('cops', 10),
                "available_barricades": pool.get('barricades', 10),
                "matrix_eta_mins": inc_eta_dict
            })
            
        # 3. Resolve allocations using MILP solver
        assignments = optimizer.allocate_resources_milp(incidents_list, stations_list)
        
        # 4. Map output to frontend schema
        formatted = []
        for assign in assignments:
            inc_id = assign["incident_id"]
            orig_inc = next((i for i in incidents_list if i["id"] == inc_id), {})
            impact = orig_inc.get("impact", "Low")
            
            allocated_cops = assign["allocated_cops"]
            demanded_cops = assign["demanded_cops"]
            allocated_barricades = assign["allocated_barricades"]
            demanded_barricades = assign["demanded_barricades"]
            
            status = "managed" if (allocated_cops >= demanded_cops and allocated_barricades >= demanded_barricades) else "under-res"
            signs = (1 if allocated_barricades > 4 else 0)
            
            formatted.append({
                'id': inc_id,
                'impact': impact,
                'demanded_cops': demanded_cops,
                'allocated_cops': allocated_cops,
                'demanded_barricades': demanded_barricades,
                'allocated_barricades': allocated_barricades,
                'signs': signs,
                'status': status
            })
            
        return jsonify(formatted)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route('/api/station_config', methods=['GET', 'POST'])
def handle_station_config():
    global STATION_SUPPLY_POOLS
    if request.method == 'POST':
        data = request.json  # New station bounds dict
        if data:
            STATION_SUPPLY_POOLS.update(data)
            recommendation_engine.STATION_SUPPLY_POOLS = STATION_SUPPLY_POOLS
            return jsonify({"status": "applied", "pools": STATION_SUPPLY_POOLS})
    return jsonify(STATION_SUPPLY_POOLS)

@app.route('/api/find_route', methods=['POST'])
def calculate_routes():
    data = request.json
    try:
        orig_lat = float(data['orig_lat'])
        orig_lon = float(data['orig_lon'])
        dest_lat = float(data['dest_lat'])
        dest_lon = float(data['dest_lon'])
        sim_hour = int(data.get('sim_hour', 18))
        sim_day = int(data.get('sim_day', 0))
        
        active_inc = data.get('active_incident') # Optional active incident dictionary
        active_inc_df = None
        if active_inc:
            active_inc_df = pd.DataFrame([active_inc])
            active_inc_df['latitude'] = active_inc_df['latitude'].astype(float)
            active_inc_df['longitude'] = active_inc_df['longitude'].astype(float)

        # Calculate dynamic edge penalties under simulation hour/day and incident shockwaves
        G_dynamic = compute_dynamic_network_states(
            G_base, 
            current_hour=sim_hour, 
            current_day=sim_day, 
            active_incident_df=active_inc_df,
            prior_traffic_matrix=mined_traffic_matrix,
            global_speed_fallback=fallback_speed,
            junctions_registry=junctions_registry
        )
        
        comparison_df, path_dijkstra, path_astar = compare_scenarios(
            G_dynamic, orig_lat, orig_lon, dest_lat, dest_lon
        )
        
        # Convert path node lists into coordinate polyline streams
        G_unproj = ox.project_graph(G_dynamic, to_crs="EPSG:4326")
        dijkstra_coords = [[float(G_unproj.nodes[n]['y']), float(G_unproj.nodes[n]['x'])] for n in path_dijkstra]
        astar_coords = [[float(G_unproj.nodes[n]['y']), float(G_unproj.nodes[n]['x'])] for n in path_astar]
        
        # Call TomTom Live Route API
        tomtom_route = None
        try:
            tomtom_route = tomtom_api.calculate_route(orig_lat, orig_lon, dest_lat, dest_lon)
        except Exception as te:
            print(f"[Warning] TomTom route fetch failed: {te}")

        tomtom_time_str = "N/A"
        tomtom_dist_str = "N/A"
        tomtom_coords = []
        if tomtom_route:
            raw_points = tomtom_route.get("coordinates", [])
            if raw_points:
                if isinstance(raw_points[0], dict):
                    tomtom_coords = [[float(pt["latitude"]), float(pt["longitude"])] for pt in raw_points]
                elif isinstance(raw_points[0], list):
                    pt0 = raw_points[0]
                    if len(pt0) >= 2:
                        if pt0[0] > pt0[1]:  # longitude (e.g. 77.6) > latitude (e.g. 12.9)
                            tomtom_coords = [[float(pt[1]), float(pt[0])] for pt in raw_points]
                        else:
                            tomtom_coords = [[float(pt[0]), float(pt[1])] for pt in raw_points]
            tomtom_time_str = f"{round(tomtom_route['time_sec']/60, 1)} mins"
            tomtom_dist_str = f"{round(tomtom_route['dist_meters']/1000, 2)} km"
            
        # Parse comparison metrics table and strip emojis
        metrics = []
        for _, row in comparison_df.iterrows():
            metric_name = str(row['Operational Metric Report'])
            dij_val = str(row['🔵 Dijkstra (Optimal)'])
            ast_val = str(row['🟢 A* (Heuristic)'])
            
            # Map TomTom metrics to the report row
            tom_val = "N/A"
            if metric_name == "Baseline Travel Time":
                tom_val = "N/A"
            elif metric_name == "Impact Mapped Travel Time":
                tom_val = tomtom_time_str
            elif metric_name == "Absolute Net Incident Delay":
                tom_val = "N/A"
            elif metric_name == "Relative Percentage Increase":
                tom_val = "N/A"
            elif metric_name == "Total Routing Distance":
                tom_val = tomtom_dist_str
            elif metric_name == "Intersections Traversed":
                tom_val = "N/A"

            metrics.append({
                'report': metric_name,
                'dijkstra': dij_val,
                'astar': ast_val,
                'tomtom': tom_val
            })
            
        return jsonify({
            'dijkstra_route': dijkstra_coords,
            'astar_route': astar_coords,
            'tomtom_route': tomtom_coords,
            'metrics_table': metrics
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
