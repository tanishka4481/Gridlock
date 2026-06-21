import os
import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

# ==========================================
# CONSTANTS & CONFIGURATION MATRIX (EDA LOOKUPS)
# ==========================================
CORRIDOR_LANES_LOOKUP = {
    'tumkur road': 4, 'orr east 1': 3, 'cbd 2': 2, 'non-corridor': 2, 'unknown': 2
}

PCU_MAP = {
    'heavy_vehicle': 3.0, 'bmtc_bus': 3.0, 'private_bus': 3.0,
    'lcv': 1.5, 'car': 1.0, 'unknown': 1.0
}

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

# Load artifacts strictly ONCE out of runtime execution loops
PIPELINE_PATH = 'traffic_impact_production_pipeline.pkl'
if os.path.exists(PIPELINE_PATH):
    print("Loading Global Production Artifacts into Memory...")
    GLOBAL_ARTIFACTS = joblib.load(PIPELINE_PATH)
else:
    GLOBAL_ARTIFACTS = None
    print("Warning: Production pipeline file missing. Ensure it is generated first.")


# ==========================================
# FUNCTION 1: PREDICT INCIDENT IMPACT
# ==========================================
def predict_incident_impact(incident_row, artifacts=GLOBAL_ARTIFACTS):
    """
    Ingests an incident data object and references warm memory structures 
    to output an isolated binary operational footprint vector.
    """
    if artifacts is None:
        return "Low"  # Clean deployment fallback proxy if pipeline file isn't populated
        
    model = artifacts['model']
    feature_cols = artifacts['features']
    encoders = artifacts['encoders']
    risk_map = artifacts['junction_risk_map']
    threshold = artifacts['threshold']
    
    start_dt = pd.to_datetime(incident_row['start_datetime'], utc=True)
    hour = start_dt.hour
    day_of_week = start_dt.dayofweek
    month = start_dt.month
    is_peak_hour = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
    
    closure_str = str(incident_row.get('requires_road_closure', 'FALSE')).upper().strip()
    requires_road_closure = 1 if closure_str == 'TRUE' else 0
    
    veh_type = str(incident_row.get('veh_type', 'unknown')).strip().lower()
    has_vehicle = 0 if veh_type == 'unknown' else 1
    
    junction = str(incident_row.get('junction', 'unknown')).strip()
    junction_risk = risk_map.get(junction, 0.5)
    
    feat_dict = {
        'hour': hour, 
        'day_of_week': day_of_week, 
        'month': month,
        'event_cause': str(incident_row.get('event_cause', 'unknown')).strip(),
        'event_type': str(incident_row.get('event_type', 'unknown')).strip(),
        'requires_road_closure': requires_road_closure,
        'priority': str(incident_row.get('priority', 'Low')).strip(),
        'corridor': str(incident_row.get('corridor', 'Non-corridor')).strip(),
        'zone': str(incident_row.get('zone', 'unknown')).strip(),
        'police_station': str(incident_row.get('police_station', 'unknown')).strip(),
        'veh_type': veh_type, 
        'is_peak_hour': is_peak_hour,
        'has_vehicle': has_vehicle, 
        'junction_risk': junction_risk
    }
    
    # Text indices conversion loop
    for col, encoder in encoders.items():
        if col in feat_dict:
            val = feat_dict[col]
            if val in encoder.classes_:
                feat_dict[col] = encoder.transform([val])[0]
            else:
                feat_dict[col] = encoder.transform(['unknown'])[0] if 'unknown' in encoder.classes_ else 0

    X_frame = pd.DataFrame([feat_dict])[feature_cols]
    
    try:
        prob_high = model.predict_proba(X_frame)[0, 1]
        predicted_class = 1 if prob_high > threshold else 0
        return "High" if predicted_class == 1 else "Low"
    except:
        return "Low"


# ==========================================
# FUNCTION 2: CALCULATE RULE DEMAND
# ==========================================
def calculate_rule_demand(incident_row, predicted_impact):
    """
    Computes baseline deployment matrix demands using incident severity, 
    vehicle constraints, and absolute time window multipliers.
    """
    priority = str(incident_row.get('priority', 'Low')).upper().strip()
    closure_str = str(incident_row.get('requires_road_closure', 'FALSE')).upper().strip()
    veh_type = str(incident_row.get('veh_type', 'unknown')).strip().lower()
    
    base_cops, base_barricades = (6, 8) if predicted_impact == "High" else (2, 2)
        
    if priority == "HIGH": 
        base_cops += 2
    if closure_str == "TRUE": 
        base_barricades += 4
    if veh_type in ['heavy_vehicle', 'bmtc_bus', 'private_bus']:
        base_cops += 1
        base_barricades += 2
        
    start_dt = pd.to_datetime(incident_row['start_datetime'], utc=True)
    hour = start_dt.hour
    is_peak = (8 <= hour <= 11) or (17 <= hour <= 20)
    
    time_multiplier = 2.5 if is_peak else 1.0
    
    raw_demand_cops = int(np.ceil(base_cops * time_multiplier))
    raw_demand_barricades = int(np.ceil(base_barricades * time_multiplier))
    
    return raw_demand_cops, raw_demand_barricades


# ==========================================
# FUNCTION 3: ADJUST BARRICADES BY PCU
# ==========================================
def adjust_barricades_by_pcu(incident_row, rule_matrix_barricades):
    veh_type = str(incident_row.get('veh_type', 'unknown')).strip().lower()
    corridor = str(incident_row.get('corridor', 'unknown')).strip().lower()
    
    pcu_weight = PCU_MAP.get(veh_type, 1.0)
    estimated_lanes = CORRIDOR_LANES_LOOKUP.get(corridor, 2)
    
    pcu_barricades = int(np.ceil(pcu_weight * estimated_lanes))
    return max(rule_matrix_barricades, pcu_barricades)


# ==========================================
# FUNCTION 4: EVALUATE SPATIAL DENSITY
# ==========================================
def evaluate_spatial_density(incident_row, historical_df, distance_threshold_meters=800.0, operational_days=30):
    if historical_df.empty: 
        return 0, 0, "NORMAL"
        
    curr_lat, curr_lon = float(incident_row['latitude']), float(incident_row['longitude'])
    curr_time = pd.to_datetime(incident_row['start_datetime'], utc=True)
    
    hist_times = pd.to_datetime(historical_df['start_datetime'], utc=True)
    time_delta_days = (curr_time - hist_times).dt.total_seconds() / (24 * 3600)
    filtered_hist = historical_df[(time_delta_days >= 0) & (time_delta_days <= operational_days)]
    
    if filtered_hist.empty: 
        return 0, 0, "NORMAL"
        
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        return 6371000.0 * (2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2.0)**2)))

    distances = haversine(curr_lon, curr_lat, filtered_hist['longitude'].values, filtered_hist['latitude'].values)
    return (2, 4, "HIGH FRICTION ZONE") if int(np.sum(distances <= distance_threshold_meters)) > 3 else (0, 0, "NORMAL")


# ==========================================
# FUNCTION 5: PROCESS SINGLE INCIDENT
# ==========================================
def process_single_incident(incident_row, historical_df=pd.DataFrame()):
    predicted_impact = predict_incident_impact(incident_row)
    raw_cops, raw_barricades = calculate_rule_demand(incident_row, predicted_impact)
    final_barricades = adjust_barricades_by_pcu(incident_row, raw_barricades)
    extra_cops, extra_barricades, zone_flag = evaluate_spatial_density(incident_row, historical_df)
    
    final_cops = raw_cops + extra_cops
    final_barricades += extra_barricades
    
    signs = (1 if final_barricades > 4 else 0) + (2 if str(incident_row.get('requires_road_closure', 'FALSE')).upper().strip() == 'TRUE' else 0)
    urgency = "IMMEDIATE" if (predicted_impact == "High" or zone_flag == "HIGH FRICTION ZONE") else "STANDARD"
    
    return {
        "incident_id": incident_row.get('id', 'UNKNOWN'), 
        "predicted_impact": predicted_impact,
        "cops_needed": final_cops, 
        "barricades_needed": final_barricades, 
        "diversion_signs": signs,
        "zone_flag": zone_flag, 
        "deployment_urgency": urgency,
        "police_station": str(incident_row.get('police_station', 'unknown')).strip().lower()
    }


# ==========================================
# FUNCTION 6: OPTIMIZE STATION RESOURCE POOL
# ==========================================
def optimize_station_resource_pool(incidents_dispatches, station_name):
    station_key = str(station_name).strip().lower()
    pool = STATION_SUPPLY_POOLS.get(station_key, STATION_SUPPLY_POOLS['unknown'])
    available_cops, available_barricades = pool['cops'], pool['barricades']
    
    for inc in incidents_dispatches:
        inc['orig_cops'] = inc['cops_needed']
        inc['orig_barricades'] = inc['barricades_needed']
        inc['orig_signs'] = inc['diversion_signs']
        
    total_cops_demanded = sum(inc['orig_cops'] for inc in incidents_dispatches)
    total_barricades_demanded = sum(inc['orig_barricades'] for inc in incidents_dispatches)
    
    if total_cops_demanded <= available_cops and total_barricades_demanded <= available_barricades:
        for inc in incidents_dispatches: 
            inc.update({'allocation_fraction': 1.0, 'status': "✅ Managed"})
        return incidents_dispatches

    prob = LpProblem("Station_Resource_Knapsack", LpMaximize)
    incident_ids = [inc['incident_id'] for inc in incidents_dispatches]
    alloc_vars = LpVariable.dicts("alloc", incident_ids, lowBound=0.0, upBound=1.0)
    
    weights = {
        inc['incident_id']: (1.0 + (2.0 if inc['predicted_impact'] == "High" else 0.0) + 
                             (1.5 if inc['zone_flag'] == "HIGH FRICTION ZONE" else 0.0)) 
        for inc in incidents_dispatches
    }
    
    prob += lpSum([
        alloc_vars[inc['incident_id']] * weights[inc['incident_id']] * inc['orig_cops'] 
        for inc in incidents_dispatches
    ])
    
    prob += lpSum([alloc_vars[inc['incident_id']] * inc['orig_cops'] for inc in incidents_dispatches]) <= available_cops, "Cop_Supply"
    prob += lpSum([alloc_vars[inc['incident_id']] * inc['orig_barricades'] for inc in incidents_dispatches]) <= available_barricades, "Barricade_Supply"
    prob.solve()
    
    for inc in incidents_dispatches:
        fraction = round(alloc_vars[inc['incident_id']].varValue, 4) if alloc_vars[inc['incident_id']].varValue is not None else 0.0
        inc.update({
            'allocation_fraction': fraction,
            'cops_needed': int(np.floor(inc['orig_cops'] * fraction)),
            'barricades_needed': int(np.floor(inc['orig_barricades'] * fraction)),
            'diversion_signs': int(np.floor(inc['orig_signs'] * fraction)),
            'status': "✅ Managed" if fraction >= 0.95 else "⚠️ Under-res"
        })
    return incidents_dispatches


# ==========================================
# FUNCTION 7: RUN INTEGRATED TRAFFIC PIPELINE
# ==========================================
def run_integrated_traffic_pipeline(active_incidents_df, historical_df=pd.DataFrame()):
    initial_dispatches = [process_single_incident(row, historical_df) for _, row in active_incidents_df.iterrows()]
    dispatch_df = pd.DataFrame(initial_dispatches)
    final_allocated_list = []
    
    for station, group in dispatch_df.groupby('police_station'):
        group_records = group.to_dict('records')
        if len(group_records) > 1:
            final_allocated_list.extend(optimize_station_resource_pool(group_records, station))
        else:
            for rec in group_records: 
                rec['orig_cops'] = rec['cops_needed']
                rec['orig_barricades'] = rec['barricades_needed']
                rec['allocation_fraction'] = 1.0
                rec['status'] = "✅ Managed"
            final_allocated_list.extend(group_records)
            
    report_df = pd.DataFrame(final_allocated_list)
    presentation_df = report_df[[
        'incident_id', 'predicted_impact', 'orig_cops', 'cops_needed', 
        'orig_barricades', 'barricades_needed', 'diversion_signs', 'status'
    ]].copy()
    
    presentation_df.columns = [
        'Incident', 'Impact', 'Demanded Cops', 'Allocated Cops', 
        'Demanded Barricades', 'Allocated Barricades', 'Signs', 'Status'
    ]
    return presentation_df