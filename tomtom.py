import os
import requests
import json
from dotenv import load_dotenv

# Initialize environment variables safely
load_dotenv()
API_KEY = os.getenv("TOMTOM_API_KEY")

class TomTomSuite:
    def __init__(self):
        self.key = API_KEY
        if not self.key:
            print("[Warning] TOMTOM_API_KEY is not configured inside your environment variables.")

    # 1. GEOCODING API
    def geocode_address(self, query="Richmond Road, Bengaluru"):
        url = f"https://api.tomtom.com/search/2/geocode/{requests.utils.quote(query)}.json"
        params = {"key": self.key, "limit": 1}
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200 and res.json().get('results'):
            pos = res.json()['results'][0]['position']
            return {"lat": pos['lat'], "lon": pos['lon']}
        return None

    # 2. REVERSE GEOCODING API
    def reverse_geocode(self, lat, lon):
        url = f"https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"
        params = {"key": self.key}
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200 and res.json().get('addresses'):
            return res.json()['addresses'][0]['address']['freeformAddress']
        return None

    # 3. SEARCH API
    def search_poi(self, query="Police Station", lat=12.9700, lon=77.6100):
        url = f"https://api.tomtom.com/search/2/search/{requests.utils.quote(query)}.json"
        params = {"key": self.key, "lat": lat, "lon": lon, "radius": 5000, "limit": 3}
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            return [{"name": r['poi']['name'], "dist": r['dist']} for r in res.json().get('results', []) if 'poi' in r]
        return []

    # 4. TRAFFIC API (Live Flow)
    def get_live_traffic_flow(self, lat, lon):
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/18/json"
        params = {"key": self.key, "point": f"{lat},{lon}", "unit": "kph", "thickness": 10}
        res = requests.get(url, params=params, timeout=5)
        
        if res.status_code == 200 and 'flowSegmentData' in res.json():
            fd = res.json()['flowSegmentData']
            return {"current": fd.get("currentSpeed"), "free_flow": fd.get("freeFlowSpeed"), "status": "Matched Live Stream"}
        
        return {"current": 35.0, "free_flow": 35.0, "status": "Fallback default speed assigned"}

    # 5. TRAFFIC STATS API
    def check_traffic_stats_job(self, job_id="sample-job-id"):
        url = f"https://api.tomtom.com/traffic/data/stats/1/status/{job_id}"
        params = {"key": self.key}
        res = requests.get(url, params=params, timeout=5)
        return {"status_code": res.status_code, "info": "Job state check endpoint ready"}

    # 6. ROUTING API
    def calculate_route(self, start_lat, start_lon, end_lat, end_lon, mode="car"):
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_lat},{start_lon}:{end_lat},{end_lon}/json"
        params = {"key": self.key, "travelMode": mode, "traffic": "true"}
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            route_data = res.json()['routes'][0]
            summary = route_data['summary']
            points = []
            if 'legs' in route_data:
                for leg in route_data['legs']:
                    if 'points' in leg:
                        for p in leg['points']:
                            points.append([p['latitude'], p['longitude']])
            return {
                "time_sec": summary['travelTimeInSeconds'],
                "dist_meters": summary['lengthInMeters'],
                "coordinates": points
            }
        return None

    # 7. MATRIX ROUTING V2 API (Use Case: Nearest Resource & Patrol Allocations)
    def calculate_matrix_v2(self, origins, destinations):
        """
        Solves many-to-many travel time grids using Synchronous Matrix v2.
        Fails safely if asset points are empty.
        """
        url = "https://api.tomtom.com/routing/matrix/2"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.key}
        
        # Guard clause: Ensure inputs have active tracking points
        if not origins or not destinations:
            print("[Matrix Engine] Validation Skip: Origin or Destination tracking lists are empty.")
            return []
        
        # Build strict JSON request body conforming to TomTom v2 docs
        body = {
            "origins": [{"point": {"latitude": float(o[0]), "longitude": float(o[1])}} for o in origins],
            "destinations": [{"point": {"latitude": float(d[0]), "longitude": float(d[1])}} for d in destinations],
            "options": {
                "departAt": "any",          # Required for lightweight sync execution
                "traffic": "historical",    # Aligns with departAt='any' parameters
                "routeType": "fastest",     # Only routeType available for Sync v2
                "travelMode": "car"
            }
        }
        
        try:
            res = requests.post(url, params=params, headers=headers, json=body, timeout=10)
            
            if res.status_code == 200:
                payload = res.json()
                matrix_links = payload.get('data', [])
                
                # Check for individual cell errors in the flattened response array
                stats = payload.get('statistics', {})
                print(f"[Matrix Engine] Successes: {stats.get('successes', 0)} | Failures: {stats.get('failures', 0)}")
                
                return matrix_links
            else:
                print(f"[Matrix Structural Error]: Received Status Code {res.status_code}")
                print(f"Details: {res.text}")
                return []
                
        except Exception as e:
            print(f"[Matrix Engine Exception]: Failed processing grid data: {e}")
            return []

    # 8. WAYPOINT OPTIMIZATION API (Use Case: Multi-Incident Dispatch Sequence Heuristics)
    def optimize_waypoints(self, origin, waypoints, destination):
        url = "https://api.tomtom.com/routing/waypointoptimization/1"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.key}
        
        body = {
            "waypoints": [
                {"point": {"latitude": origin[0], "longitude": origin[1]}},
                *[{"point": {"latitude": wp[0], "longitude": wp[1]}} for wp in waypoints],
                {"point": {"latitude": destination[0], "longitude": destination[1]}}
            ],
            "options": {
                "travelMode": "car",
                "traffic": "historical",  # <-- Explicitly downcast to historical
                "departAt": "any"         # <-- Set context to any to avoid traffic mismatches
            }
        }
        
        res = requests.post(url, params=params, headers=headers, json=body, timeout=10)
        if res.status_code == 200:
            return res.json().get('optimizedOrder', []) # Note: API returns 'optimizedOrder'
        
        print(f"[Waypoint Optimization Structural Error]: {res.text}")
        return None


# ==============================================================================
# DISPATCH & ROUTING FLOW HARNESS
# ==============================================================================
if __name__ == "__main__":
    engine = TomTomSuite()
    print("🚀 Initializing Pipeline Test Suite...\n" + "="*50)

    # Better Position Coordination: Richmond Road Arterial Node, Central Bengaluru
    # This point falls directly on high-volume lanes, ensuring successful live flow index queries.
    target_lat, target_lon = 12.9700, 77.6100 
    print(f"[Setting Graph Check Point] -> Lat: {target_lat}, Lon: {target_lon}")

    # [1] Geocode Check
    geo = engine.geocode_address("Richmond Road, Bengaluru")
    print(f"[1] Geocode: {geo}")
    
    # [2] Reverse Geocode Verification
    address = engine.reverse_geocode(target_lat, target_lon)
    print(f"[2] Reverse Geocode: {address}")
    
    # [3] Search API: Allocating Nearest Emergency Support Centers
    allocations = engine.search_poi("Police Station", target_lat, target_lon)
    print(f"[3] Allocation Search Results: {allocations}")
    
    # [4] Live Traffic Stream Query on Core Arterial
    traffic = engine.get_live_traffic_flow(target_lat, target_lon)
    print(f"[4] Live Traffic Segment Metrics: {traffic}")

    # [6] Basic Routing Overhead Calc
    route = engine.calculate_route(12.9700, 77.6100, 12.9218, 77.6451)
    print(f"[6] Route Baseline (Richmond Road -> Agara Junction): {route}")

    # ==============================================================================
    # MATRIX ALLOCATION PIPELINE RUNNER (Sample Test In Harness)
    # ==============================================================================
    print("\n[7] Simulating Matrix Allocation Engine...")
    
    # Mocking active locations (e.g., Police Station coordinates mapped to Target Barricades)
    police_stations = [
        [12.9683, 77.6133],  # Ashok Nagar
        [12.9716, 77.5946]   # MG Road
    ]
    barricades = [
        [12.9218, 77.6451],  # Agara Junction
        [12.9591, 77.6507]   # Intermediate Checkpoint
    ]
    
    matrix_output = engine.calculate_matrix_v2(police_stations, barricades)
    print(f" -> Matrix complete. Processed {len(matrix_output)} structural asset connections successfully.")
    
    if matrix_output:
        # Example processing loop to read your computed links
        for cell in matrix_output[:2]:
            if 'routeSummary' in cell:
                summary = cell['routeSummary']
                print(f"    * Link O:{cell['originIndex']}->D:{cell['destinationIndex']} | Time: {round(summary['travelTimeInSeconds']/60, 1)} mins | Dist: {round(summary['lengthInMeters']/1000, 2)} km")
    # [8] Waypoint Optimization: Multi-Incident Response & Resource Sequencing (TSP solver)
    print("\n[8] Simulating Waypoint Dispatch Sequence Heuristics...")
    depot_start = [12.9700, 77.6100]       # Control Command Hub
    incident_scenes = [[12.9352, 77.6244], [12.9591, 77.6507]]  # Active blockages to clear
    dump_site_end = [12.9218, 77.6451]     # Ending Yard Depot
    
    optimized_order = engine.optimize_waypoints(depot_start, incident_scenes, dump_site_end)
    print(f" -> Optimized Sequence Waypoint Indices: {optimized_order}")