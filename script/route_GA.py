import csv, json, math, os, socket, time, urllib.error, urllib.request, random, sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable

# Configuration
ROAD_FACTOR = 1.3
MAX_ROUTE_HOURS = 4.0
AVG_SPEED_KMH = 40.0
AVG_STOP_TIME_HOURS = 10 / 60
CC_LAT, CC_LON = 12.308573, 78.535901
OSRM_BASE_URL = "http://router.project-osrm.org"
OSRM_MAX_RETRIES, OSRM_RETRY_BACKOFF = 3, 2
DISTANCE_MODE = "osrm"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "..", "csv_files")

@dataclass
class HMB:
    sap_code: str; name: str; lat: float; lon: float; sequence: int; distance_km: float

@dataclass
class Route:
    code: str; name: str
    hmbs: list = field(default_factory=list)
    capacity: int = 0; current_milk_qty: float = 0.0

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def road_distance(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2) * ROAD_FACTOR

def _osrm_request(url, timeout=30):
    for attempt in range(1, OSRM_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as res:
                return json.loads(res.read().decode())
        except Exception:
            if attempt < OSRM_MAX_RETRIES: time.sleep(OSRM_RETRY_BACKOFF * (2 ** (attempt - 1)))
    return {"distances": []}

def osrm_table_query(coords):
    if len(coords) < 2: return [[0.0]]
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coord_str}?annotations=distance"
    data = _osrm_request(url, timeout=45)
    if "distances" in data and data["distances"]:
        return [[d / 1000.0 if d is not None else 0.0 for d in row] for row in data["distances"]]
    return [[0.0]*len(coords) for _ in coords]

def build_distance_matrix(coords, mode="haversine"):
    if len(coords) < 2: return [[0.0]]
    if mode == "osrm": return osrm_table_query(coords)
    return [[0.0 if i==j else road_distance(c1[0],c1[1], c2[0],c2[1]) for j, c2 in enumerate(coords)] for i, c1 in enumerate(coords)]

def route_distance_from_matrix(order, matrix):
    if not order: return 0.0
    return matrix[0][order[0]] + sum(matrix[order[i]][order[i+1]] for i in range(len(order)-1)) + matrix[order[-1]][0]

def load_route_summary():
    routes_info = {}
    with open(os.path.join(CSV_DIR, "HMB Details_Summary.csv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if not row or not row[0].strip() or row[0].strip() == "Grand Total": continue
            code = row[2].strip() if len(row) > 2 else ""
            if code.startswith("M"):
                routes_info[code] = {
                    "route_name": row[3].strip() if len(row)>3 else "",
                    "capacity": int(float(row[8])) if len(row)>8 and row[8].strip() else 0,
                    "milk_qty": float(row[9]) if len(row)>9 and row[9].strip() else 0.0,
                }
    return routes_info

def load_route_data():
    summary = load_route_summary()
    route_hmbs = {}
    with open(os.path.join(CSV_DIR, "HMB Details_Master Data.csv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if len(row) < 11: continue
            plant, r_name, c_code, c_name, seq_str, r_code, dist_str, coord_str = (row[i].strip() for i in (0,2,3,4,5,6,7,10))
            if plant != "1142" or not r_code.startswith("M") or c_name.lower() in ("cc", "total", ""): continue
            try: coords = (float(coord_str.split(",")[0]), float(coord_str.split(",")[1]))
            except: continue
            try: seq, dist = int(float(seq_str)) if seq_str else 0, float(dist_str) if dist_str else 0.0
            except: continue
            route_hmbs.setdefault(r_code, {"name": r_name, "hmbs": []})["hmbs"].append((seq, HMB(c_code, c_name, coords[0], coords[1], seq, dist)))

    return sorted([Route(code, d["name"] or summary.get(code, {}).get("route_name", ""),
                         [h for _, h in sorted(d["hmbs"], key=lambda x: x[0])],
                         summary.get(code, {}).get("capacity", 0),
                         summary.get(code, {}).get("milk_qty", 0.0))
                   for code, d in route_hmbs.items()], key=lambda r: r.code)

def estimate_route_time(total_km, num_stops):
    return (total_km / AVG_SPEED_KMH) + (num_stops * AVG_STOP_TIME_HOURS)

# --- 3 Mutation Variants ---
def reassign_mutation(g, start, end, num_routes, rate=0.2):
    if random.random() < rate:
        g[random.randint(start, end - 1)] = random.randint(0, num_routes - 1)
    return g

def swap_mutation(g, start, end, rate=0.15):
    if random.random() < rate and end - start >= 2:
        i, j = random.sample(range(start, end), 2)
        g[i], g[j] = g[j], g[i]
    return g

def inversion_mutation(g, start, end, rate=0.05):
    if random.random() < rate and end - start >= 2:
        i, j = sorted(random.sample(range(start, end), 2))
        g[i:j+1] = reversed(g[i:j+1])
    return g

def uniform_crossover(p1, p2):
    return [g1 if random.random() < 0.5 else g2 for g1, g2 in zip(p1, p2)], \
           [g2 if random.random() < 0.5 else g1 for g1, g2 in zip(p1, p2)]

def full_route_fitness(genome, all_hmbs, num_routes, capacities, milk_qtys, matrix):
    n = len(all_hmbs)
    assigns, prios = genome[:n], genome[n:]
    route_groups = [[] for _ in range(num_routes)]
    for i in range(n): route_groups[assigns[i] % num_routes].append((prios[i], i))

    tot_dist, time_pen, cap_pen = 0.0, 0.0, 0.0
    for r_idx, grp in enumerate(route_groups):
        if not grp: continue
        grp.sort(key=lambda x: x[0])
        dist = route_distance_from_matrix([idx + 1 for _, idx in grp], matrix)
        tot_dist += dist
        t = estimate_route_time(dist, len(grp))
        if t > MAX_ROUTE_HOURS:
            time_pen += (t - MAX_ROUTE_HOURS) * 100
        cap = capacities[r_idx] if capacities[r_idx] > 0 else 1
        if milk_qtys[r_idx] > cap:
            cap_pen += ((milk_qtys[r_idx] - cap) / cap) * 500

    empty_pen = sum(50 for g in route_groups if not g)
    return 0.4*tot_dist + 0.2*time_pen + 0.3*cap_pen + empty_pen


def ga_full_route_optimize(new_lat=None, new_lon=None, gens=500, pop_size=200, mode="haversine", json_output=False):
    """Run full GA route re-optimization. Returns dict with results."""
    routes = load_route_data()
    num_routes = len(routes)
    all_hmbs, orig_assigns = [], []
    route_hmb_indices = [[] for _ in range(num_routes)]

    for r_idx, route in enumerate(routes):
        for hmb in route.hmbs:
            all_hmbs.append((hmb.lat, hmb.lon, hmb.name))
            orig_assigns.append(r_idx)
            route_hmb_indices[r_idx].append(len(all_hmbs) - 1)

    has_new = new_lat is not None and new_lon is not None
    if has_new:
        all_hmbs.append((new_lat, new_lon, "NEW HMB"))
        orig_assigns.append(-1)

    n = len(all_hmbs)
    caps = [r.capacity for r in routes]
    milk = [r.current_milk_qty for r in routes]
    matrix = build_distance_matrix([(CC_LAT, CC_LON)] + [(h[0], h[1]) for h in all_hmbs], mode)

    # Original total distance
    original_total_km = 0.0
    for r_idx in range(num_routes):
        order = [idx + 1 for idx in route_hmb_indices[r_idx]]
        original_total_km += route_distance_from_matrix(order, matrix)

    if not json_output:
        print(f"Total HMBs: {n} across {num_routes} routes | Mode: {mode}")
        print(f"Original total distance: {original_total_km:.1f} KM")

    # Initialize population
    pop = []
    for _ in range(pop_size):
        a = orig_assigns[:]
        for i in range(n):
            if random.random() < 0.2: a[i] = random.randint(0, num_routes - 1)
        if has_new: a[-1] = random.randint(0, num_routes - 1)
        pop.append(a + [random.randint(0, 999) for _ in range(n)])

    best_g, best_f = None, float("inf")
    for gen in range(gens):
        fits = [full_route_fitness(g, all_hmbs, num_routes, caps, milk, matrix) for g in pop]
        best_idx = min(range(len(fits)), key=lambda i: fits[i])
        if fits[best_idx] < best_f: best_f, best_g = fits[best_idx], pop[best_idx][:]

        new_pop = [ind[:] for ind, _ in sorted(zip(pop, fits), key=lambda x: x[1])[:2]]

        while len(new_pop) < pop_size:
            p1 = min(random.sample(list(zip(pop, fits)), 3), key=lambda x: x[1])[0]
            p2 = min(random.sample(list(zip(pop, fits)), 3), key=lambda x: x[1])[0]
            c1_a, c2_a = uniform_crossover(p1[:n], p2[:n])
            c1_p, c2_p = uniform_crossover(p1[n:], p2[n:])
            c1, c2 = c1_a + c1_p, c2_a + c2_p

            for child in (c1, c2):
                child = reassign_mutation(child, 0, n, num_routes, rate=0.2)
                child = swap_mutation(child, n, 2*n, rate=0.15)
                child = inversion_mutation(child, n, 2*n, rate=0.05)
                new_pop.append(child)

        pop = new_pop[:pop_size]
        if not json_output and gen % 50 == 0:
            print(f"Gen {gen} | Best Fitness: {best_f:.2f}")

    # Decode best solution
    assigns = best_g[:n]
    prios = best_g[n:]
    groups = [[] for _ in range(num_routes)]
    for i in range(n):
        r_idx = assigns[i] % num_routes
        groups[r_idx].append((prios[i], all_hmbs[i], orig_assigns[i], i))

    for grp in groups:
        grp.sort(key=lambda x: x[0])

    # Calculate optimized total
    optimized_total_km = 0.0
    result_routes = []
    for r_idx, grp in enumerate(groups):
        route = routes[r_idx]
        if not grp:
            result_routes.append({
                "route_code": route.code, "route_name": route.name,
                "hmb_count": 0, "distance_km": 0, "original_km": 0,
                "diff_km": 0, "est_time_h": 0, "time_ok": True, "cap_ok": True,
                "sequence": [], "hmbs": []
            })
            continue

        order = [idx + 1 for _, _, _, idx in grp]
        dist = route_distance_from_matrix(order, matrix)
        optimized_total_km += dist

        orig_order = [idx + 1 for idx in route_hmb_indices[r_idx]]
        orig_dist = route_distance_from_matrix(orig_order, matrix)
        est_time = estimate_route_time(dist, len(grp))

        hmbs_list = []
        seq_names = []
        for _, hmb_data, orig_route, _ in grp:
            from_route = None
            if orig_route != r_idx and orig_route != -1:
                from_route = routes[orig_route].code
            elif orig_route == -1:
                from_route = "NEW"
            hmbs_list.append({
                "name": hmb_data[2], "lat": hmb_data[0], "lng": hmb_data[1],
                "from_route": from_route
            })
            seq_names.append(hmb_data[2])

        result_routes.append({
            "route_code": route.code,
            "route_name": route.name,
            "hmb_count": len(grp),
            "distance_km": round(dist, 1),
            "original_km": round(orig_dist, 1),
            "diff_km": round(dist - orig_dist, 1),
            "est_time_h": round(est_time, 1),
            "time_ok": est_time <= MAX_ROUTE_HOURS,
            "cap_ok": route.current_milk_qty <= route.capacity,
            "capacity": route.capacity,
            "milk_qty": route.current_milk_qty,
            "sequence": seq_names,
            "hmbs": hmbs_list
        })

    saving = original_total_km - optimized_total_km
    result = {
        "type": "full_optimization",
        "original_km": round(original_total_km, 1),
        "optimized_km": round(optimized_total_km, 1),
        "saving_km": round(saving, 1),
        "saving_pct": round((saving / original_total_km * 100) if original_total_km > 0 else 0, 1),
        "routes": result_routes,
        "cc": {"lat": CC_LAT, "lng": CC_LON},
        "new_hmb": {"lat": new_lat, "lng": new_lon} if has_new else None,
    }

    if json_output:
        print(json.dumps(result))
    else:
        print(f"\nOptimized Routes:")
        for r in result_routes:
            if r["hmb_count"] == 0: continue
            print(f"Route {r['route_code']}: CC -> {' -> '.join(r['sequence'])} -> CC")
        print(f"\nOriginal: {result['original_km']} KM | Optimized: {result['optimized_km']} KM | Saved: {result['saving_km']} KM ({result['saving_pct']}%)")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GA Full Route Re-Optimization")
    parser.add_argument("--lat", type=float, default=None, help="Latitude of new HMB (optional)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude of new HMB (optional)")
    parser.add_argument("--mode", choices=["haversine", "osrm"], default="osrm", help="Distance mode")
    parser.add_argument("--gens", type=int, default=500, help="GA generations")
    parser.add_argument("--pop", type=int, default=200, help="Population size")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    ga_full_route_optimize(args.lat, args.lon, gens=args.gens, pop_size=args.pop,
                           mode=args.mode, json_output=args.json)
