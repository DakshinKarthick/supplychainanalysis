import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "..", "csv_files")

master_file = os.path.join(CSV_DIR, "HMB Details_Master Data.csv")
summary_file = os.path.join(CSV_DIR, "HMB Details_Summary.csv")
output_file = os.path.join(CSV_DIR, "Unified_Route_Data.csv")

def parse_lat_lon(coord_str):
    if not coord_str or not coord_str.strip(): return None
    try:
        parts = coord_str.strip().split(",")
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except:
        pass
    return None

summary = {}
with open(summary_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row or not row[0].strip() or row[0].strip() == "Grand Total": continue
        route_code = row[2].strip() if len(row) > 2 else ""
        if not route_code.startswith("M"): continue
        summary[route_code] = {
            "capacity": int(float(row[8])) if len(row) > 8 and row[8].strip() else 0,
            "milk_qty": float(row[9]) if len(row) > 9 and row[9].strip() else 0.0,
            "per_day_km": float(row[10]) if len(row) > 10 and row[10].strip() else 0.0,
            "uti_percent": float(row[15]) if len(row) > 15 and row[15].strip() else 0.0,
            "transporter": row[5].strip() if len(row) > 5 else "",
            "vehicle_type": row[7].strip() if len(row) > 7 else ""
        }

rows_to_write = []
with open(master_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        if not row or len(row) < 11: continue
        plant = row[0].strip()
        cc_coords = row[1].strip()
        route_name = row[2].strip()
        center_code = row[3].strip()
        center_name = row[4].strip()
        sequence_str = row[5].strip()
        route_code = row[6].strip()
        distance_str = row[7].strip()
        hmb_coords = row[10].strip()

        if not plant or plant != "1142": continue
        if not route_code.startswith("M"): continue
        if center_name.lower() in ("cc", "total", ""): continue

        coords = parse_lat_lon(hmb_coords)
        if not coords: continue

        info = summary.get(route_code, {})
        
        flat_row = [
            plant,
            "Uthangarai CC",
            cc_coords,
            route_code,
            route_name,
            info.get("capacity", 0),
            info.get("milk_qty", 0.0),
            info.get("per_day_km", 0.0),
            info.get("uti_percent", 0.0),
            info.get("transporter", ""),
            info.get("vehicle_type", ""),
            sequence_str,
            center_code,
            center_name,
            hmb_coords,
            distance_str
        ]
        rows_to_write.append(flat_row)

header = [
    "CC Code", "CC Name", "CC Lat & Lon", 
    "Route Code", "Route Name", "Route Capacity", "Route Milk Qty", 
    "Per Day KM", "UTI Percent", "Transporter Name", "Vehicle Type", 
    "HMB Sequence", "HMB SAP Code", "HMB Name", "HMB Lat & Lon", "HMB Distance KM"
]

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows_to_write)

print("Created Unified_Route_Data.csv successfully!")
