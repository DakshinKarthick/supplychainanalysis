import csv
import random
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "..", "csv_files")
output_file = os.path.join(CSV_DIR, "Mock_Coimbatore_Route_Data.csv")

CC_NAME = "Coimbatore CC"
CC_CODE = "9999"
CC_LAT = 11.016844
CC_LON = 76.955833

# Generate 4 random routes
routes = [
    {"code": "C1001", "name": "North Route"},
    {"code": "C1002", "name": "East Route"},
    {"code": "C1003", "name": "South Route"},
    {"code": "C1004", "name": "West Route"},
]

rows = []
sap_counter = 10000

for r in routes:
    capacity = random.choice([2000, 2200, 2500, 3100])
    num_hmbs = random.randint(5, 10)
    milk_qty = random.randint(800, capacity - 200)
    uti = round(milk_qty / capacity * 100, 2)
    per_day_km = round(random.uniform(80, 150), 1)
    
    # define an angle for this route to cluster its HMBs
    base_angle = random.uniform(0, 2 * math.pi)
    
    for seq in range(1, num_hmbs + 1):
        # randomly place HMB around the base angle, distance 5-40 km
        angle = base_angle + random.uniform(-0.5, 0.5)
        dist_from_cc = random.uniform(5, 40)
        
        # approximate 1 degree = ~111 km
        lat_offset = (dist_from_cc * math.cos(angle)) / 111.0
        lon_offset = (dist_from_cc * math.sin(angle)) / (111.0 * math.cos(math.radians(CC_LAT)))
        
        hmb_lat = CC_LAT + lat_offset
        hmb_lon = CC_LON + lon_offset
        
        hmb_name = f"Village {sap_counter}"
        hmb_dist = round(random.uniform(2, 15), 1)
        
        row = [
            CC_CODE, CC_NAME, f"{CC_LAT:.6f}, {CC_LON:.6f}",
            r["code"], r["name"], capacity, milk_qty, per_day_km, uti,
            "Mock Transporter", "Truck",
            seq, str(sap_counter), hmb_name, f"{hmb_lat:.6f}, {hmb_lon:.6f}", hmb_dist
        ]
        sap_counter += 1
        rows.append(row)

header = [
    "CC Code", "CC Name", "CC Lat & Lon", 
    "Route Code", "Route Name", "Route Capacity", "Route Milk Qty", 
    "Per Day KM", "UTI Percent", "Transporter Name", "Vehicle Type", 
    "HMB Sequence", "HMB SAP Code", "HMB Name", "HMB Lat & Lon", "HMB Distance KM"
]

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Generated {output_file}")
