import os
import json
import datetime
import processing
from qgis.core import (QgsProject, QgsFeature, QgsGeometry, 
                       QgsSpatialIndex, QgsField, QgsVectorLayer,
                       QgsVectorFileWriter)
from PyQt5.QtCore import QVariant

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
raw_osm_layer_name = 'osm_streets_berlin_cut_utm' 
trajectories_layer_name = 'traj_mehringdamm_orig_UTM_AOIcut' 
datetime_field = 'start_timestamp' 
unique_id_field = 'trip_id' 

# Parameters
search_radius = 5 
split_length = 2 
similarity_tolerance = 3 
scaling_factor = 5 

# Emission Factors (passenger car estimates in g/km)
EF_NOX = 0.180  # Updated from Delphi's 0.352
EF_PM10 = 0.020 # Updated from Delphi's 0.004 to include non-exhaust wear
V_RATIO = 0.5   # Ratio for NO2/NOx split and PM2.5/PM10 split

# Output Paths
output_dir = 'D:/enviprojects/Berlin_Mehringdamm_Base/TrajectoryOutputData/'
output_csv_path = os.path.join(output_dir, 'output_counts.csv')
output_gpkg_path = os.path.join(output_dir, 'output_counts.gpkg')
output_json_path = os.path.join(output_dir, 'projectdatabase.json') # New JSON output

osm_layer = QgsProject.instance().mapLayersByName(raw_osm_layer_name)[0]
traj_layer = QgsProject.instance().mapLayersByName(trajectories_layer_name)[0]
crs_str = osm_layer.crs().toWkt()

# ==========================================
# 2. PREPROCESS OSM (Filter & Split)
# ==========================================
print("Step 1: Filtering and splitting OSM lines...")
expression = "\"fclass\" IN ('primary', 'primary_link', 'residential', 'secondary', 'secondary_link')"
filtered_osm = processing.run("native:extractbyexpression", {
    'INPUT': osm_layer, 'EXPRESSION': expression, 'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

split_osm = processing.run("native:splitlinesbylength", {
    'INPUT': filtered_osm, 'LENGTH': split_length, 'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

# ==========================================
# 3. SETUP MEMORY LAYER & BUILD SPATIAL INDEX
# ==========================================
print("Step 2: Preparing trajectory data...")
memory_layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Temp_Counts", "memory")
provider = memory_layer.dataProvider()
provider.addAttributes([QgsField("tempID", QVariant.Int)])
for h in range(24): provider.addAttributes([QgsField(f"h_{h:02d}", QVariant.Int)])
memory_layer.updateFields()

new_features = []
for i, feat in enumerate(split_osm.getFeatures()):
    new_feat = QgsFeature(memory_layer.fields())
    new_feat.setGeometry(feat.geometry())
    new_feat.setAttribute("tempID", i)
    for h in range(24): new_feat.setAttribute(f"h_{h:02d}", 0)
    new_features.append(new_feat)
provider.addFeatures(new_features)

traj_index = QgsSpatialIndex(traj_layer.getFeatures())
traj_data = {}
for feat in traj_layer.getFeatures():
    start_hour = int(feat[datetime_field] // 3600)
    if 0 <= start_hour <= 23:
        traj_data[feat.id()] = {'geom': feat.geometry(), 'hour': start_hour, 'group_id': feat[unique_id_field]}

# ==========================================
# 4. COUNT UNIQUE VEHICLES
# ==========================================
print("Step 3: Counting unique trajectories per segment...")
update_map = {}
h_indices = {h: memory_layer.fields().indexOf(f"h_{h:02d}") for h in range(24)}

for seg_feat in memory_layer.getFeatures():
    seg_geom = seg_feat.geometry()
    search_rect = seg_geom.boundingBox()
    search_rect.grow(search_radius)
    candidate_ids = traj_index.intersects(search_rect)
    
    unique_counts = {h: set() for h in range(24)}
    for c_id in candidate_ids:
        if c_id not in traj_data: continue
        if seg_geom.distance(traj_data[c_id]['geom']) <= search_radius:
            unique_counts[traj_data[c_id]['hour']].add(traj_data[c_id]['group_id'])
            
    attr_update = {}
    for h in range(24): attr_update[h_indices[h]] = len(unique_counts[h])
    update_map[seg_feat.id()] = attr_update

provider.changeAttributeValues(update_map)

# ==========================================
# 5. MERGE SIMILAR ADJACENT SEGMENTS
# ==========================================
print("Step 4: Merging similar adjacent segments...")
mem_index = QgsSpatialIndex(memory_layer.getFeatures())
mem_features = {f.id(): f for f in memory_layer.getFeatures()}

visited = set()
groups = []

for f_id, f in mem_features.items():
    if f_id in visited: continue
    current_group, queue = [f_id], [f_id]
    visited.add(f_id)
    
    while queue:
        curr_id = queue.pop(0)
        curr_feat = mem_features[curr_id]
        
        bbox = curr_feat.geometry().boundingBox()
        bbox.grow(0.01) 
        candidates = mem_index.intersects(bbox)
        
        for cand_id in candidates:
            if cand_id in visited: continue
            cand_feat = mem_features[cand_id]
            
            if curr_feat.geometry().distance(cand_feat.geometry()) < 0.01:
                seed_feat = mem_features[current_group[0]]
                is_similar = True
                for h in range(24):
                    if abs(seed_feat[f"h_{h:02d}"] - cand_feat[f"h_{h:02d}"]) > similarity_tolerance:
                        is_similar = False
                        break
                if is_similar:
                    visited.add(cand_id)
                    queue.append(cand_id)
                    current_group.append(cand_id)
    groups.append(current_group)

# ==========================================
# 6. CREATE FINAL LAYER AND SCALE
# ==========================================
print("Step 5: Applying scaling factor...")
final_layer = QgsVectorLayer(f"MultiLineString?crs={crs_str}", "Final_Merged_Counts", "memory")
final_prov = final_layer.dataProvider()
final_prov.addAttributes([QgsField("enviID", QVariant.String, len=6)])
for h in range(24): final_prov.addAttributes([QgsField(f"h_{h:02d}", QVariant.Int)])
final_layer.updateFields()

final_feats = []
envi_id_counter = 1

for grp in groups:
    geoms = [mem_features[fid].geometry() for fid in grp]
    merged_geom = QgsGeometry.unaryUnion(geoms)
    
    new_feat = QgsFeature(final_layer.fields())
    new_feat.setGeometry(merged_geom)
    new_feat.setAttribute("enviID", f"{envi_id_counter:06d}")
    
    for h in range(24):
        avg_raw_count = sum([mem_features[fid][f"h_{h:02d}"] for fid in grp]) / len(grp)
        scaled_count = round(avg_raw_count * scaling_factor)
        new_feat.setAttribute(f"h_{h:02d}", scaled_count)
        
    final_feats.append(new_feat)
    envi_id_counter += 1

final_prov.addFeatures(final_feats)

# ==========================================
# 7. GENERATE ENVI-MET JSON DATABASE
# ==========================================
print("Step 6: Generating ENVI-met JSON Database...")

ef_no = EF_NOX * (1 - V_RATIO)
ef_no2 = EF_NOX * V_RATIO
ef_pm25 = EF_PM10 * V_RATIO

json_db = {
    "envimetDatafile": {
        "header": {
            "fileType": "databaseJSON",
            "version": 1,
            "revisionDate": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "remark": "Auto-generated by QGIS Trajectory Script",
            "description": "Traffic Emission Line Sources"
        },
        "soils": [], "soilProfiles": [], "materials": [], "walls": [], 
        "singleFaces": [], "waterSources": [], "emitters": [], 
        "simplePlants": [], "greening": [], "plants3d": []
    }
}

for feat in final_layer.getFeatures():
    envi_id = feat["enviID"]
    em_usr, em_no, em_no2, em_o3, em_pm10, em_pm25 = [], [], [], [], [], []
    
    for h in range(24):
        q = feat[f"h_{h:02d}"]
        
        # Calculate emissions in µg/(m*s) for line sources
        em_usr.append(0.0)
        em_o3.append(0.0)
        em_no.append(float((q * ef_no) / 3.6))
        em_no2.append(float((q * ef_no2) / 3.6))
        em_pm10.append(float((q * EF_PM10) / 3.6))
        em_pm25.append(float((q * ef_pm25) / 3.6))

    emitter = {
        "id": envi_id,
        "desc": f"Traffic Line {envi_id}",
        "col": "81E908",
        "grp": "Emitters",
        "height": 0.5,
        "geom": "line", # Changed from "point" to "line"
        "emissionUsr": em_usr,
        "emissionNO": em_no,
        "emissionNO2": em_no2,
        "emissionO3": em_o3,
        "emissionPM10": em_pm10,
        "emissionPM25": em_pm25,
        "cost": 0,
        "remark": "Generated Line Source"
    }
    json_db["envimetDatafile"]["emitters"].append(emitter)

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(json_db, f, indent=4)
print(f"SUCCESS! Database created: {output_json_path}")

# ==========================================
# 8. EXPORT TO GEOPACKAGE AND CSV
# ==========================================
print("Step 7: Saving geometric results to GPKG and CSV...")
save_opts = QgsVectorFileWriter.SaveVectorOptions()
save_opts.driverName = "GPKG"
save_opts.layerName = "segment_counts"
if QgsVectorFileWriter.writeAsVectorFormatV3(final_layer, output_gpkg_path, QgsProject.instance().transformContext(), save_opts)[0] == QgsVectorFileWriter.NoError:
    QgsProject.instance().addMapLayer(QgsVectorLayer(f"{output_gpkg_path}|layername=segment_counts", "Final Merged Counts", "ogr"))

save_opts.driverName = "CSV"
save_opts.layerOptions = ["SEPARATOR=SEMICOLON"]
QgsVectorFileWriter.writeAsVectorFormatV3(final_layer, output_csv_path, QgsProject.instance().transformContext(), save_opts)

print("--- Workflow Complete ---")