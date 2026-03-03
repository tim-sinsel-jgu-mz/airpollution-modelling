import os
import processing
from qgis.core import (QgsProject, QgsFeature, QgsGeometry, 
                       QgsSpatialIndex, QgsField, QgsVectorLayer,
                       QgsVectorFileWriter)
from PyQt5.QtCore import QVariant

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
# Input Layers
raw_osm_layer_name = 'osm_streets_berlin_cut_utm' # Point this to your RAW OSM layer now
trajectories_layer_name = 'traj_mehringdamm_orig_UTM_AOIcut' 
datetime_field = 'start_timestamp' 
unique_id_field = 'trip_id' 

# Parameters
search_radius = 5 # in meters (ensure CRS is metric!)
split_length = 2 # Max length to split OSM lines (in meters)
similarity_tolerance = 3 # Max variance in raw trajectory counts to allow merging
scaling_factor = 5 # Multiply final counts by this to simulate 100% traffic

# Output Paths
output_csv_path = 'D:/enviprojects/Berlin_Mehringdamm_Base/TrajectoryOutputData/output_counts.csv' 
output_gpkg_path = 'D:/enviprojects/Berlin_Mehringdamm_Base/TrajectoryOutputData/output_counts.gpkg' 

# Get layers
osm_layer = QgsProject.instance().mapLayersByName(raw_osm_layer_name)[0]
traj_layer = QgsProject.instance().mapLayersByName(trajectories_layer_name)[0]
crs_str = osm_layer.crs().toWkt()

# ==========================================
# 2. PREPROCESS OSM (Filter & Split)
# ==========================================
print("Step 1: Filtering OSM classes...")
expression = "\"fclass\" IN ('primary', 'primary_link', 'residential', 'secondary', 'secondary_link')"
filtered_osm = processing.run("native:extractbyexpression", {
    'INPUT': osm_layer,
    'EXPRESSION': expression,
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

print(f"Step 2: Splitting lines to {split_length}m maximum length...")
split_osm = processing.run("native:splitlinesbylength", {
    'INPUT': filtered_osm,
    'LENGTH': split_length,
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

# ==========================================
# 3. SETUP MEMORY LAYER FOR COUNTING
# ==========================================
print("Step 3: Setting up memory layer for counting...")
memory_layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Temp_Counts", "memory")
provider = memory_layer.dataProvider()

# Only add tempID and hour fields for simplicity during calculation
provider.addAttributes([QgsField("tempID", QVariant.Int)])
for h in range(24):
    provider.addAttributes([QgsField(f"h_{h:02d}", QVariant.Int)])
memory_layer.updateFields()

new_features = []
for i, feat in enumerate(split_osm.getFeatures()):
    new_feat = QgsFeature(memory_layer.fields())
    new_feat.setGeometry(feat.geometry())
    new_feat.setAttribute("tempID", i)
    for h in range(24):
        new_feat.setAttribute(f"h_{h:02d}", 0)
    new_features.append(new_feat)
provider.addFeatures(new_features)

# ==========================================
# 4. BUILD SPATIAL INDEX & TRAJECTORY DATA
# ==========================================
print("Step 4: Building spatial index for trajectories...")
traj_index = QgsSpatialIndex(traj_layer.getFeatures())

traj_data = {}
for feat in traj_layer.getFeatures():
    start_hour = int(feat[datetime_field] // 3600)
    if 0 <= start_hour <= 23:
        traj_data[feat.id()] = {
            'geom': feat.geometry(),
            'hour': start_hour,
            'group_id': feat[unique_id_field]
        }
print("Spatial index built!")

# ==========================================
# 5. COUNT UNIQUE VEHICLES
# ==========================================
print("Step 5: Processing segments and counting unique vehicles...")
update_map = {}
h_indices = {h: memory_layer.fields().indexOf(f"h_{h:02d}") for h in range(24)}

for seg_feat in memory_layer.getFeatures():
    seg_geom = seg_feat.geometry()
    
    search_rect = seg_geom.boundingBox()
    search_rect.grow(search_radius)
    candidate_ids = traj_index.intersects(search_rect)
    
    unique_counts = {h: set() for h in range(24)}
    
    for c_id in candidate_ids:
        if c_id not in traj_data: 
            continue
            
        t_geom = traj_data[c_id]['geom']
        t_hour = traj_data[c_id]['hour']
        t_group_id = traj_data[c_id]['group_id']
        
        if seg_geom.distance(t_geom) <= search_radius:
            unique_counts[t_hour].add(t_group_id)
            
    attr_update = {}
    for h in range(24):
        attr_update[h_indices[h]] = len(unique_counts[h])
        
    update_map[seg_feat.id()] = attr_update

provider.changeAttributeValues(update_map)

# ==========================================
# 6. MERGE SIMILAR ADJACENT SEGMENTS
# ==========================================
print(f"Step 6: Merging touching segments with <= {similarity_tolerance} count variance...")
mem_index = QgsSpatialIndex(memory_layer.getFeatures())
mem_features = {f.id(): f for f in memory_layer.getFeatures()}

visited = set()
groups = []

for f_id, f in mem_features.items():
    if f_id in visited:
        continue
        
    current_group = [f_id]
    queue = [f_id]
    visited.add(f_id)
    
    while queue:
        curr_id = queue.pop(0)
        curr_feat = mem_features[curr_id]
        
        # Find neighbors within a tiny buffer to catch touching lines
        bbox = curr_feat.geometry().boundingBox()
        bbox.grow(0.01) 
        candidates = mem_index.intersects(bbox)
        
        for cand_id in candidates:
            if cand_id in visited:
                continue
                
            cand_feat = mem_features[cand_id]
            
            # Check if lines actually touch
            if curr_feat.geometry().distance(cand_feat.geometry()) < 0.01:
                
                # Check similarity against the SEED feature of the group to prevent unbounded drifting
                seed_feat = mem_features[current_group[0]]
                is_similar = True
                
                for h in range(24):
                    seed_count = seed_feat[f"h_{h:02d}"]
                    cand_count = cand_feat[f"h_{h:02d}"]
                    if abs(seed_count - cand_count) > similarity_tolerance:
                        is_similar = False
                        break
                        
                if is_similar:
                    visited.add(cand_id)
                    queue.append(cand_id)
                    current_group.append(cand_id)
                    
    groups.append(current_group)

# ==========================================
# 7. CREATE FINAL LAYER AND APPLY SCALING
# ==========================================
print(f"Step 7: Applying {scaling_factor}x scaling and generating final layer...")
final_layer = QgsVectorLayer(f"MultiLineString?crs={crs_str}", "Final_Merged_Counts", "memory")
final_prov = final_layer.dataProvider()

# --- CHANGED: enviID is now a 6-character String ---
final_prov.addAttributes([QgsField("enviID", QVariant.String, len=6)])
for h in range(24):
    final_prov.addAttributes([QgsField(f"h_{h:02d}", QVariant.Int)])
final_layer.updateFields()

final_feats = []
envi_id_counter = 1

for grp in groups:
    # Merge geometries of the group
    geoms = [mem_features[fid].geometry() for fid in grp]
    merged_geom = QgsGeometry.unaryUnion(geoms)
    
    new_feat = QgsFeature(final_layer.fields())
    new_feat.setGeometry(merged_geom)
    
    # --- CHANGED: Apply zero-padding to create a 6-digit string like "000001" ---
    new_feat.setAttribute("enviID", f"{envi_id_counter:06d}")
    
    for h in range(24):
        # Calculate the average raw count for this group, then scale it
        avg_raw_count = sum([mem_features[fid][f"h_{h:02d}"] for fid in grp]) / len(grp)
        scaled_count = round(avg_raw_count * scaling_factor)
        new_feat.setAttribute(f"h_{h:02d}", scaled_count)
        
    final_feats.append(new_feat)
    envi_id_counter += 1

final_prov.addFeatures(final_feats)
print(f"   > Reduced {len(mem_features)} raw segments into {len(final_feats)} merged groups.")

# ==========================================
# 8. EXPORT TO GEOPACKAGE AND CSV
# ==========================================
print("Step 8: Saving results to GeoPackage and CSV...")

# Save GPKG
save_options_gpkg = QgsVectorFileWriter.SaveVectorOptions()
save_options_gpkg.driverName = "GPKG"
save_options_gpkg.layerName = "segment_counts"

res_gpkg = QgsVectorFileWriter.writeAsVectorFormatV3(
    final_layer,
    output_gpkg_path,
    QgsProject.instance().transformContext(),
    save_options_gpkg
)

if res_gpkg[0] == QgsVectorFileWriter.NoError:
    print(f"SUCCESS! Created GeoPackage: {output_gpkg_path}")
    vlayer = QgsVectorLayer(f"{output_gpkg_path}|layername=segment_counts", "Final Merged Counts", "ogr")
    if vlayer.isValid():
        QgsProject.instance().addMapLayer(vlayer)
else:
    print(f"Error saving GeoPackage: {res_gpkg}")

# Save CSV
save_options_csv = QgsVectorFileWriter.SaveVectorOptions()
save_options_csv.driverName = "CSV"
# --- CHANGED: Inject OGR parameter to force semicolon separator ---
save_options_csv.layerOptions = ["SEPARATOR=SEMICOLON"]

res_csv = QgsVectorFileWriter.writeAsVectorFormatV3(
    final_layer,
    output_csv_path,
    QgsProject.instance().transformContext(),
    save_options_csv
)

if res_csv[0] == QgsVectorFileWriter.NoError:
    print(f"SUCCESS! Created CSV: {output_csv_path}")
else:
    print(f"Error saving CSV: {res_csv}")

print("--- Workflow Complete ---")