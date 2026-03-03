import os
from qgis.core import (QgsProject, QgsFeature, QgsGeometry, 
                       QgsSpatialIndex, QgsField, QgsVectorLayer,
                       QgsVectorFileWriter)
from PyQt5.QtCore import QVariant

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
segments_layer_name = 'OSM_streets_Berlin_cut_onlyRelevant_1mMaxLengths' # Replace with your layer name
trajectories_layer_name = 'traj_mehringdamm_orig_UTM_AOIcut' # Replace with your layer name
datetime_field = 'start_timestamp' # Replace with your datetime column name 
unique_id_field = 'trip_id' 

search_radius = 5 # in meters (ensure CRS is metric!)
output_csv_path = 'D:/enviprojects/Berlin_Mehringdamm_Base/TrajectoryOutputData/output_counts.csv' # CHANGE THIS PATH
output_gpkg_path = 'D:/enviprojects/Berlin_Mehringdamm_Base/TrajectoryOutputData/output_counts.gpkg' # CHANGE THIS PATH


# Get layers
segments_layer = QgsProject.instance().mapLayersByName(segments_layer_name)[0]
traj_layer = QgsProject.instance().mapLayersByName(trajectories_layer_name)[0]

# ==========================================
# 2. CREATE A SAFE COPY IN MEMORY
# ==========================================
print("Creating a copy of the segments layer...")
crs_str = segments_layer.crs().toWkt()
memory_layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Segment_Counts", "memory")
provider = memory_layer.dataProvider()

# Copy original fields AND add our new ones
fields = segments_layer.fields()
provider.addAttributes(fields.toList())
provider.addAttributes([QgsField("enviID", QVariant.Int)])
for h in range(24):
    provider.addAttributes([QgsField(f"h_{h:02d}", QVariant.Int)])
memory_layer.updateFields()

# Copy features over
new_features = []
for feat in segments_layer.getFeatures():
    new_feat = QgsFeature(memory_layer.fields())
    new_feat.setGeometry(feat.geometry())
    
    # Copy original attributes
    for i in range(len(fields)):
        new_feat.setAttribute(i, feat.attribute(i))
        
    new_features.append(new_feat)
provider.addFeatures(new_features)

# ==========================================
# 3. BUILD SPATIAL INDEX & TRAJECTORY DATA
# ==========================================
print("Building spatial index for trajectories...")
traj_index = QgsSpatialIndex(traj_layer.getFeatures())

traj_data = {}
for feat in traj_layer.getFeatures():
    # Convert seconds_start to an hour index (0-23)
    start_hour = int(feat[datetime_field] // 3600)
    
    if 0 <= start_hour <= 23:
        traj_data[feat.id()] = {
            'geom': feat.geometry(),
            'hour': start_hour,
            'group_id': feat[unique_id_field]
        }
print("Spatial index built!")

# ==========================================
# 4. ITERATE AND COUNT UNIQUE VEHICLES
# ==========================================
print("Processing street segments and counting unique vehicles...")
update_map = {}
envi_counter = 1

# Get the starting index for our new fields
envi_idx = memory_layer.fields().indexOf("enviID")
h_indices = {h: memory_layer.fields().indexOf(f"h_{h:02d}") for h in range(24)}

for seg_feat in memory_layer.getFeatures():
    seg_geom = seg_feat.geometry()
    
    # Bounding box + radius for fast filtering
    search_rect = seg_geom.boundingBox()
    search_rect.grow(search_radius)
    candidate_ids = traj_index.intersects(search_rect)
    
    # Use sets to store unique group_ids per hour
    unique_counts = {h: set() for h in range(24)}
    
    # Exact distance check
    for c_id in candidate_ids:
        # Some trajectories might have hours outside 0-23 and were skipped
        if c_id not in traj_data: 
            continue
            
        t_geom = traj_data[c_id]['geom']
        t_hour = traj_data[c_id]['hour']
        t_group_id = traj_data[c_id]['group_id']
        
        # If the trajectory is within 0.7m of the segment line
        if seg_geom.distance(t_geom) <= search_radius:
            unique_counts[t_hour].add(t_group_id)
            
    # Prepare updates for this segment
    attr_update = {envi_idx: envi_counter}
    for h in range(24):
        # The count is the number of unique group_ids
        attr_update[h_indices[h]] = len(unique_counts[h])
        
    update_map[seg_feat.id()] = attr_update
    envi_counter += 1

# Apply all counts to the memory layer
provider.changeAttributeValues(update_map)

# ==========================================
# 5. EXPORT TO GEOPACKAGE AND CSV
# ==========================================
print("Saving results to GeoPackage...")
save_options_gpkg = QgsVectorFileWriter.SaveVectorOptions()
save_options_gpkg.driverName = "GPKG"
save_options_gpkg.layerName = "segment_counts" # Name of the layer inside the GPKG

# Capture all returned values in a single variable 'res_gpkg' to avoid the unpack error
res_gpkg = QgsVectorFileWriter.writeAsVectorFormatV3(
    memory_layer,
    output_gpkg_path,
    QgsProject.instance().transformContext(),
    save_options_gpkg
)

if res_gpkg[0] == QgsVectorFileWriter.NoError:
    print(f"SUCCESS! Created GeoPackage: {output_gpkg_path}")
    
    # Load the GPKG into the map so you can visually check the lines
    vlayer = QgsVectorLayer(f"{output_gpkg_path}|layername=segment_counts", "Segment Counts", "ogr")
    if vlayer.isValid():
        QgsProject.instance().addMapLayer(vlayer)
else:
    print(f"Error saving GeoPackage. Error details: {res_gpkg}")


print("Saving results to CSV...")
save_options_csv = QgsVectorFileWriter.SaveVectorOptions()
save_options_csv.driverName = "CSV"
# Optional: drop geometry for a cleaner CSV by uncommenting below
# save_options_csv.skipAttributeCreation = True 

res_csv = QgsVectorFileWriter.writeAsVectorFormatV3(
    memory_layer,
    output_csv_path,
    QgsProject.instance().transformContext(),
    save_options_csv
)

if res_csv[0] == QgsVectorFileWriter.NoError:
    print(f"SUCCESS! Created CSV: {output_csv_path}")
else:
    print(f"Error saving CSV. Error details: {res_csv}")

print("--- Workflow Complete ---")