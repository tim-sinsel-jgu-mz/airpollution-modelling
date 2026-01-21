# -*- coding: utf-8 -*-
"""
QGIS Trajectory Processing Script

INSTRUCTIONS FOR USER:
1.  Ensure 'trajectories.gpkg', 'osm_roads_intersections.shp' and 'study_area.gpkg' are in the same 
    folder as this script.
2.  Please export your layers in QGIS to the same METRIC CRS (e.g., UTM) before starting.
    You can do this by right-clicking the layer -> Export -> Save Features As...
    and selecting a metric CRS in the CRS dropdown (preferably your Project CRS).
3.  If CLIP_TO_STUDY_AREA is True, the trajectories layer will be clipped to the size of the study area.    
4.  If SCALING_ENABLED is True, results will be multiplied by the SCALING_FACTOR.
"""
 
print("--- Script Loaded. Initializing... ---")

import os
import processing
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsField,
    QgsFields,
    QgsFeature,
    QgsGeometry,
    QgsSpatialIndex,
    QgsCoordinateReferenceSystem,
    QgsVectorFileWriter
)
from PyQt5.QtCore import QVariant

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_TRAJECTORIES = "trajectories.gpkg"
INPUT_INTERSECTIONS = "osm_roads_intersections.shp"
INPUT_STUDY_AREA = "study_area.gpkg"
OUTPUT_BASENAME = "trajectory_counts"

# --- OPTIONS ---
SCALING_ENABLED = True     # Set to True to multiply counts by factor
SCALING_FACTOR = 5          # Factor to multiply by (e.g. 5 for 20% sample)

CLIP_TO_STUDY_AREA = True   # Set to True to clip trajectories before processing

# Geometry Parameters
BUFFER_DIST = 0.75          # Lane Buffer Radius in m. Default is 0.75
SIMPLIFY_TOLERANCE = 0.5    # Simplify Geometries tolerance in m. Default is 0.5

# Sampling Parameters (for Lane Creation Only)
SAMPLING_RATE = 6           # Every x-th trajectory within the defined time windows will be selected for the lane buffer creation. Dissolve tool often failed when exceeding ~5000 trajectories.
WINDOWS = [(6, 9), (16, 19)]    # Time windows used to select the trajectories from. Peak traffic hours, so most if not every lane should have been used.

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_script_dir():
    try:
        return os.path.dirname(os.path.realpath(__file__))
    except NameError:
        pass
    return QgsProject.instance().homePath()

def get_unique_filepath(folder, basename, extension):
    counter = 0
    filename = f"{basename}.{extension}"
    full_path = os.path.join(folder, filename)
    while os.path.exists(full_path):
        counter += 1
        filename = f"{basename}_{counter}.{extension}"
        full_path = os.path.join(folder, filename)
    return full_path

def check_metric_crs(layer):
    return not layer.crs().isGeographic()

# ==============================================================================
# MAIN PROCESS
# ==============================================================================

def main():
    print(f"--- Starting Analysis ---")
    print(f"   > Scaling: {'ON (x' + str(SCALING_FACTOR) + ')' if SCALING_ENABLED else 'OFF'}")
    print(f"   > Clipping: {'ON' if CLIP_TO_STUDY_AREA else 'OFF'}")
    
    work_dir = get_script_dir()
    traj_path = os.path.join(work_dir, INPUT_TRAJECTORIES)
    inter_path = os.path.join(work_dir, INPUT_INTERSECTIONS)
    area_path = os.path.join(work_dir, INPUT_STUDY_AREA)

    # 1. LOAD LAYERS
    if not os.path.exists(traj_path) or not os.path.exists(inter_path):
        print("ERROR: Main input files missing.")
        return

    traj_layer = QgsVectorLayer(traj_path, "Trajectories", "ogr")
    inter_layer = QgsVectorLayer(inter_path, "Intersections", "ogr")

    if not traj_layer.isValid() or not inter_layer.isValid():
        print("ERROR: Could not load main layers.")
        return

    if not check_metric_crs(traj_layer) or not check_metric_crs(inter_layer):
        print("ERROR: Layers must be in a metric CRS (e.g. UTM).")
        return

    # 2. PRE-PROCESS: CLIPPING (OPTIONAL)
    # The 'working_layer' variable will hold the layer we use for everything else
    working_layer = traj_layer 

    if CLIP_TO_STUDY_AREA:
        if not os.path.exists(area_path):
            print(f"ERROR: Clipping is ON but '{INPUT_STUDY_AREA}' was not found.")
            return
            
        area_layer = QgsVectorLayer(area_path, "Study Area", "ogr")
        if not area_layer.isValid():
            print("ERROR: Study area layer failed to load.")
            return
            
        print("Step 0: Clipping trajectories to study area...")
        clipped = processing.run("native:clip", {
            'INPUT': traj_layer,
            'OVERLAY': area_layer,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
        
        # Update working layer to be the clipped version
        working_layer = clipped
        print(f"   > Trajectories reduced from {traj_layer.featureCount()} to {working_layer.featureCount()}")

    # 3. GENERATE LANE POLYGONS (SAMPLED)
    print("Step 1: Creating Lane Polygons...")
    
    # Build expression
    time_conditions = [f'("seconds_start" >= {s*3600} AND "seconds_start" < {e*3600})' for s, e in WINDOWS]
    combined_expression = f"({' OR '.join(time_conditions)}) AND (@id % {SAMPLING_RATE} = 0)"
    
    # Note: We use 'working_layer' here (either raw or clipped)
    extracted = processing.run("native:extractbyexpression", {
        'INPUT': working_layer,  
        'EXPRESSION': combined_expression,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    buffered = processing.run("native:buffer", {'INPUT': extracted, 'DISTANCE': BUFFER_DIST, 'END_CAP_STYLE': 2, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    simplified = processing.run("native:simplifygeometries", {'INPUT': buffered, 'TOLERANCE': SIMPLIFY_TOLERANCE, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    dissolved = processing.run("native:dissolve", {'INPUT': simplified, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    lanes_layer = processing.run("native:splitwithlines", {'INPUT': dissolved, 'LINES': inter_layer, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

    # 4. PREPARE OUTPUT LAYER (NO TOTAL COUNT)
    fields = QgsFields()
    fields.append(QgsField("ENVIID", QVariant.String, len=6))
    for h in range(24):
        fields.append(QgsField(f"h_{h}", QVariant.Int))
    # 'total_cnt' removed
    
    crs_str = working_layer.crs().toWkt()
    out_layer = QgsVectorLayer(f"Polygon?crs={crs_str}", "Lanes", "memory")
    out_dp = out_layer.dataProvider()
    out_dp.addAttributes(fields)
    out_layer.updateFields()
    
    new_feats = []
    for i, feat in enumerate(lanes_layer.getFeatures()):
        new_feat = QgsFeature(out_layer.fields())
        new_feat.setGeometry(feat.geometry())
        new_feat.setAttribute("ENVIID", f"{i+1:06d}")
        for h in range(24): new_feat.setAttribute(f"h_{h}", 0)
        new_feats.append(new_feat)
    out_dp.addFeatures(new_feats)
    
    spatial_index = QgsSpatialIndex(out_layer.getFeatures())

    # 5. COUNTING LOGIC
    print("Step 2: Counting trajectories based on start hour...")
    # Map structure: { lane_fid: { hour: set(group_ids) } }
    counts_map = { f.id(): { h: set() for h in range(24) } for f in out_layer.getFeatures() }
    
    total_feats = working_layer.featureCount()
    step = 0
    
    # Iterate through 'working_layer' (clipped or raw)
    for traj in working_layer.getFeatures():
        step += 1
        if step % 5000 == 0: print(f"  Processed {step}/{total_feats} trajectories...")
            
        geom = traj.geometry()
        if not geom or geom.isEmpty(): continue
            
        start_hour = int(traj["seconds_start"] // 3600)
        if start_hour < 0 or start_hour > 23: continue 
        
        group_id = traj["group_id"]
        candidates = spatial_index.intersects(geom.boundingBox())
        for l_id in candidates:
            lane_feat = out_layer.getFeature(l_id)
            if lane_feat.geometry().intersects(geom):
                counts_map[l_id][start_hour].add(group_id)

    # 6. FINALIZE ATTRIBUTES (SCALING & HOURLY ONLY)
    print("Step 3: Applying counts and scaling...")
    updates = {}
    idx_map = { h: out_layer.fields().indexOf(f"h_{h}") for h in range(24) }
    
    for l_id, h_data in counts_map.items():
        attr_map = {}
        for h in range(24):
            unique_ids = h_data[h]
            count = len(unique_ids)
            
            if SCALING_ENABLED:
                count *= SCALING_FACTOR
            
            attr_map[idx_map[h]] = count
        
        updates[l_id] = attr_map

    out_dp.changeAttributeValues(updates)

    # 7. SAVE
    out_gpkg = get_unique_filepath(work_dir, OUTPUT_BASENAME, "gpkg")
    out_csv = get_unique_filepath(work_dir, OUTPUT_BASENAME, "csv")
    
    save_opt = QgsVectorFileWriter.SaveVectorOptions()
    save_opt.driverName = "GPKG"
    QgsVectorFileWriter.writeAsVectorFormatV3(out_layer, out_gpkg, QgsProject.instance().transformContext(), save_opt)
    save_opt.driverName = "CSV"
    QgsVectorFileWriter.writeAsVectorFormatV3(out_layer, out_csv, QgsProject.instance().transformContext(), save_opt)
    
    print(f"SUCCESS! Created:\n{out_gpkg}\n{out_csv}")

main()