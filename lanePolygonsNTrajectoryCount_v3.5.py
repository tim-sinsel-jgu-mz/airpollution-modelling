# -*- coding: utf-8 -*-
"""
QGIS Trajectory Processing Script

INSTRUCTIONS FOR USER:
1.  Ensure 'trajectories.gpkg' and 'osm_roads_intersections.shp' are in the same 
    folder as this script.
2.  Please ensure your 'trajectories.gpkg' is cropped to your specific study area 
    BEFORE running this script to ensure performance.
3.  Please export your layers in QGIS to a METRIC CRS (e.g., UTM) before starting.
    You can do this by right-clicking the layer -> Export -> Save Features As...
    and selecting a metric CRS in the CRS dropdown (preferably your Project CRS).
4.  Set both 'trajectories.gpkg' and 'osm_roads_intersections.shp' to the
    same CRS this way.
4.  If SCALING_ENABLED is True, results will be multiplied by the SCALING_FACTOR.
"""

print("--- Script Loaded. Initializing with Scaling Options... ---")

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
    QgsWkbTypes,
    QgsProcessingFeatureSourceDefinition,
    QgsCoordinateReferenceSystem,
    QgsVectorFileWriter
)
from PyQt5.QtCore import QVariant

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Change file names here if necessary
INPUT_TRAJECTORIES = "trajectories.gpkg"
INPUT_INTERSECTIONS = "osm_roads_intersections.shp"
OUTPUT_BASENAME = "trajectory_counts"

# --- SCALING FEATURE ---
SCALING_ENABLED = True  # Set to True to multiply counts by the factor below
SCALING_FACTOR = 5       # Default is 5 (for 20% sample data)

# Geometry Parameters
BUFFER_DIST = 0.75  # Lane Buffer Radius in m
SIMPLIFY_TOLERANCE = 0.5    # Simplify Geometries tolerance in m

# Sampling Parameters (for Lane Creation Only)
SAMPLING_RATE = 6   # Every x-th trajectory within the defined time windows will be selected for the lane buffer creation. Default is 6.
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
    if SCALING_ENABLED:
        print(f"--- Starting Analysis (SCALING ON: Factor x{SCALING_FACTOR}) ---")
    else:
        print("--- Starting Analysis (SCALING OFF: Raw Counts) ---")
    
    work_dir = get_script_dir()
    traj_path = os.path.join(work_dir, INPUT_TRAJECTORIES)
    inter_path = os.path.join(work_dir, INPUT_INTERSECTIONS)

    if not os.path.exists(traj_path) or not os.path.exists(inter_path):
        print("ERROR: Input files missing. Check your directory.")
        return

    traj_layer = QgsVectorLayer(traj_path, "Trajectories", "ogr")
    inter_layer = QgsVectorLayer(inter_path, "Intersections", "ogr")

    if not traj_layer.isValid() or not inter_layer.isValid():
        print("ERROR: Could not load layers.")
        return

    if not check_metric_crs(traj_layer) or not check_metric_crs(inter_layer):
        print("ERROR: Layers must be in a metric CRS (e.g. UTM).")
        return

    # 1. GENERATE LANE POLYGONS
    print("Step 1: Creating Lane Polygons...")
    time_conditions = [f'("seconds_start" >= {s*3600} AND "seconds_start" < {e*3600})' for s, e in WINDOWS]
    combined_expression = f"({' OR '.join(time_conditions)}) AND (@id % {SAMPLING_RATE} = 0)"
    
    extracted = processing.run("native:extractbyexpression", {
        'INPUT': traj_layer,
        'EXPRESSION': combined_expression,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    buffered = processing.run("native:buffer", {'INPUT': extracted, 'DISTANCE': BUFFER_DIST, 'END_CAP_STYLE': 2, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    simplified = processing.run("native:simplifygeometries", {'INPUT': buffered, 'TOLERANCE': SIMPLIFY_TOLERANCE, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    dissolved = processing.run("native:dissolve", {'INPUT': simplified, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    lanes_layer = processing.run("native:splitwithlines", {'INPUT': dissolved, 'LINES': inter_layer, 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

    # 2. PREPARE OUTPUT LAYER
    fields = QgsFields()
    fields.append(QgsField("ENVIID", QVariant.String, len=6))
    for h in range(24):
        fields.append(QgsField(f"h_{h}", QVariant.Int))
    fields.append(QgsField("total_cnt", QVariant.Int))
    
    out_layer = QgsVectorLayer(f"Polygon?crs={traj_layer.crs().toWkt()}", "Lanes", "memory")
    out_dp = out_layer.dataProvider()
    out_dp.addAttributes(fields)
    out_layer.updateFields()
    
    new_feats = []
    for i, feat in enumerate(lanes_layer.getFeatures()):
        new_feat = QgsFeature(out_layer.fields())
        new_feat.setGeometry(feat.geometry())
        new_feat.setAttribute("ENVIID", f"{i+1:06d}")
        for h in range(24): new_feat.setAttribute(f"h_{h}", 0)
        new_feat.setAttribute("total_cnt", 0)
        new_feats.append(new_feat)
    out_dp.addFeatures(new_feats)
    
    spatial_index = QgsSpatialIndex(out_layer.getFeatures())

    # 3. COUNTING LOGIC
    print("Step 2: Counting trajectories based on start hour...")
    counts_map = { f.id(): { h: set() for h in range(24) } for f in out_layer.getFeatures() }
    
    total_traj = traj_layer.featureCount()
    step = 0
    for traj in traj_layer.getFeatures():
        step += 1
        if step % 5000 == 0: print(f"  Processed {step}/{total_traj} trajectories...")
            
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

    # 4. FINALIZE ATTRIBUTES (WITH SCALING)
    print("Step 3: Calculating totals and applying scaling...")
    updates = {}
    idx_map = { h: out_layer.fields().indexOf(f"h_{h}") for h in range(24) }
    idx_total = out_layer.fields().indexOf("total_cnt")
    
    for l_id, h_data in counts_map.items():
        attr_map = {}
        daily_unique = set()
        
        for h in range(24):
            unique_ids = h_data[h]
            count = len(unique_ids)
            
            # Apply scaling if enabled
            if SCALING_ENABLED:
                count *= SCALING_FACTOR
            
            attr_map[idx_map[h]] = count
            daily_unique.update(unique_ids)
        
        total_count = len(daily_unique)
        if SCALING_ENABLED:
            total_count *= SCALING_FACTOR
            
        attr_map[idx_total] = total_count
        updates[l_id] = attr_map

    out_dp.changeAttributeValues(updates)

    # 5. SAVE
    out_gpkg = get_unique_filepath(work_dir, OUTPUT_BASENAME, "gpkg")
    out_csv = get_unique_filepath(work_dir, OUTPUT_BASENAME, "csv")
    
    save_opt = QgsVectorFileWriter.SaveVectorOptions()
    save_opt.driverName = "GPKG"
    QgsVectorFileWriter.writeAsVectorFormatV3(out_layer, out_gpkg, QgsProject.instance().transformContext(), save_opt)
    save_opt.driverName = "CSV"
    QgsVectorFileWriter.writeAsVectorFormatV3(out_layer, out_csv, QgsProject.instance().transformContext(), save_opt)
    
    print(f"SUCCESS! Created:\n{out_gpkg}\n{out_csv}")

main()