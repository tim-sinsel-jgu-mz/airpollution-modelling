# ============================================================
# USER INSTRUCTIONS (PLEASE READ BEFORE RUNNING THE SCRIPT)
# ============================================================
# 1) PLEASE cut the OpenStreetMap roads layer down to the size
#    of your study area BEFORE running this script.
#
# 2) PLEASE export the OSM roads layer in QGIS before using
#    this script:
#       - Set the CRS explicitly to a METRIC CRS
#         (preferably the PROJECT CRS)
#       - Set the filename according to INPUT_FILENAME below
#
# 3) Place this script (.py file) in the SAME FOLDER as
#    the input shapefile and its accompanying files
#    (.dbf, .shx, etc.)
#
# 4) The resulting intersection buffers may require some
#    MANUAL CLEANUP or TWEAKING before further use.
# ============================================================

import os
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsVectorFileWriter
)
import processing

# ============================================================
# USER-DEFINED PARAMETERS
# ============================================================

# Input and output filenames (must be in the same folder as this script)
INPUT_FILENAME = "osm_roads.shp"
OUTPUT_FILENAME = "osm_roads_intersections.shp"

# Buffer radius in meters
BUFFER_RADIUS = 20  # default: 20 m

# ============================================================
# FIND SCRIPT DIRECTORY AND INPUT FILE
# ============================================================

try:
    script_dir = os.path.dirname(__file__)
except NameError:
    print("ERROR: __file__ is not defined.")
    print("Please run this script as a .py file using 'Run Script' in QGIS.")
    raise

input_path = os.path.join(script_dir, INPUT_FILENAME)

if not os.path.exists(input_path):
    print(f"ERROR: Could not find input file '{INPUT_FILENAME}'.")
    print("Make sure it is located in the same folder as this script.")
    raise FileNotFoundError

# ============================================================
# LOAD INPUT LAYER
# ============================================================

input_layer = QgsVectorLayer(input_path, "osm_roads", "ogr")

if not input_layer.isValid():
    print(f"ERROR: The input layer '{INPUT_FILENAME}' could not be loaded.")
    raise RuntimeError

# ============================================================
# CRS CHECK (MUST BE METRIC)
# ============================================================

crs = input_layer.crs()

if crs.isGeographic():
    print("ERROR: The CRS of the input layer is NOT metric.")
    print("Please export the layer in QGIS and set a METRIC CRS")
    print("(preferably the Project CRS) in the export settings.")
    raise RuntimeError

# ============================================================
# FILTER ROAD CLASSES
# ============================================================

allowed_classes = (
    "'primary', "
    "'secondary', "
    "'tertiary', "
    "'secondary_link', "
    "'residential', "
    "'living_street'"
)

filtered = processing.run(
    "native:extractbyexpression",
    {
        "INPUT": input_layer,
        "EXPRESSION": f"\"fclass\" IN ({allowed_classes})",
        "OUTPUT": "memory:"
    }
)["OUTPUT"]

# ============================================================
# DISSOLVE BY STREET NAME
# ============================================================

dissolved_roads = processing.run(
    "native:dissolve",
    {
        "INPUT": filtered,
        "FIELD": ["name"],
        "OUTPUT": "memory:"
    }
)["OUTPUT"]

# ============================================================
# LINE INTERSECTIONS (A = B = dissolved roads)
# ============================================================

intersections = processing.run(
    "native:lineintersections",
    {
        "INPUT": dissolved_roads,
        "INTERSECT": dissolved_roads,
        "INPUT_FIELDS": [],
        "INTERSECT_FIELDS": [],
        "OUTPUT": "memory:"
    }
)["OUTPUT"]

# ============================================================
# DELETE DUPLICATE GEOMETRIES (POINTS)
# ============================================================

unique_points = processing.run(
    "native:deleteduplicategeometries",
    {
        "INPUT": intersections,
        "OUTPUT": "memory:"
    }
)["OUTPUT"]

# ============================================================
# BUFFER INTERSECTION POINTS
# ============================================================

buffers = processing.run(
    "native:buffer",
    {
        "INPUT": unique_points,
        "DISTANCE": BUFFER_RADIUS,
        "SEGMENTS": 20,
        "END_CAP_STYLE": 0,  # round
        "JOIN_STYLE": 0,
        "MITER_LIMIT": 2,
        "DISSOLVE": False,
        "OUTPUT": "memory:"
    }
)["OUTPUT"]


# ============================================================
# DETERMINE UNIQUE OUTPUT FILENAME
# ============================================================

base_name, ext = os.path.splitext(OUTPUT_FILENAME)
final_output_name = OUTPUT_FILENAME
counter = 1

while os.path.exists(os.path.join(script_dir, final_output_name)):
    final_output_name = f"{base_name}_{counter}{ext}"
    counter += 1

output_path = os.path.join(script_dir, final_output_name)

# ============================================================
# EXPORT FINAL RESULT
# ============================================================

QgsVectorFileWriter.writeAsVectorFormat(
    buffers,
    output_path,
    "UTF-8",
    buffers.crs(),
    "ESRI Shapefile"
)

# ============================================================
# LOAD RESULT INTO QGIS
# ============================================================

layer_name = os.path.splitext(final_output_name)[0]
result_layer = QgsVectorLayer(output_path, layer_name, "ogr")
QgsProject.instance().addMapLayer(result_layer)

print(f"SUCCESS: '{final_output_name}' created and loaded into QGIS.")
