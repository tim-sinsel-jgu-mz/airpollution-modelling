import json
import os
from datetime import datetime

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# 1. File Paths
INPUT_FILE = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\merge7_clean_2024_Jun_Nov_smthWind_realBG2.FOX" # Replace with your actual input file path

# 2. Time Range (Inclusive)
# The script will interpolate the values for all time steps that fall within 
# this range, using the time step immediately BEFORE and AFTER this range.
START_TIME_STR = "2018-06-26 11:30:00"
END_TIME_STR = "2018-06-26 12:00:00"

# 3. Variables to Interpolate
# Define the path to each value as a list of keys (strings) and indices (integers).
# Examples based on ENVI-met FOX structure:
# - Top-level scalar: ["swDir"]
# - Inside dictionary: ["backgrPollutants", "O3"]
# - Inside array: ["tProfile", 0, "value"] or ["windProfile", 0, "wSpdValue"]
VARIABLES_TO_INTERPOLATE = [
    ["swDir"],
    ["swDif"]
]

# =============================================================================
# SCRIPT LOGIC (No need to edit below)
# =============================================================================

def get_nested_value(data_dict, path):
    """Retrieves a value from a nested dictionary/list using a path."""
    curr = data_dict
    for key in path:
        curr = curr[key]
    return curr

def set_nested_value(data_dict, path, value):
    """Sets a value in a nested dictionary/list using a path."""
    curr = data_dict
    for key in path[:-1]:
        curr = curr[key]
    curr[path[-1]] = value

def main():
    # Generate Output Filename
    filename, ext = os.path.splitext(INPUT_FILE)
    output_file = f"{filename}_fix2606{ext}"

    # Load JSON
    print(f"Loading '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file '{INPUT_FILE}'.")
        return

    # Parse target datetimes
    time_format = "%Y-%m-%d %H:%M:%S"
    target_start = datetime.strptime(START_TIME_STR, time_format)
    target_end = datetime.strptime(END_TIME_STR, time_format)

    # Parse all timestamps in the file
    timesteps = data.get('timestepList', [])
    times = []
    for ts in timesteps:
        dt_str = f"{ts['date']} {ts['time']}"
        times.append(datetime.strptime(dt_str, time_format))

    # Find indices for the interpolation range
    interp_indices = [i for i, t in enumerate(times) if target_start <= t <= target_end]

    if not interp_indices:
        print("Error: No time steps found falling within the specified start and end times.")
        return

    # Identify outer boundary indices
    idx_first_interp = interp_indices[0]
    idx_last_interp = interp_indices[-1]

    idx_before = idx_first_interp - 1
    idx_after = idx_last_interp + 1

    # Bounds checking
    if idx_before < 0:
        print("Error: Cannot interpolate. The start time encompasses the very first time step, so there is no 'outer' previous step to interpolate from.")
        return
    if idx_after >= len(timesteps):
        print("Error: Cannot interpolate. The end time encompasses the very last time step, so there is no 'outer' next step to interpolate towards.")
        return

    t0 = times[idx_before]
    t1 = times[idx_after]
    total_time_delta = (t1 - t0).total_seconds()

    print(f"Interpolating {len(interp_indices)} time steps.")
    print(f"Using outer bounds: {t0} (Index {idx_before}) -> {t1} (Index {idx_after})")

    # Perform Interpolation
    for var_path in VARIABLES_TO_INTERPOLATE:
        try:
            val_0 = get_nested_value(timesteps[idx_before], var_path)
            val_1 = get_nested_value(timesteps[idx_after], var_path)
            
            # Type check to ensure we are interpolating numbers
            if not isinstance(val_0, (int, float)) or not isinstance(val_1, (int, float)):
                print(f"Warning: Skipping {var_path} as boundary values are not numeric.")
                continue

            # Interpolate for each time step in the target range
            for i in interp_indices:
                t_current = times[i]
                current_time_delta = (t_current - t0).total_seconds()
                
                # Linear Interpolation Formula
                interpolated_value = val_0 + (val_1 - val_0) * (current_time_delta / total_time_delta)
                
                # Update the JSON structure
                set_nested_value(timesteps[i], var_path, interpolated_value)
                
            print(f"Successfully interpolated: {' -> '.join(map(str, var_path))}")
            
        except (KeyError, IndexError):
            print(f"Warning: Could not resolve path {' -> '.join(map(str, var_path))} at outer indices. Check your configuration.")

    # Save output
    print(f"Saving to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()