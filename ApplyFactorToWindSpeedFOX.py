import json
from datetime import datetime

# ================= USER CONFIGURATION =================
INPUT_FILE = 'merge7_clean_2024_Jun_Nov_smthWind_realBG2_fix0626.FOX'
OUTPUT_FILE = 'merge7_clean_2024_Jun_Nov_smthWind_realBG2_fix0626_wind0.5.FOX'

# Define the date-time range (Inclusive)
# Format: YYYY-MM-DD HH:MM:SS
START_RANGE = "2018-06-01 01:10:00"
END_RANGE   = "2018-12-01 00:50:00"

# Multiplication factor for wind speed (wSpdValue)
WIND_FACTOR = 0.5 
# ======================================================

def modify_fox_wind_speed():
    # Load the date-time range into datetime objects for comparison
    start_dt = datetime.strptime(START_RANGE, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(END_RANGE, "%Y-%m-%d %H:%M:%S")

    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON. Ensure the FOX file is valid.")
        return

    modified_count = 0

    # Iterate through each timestep in the forcing file
    for timestep in data.get("timestepList", []):
        # Combine date and time strings to create a datetime object
        current_dt_str = f"{timestep['date']} {timestep['time']}"
        current_dt = datetime.strptime(current_dt_str, "%Y-%m-%d %H:%M:%S")

        # Check if the current timestep falls within the user-defined range
        if start_dt <= current_dt <= end_dt:
            # Apply factor to all entries in the windProfile list
            for entry in timestep.get("windProfile", []):
                if "wSpdValue" in entry:
                    entry["wSpdValue"] *= WIND_FACTOR
            modified_count += 1

    # Save the modified data to the new file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Success!")
    print(f"Modified {modified_count} timesteps.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    modify_fox_wind_speed()