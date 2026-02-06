import pandas as pd
import json
import os
from datetime import datetime, timedelta
import pytz
import warnings

# Suppress warnings about 'T' vs 'min' in newer pandas versions
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# USER CONFIGURATION
# ==========================================

# --- File Names ---
INPUT_CSV_FILE = 'ber_mc010_20240714-20240716.csv'
INPUT_FOX_FILE = 'merge7_clean_2024_Jun_Nov_smthWind.FOX'
OUTPUT_FOX_FILE = INPUT_FOX_FILE.replace('.FOX', '_hourlyBG.FOX')

# --- Time Zone Settings ---
# BLUME usually uses German wall clock time (switches MEZ/MESZ automatically)
CSV_TIMEZONE = 'Europe/Berlin' 
# FOX/ENVI-met usually uses fixed Local Standard Time (Winter time) or UTC
FOX_OFFSET_HOURS = 1  # 1 = UTC+1 (MEZ). Set to 0 if FOX is in pure UTC.

# --- Data Matching Settings ---
# If True, treats CSV data AS IF it happened in the year found inside the FOX file.
IGNORE_YEAR_MISMATCH = True 

# If True, replaces all remaining '-999' values in the FOX file with '0.0'.
# This applies to pollutants not in your CSV and timesteps outside your CSV range.
SET_MISSING_TO_ZERO = True

# --- Column Mapping ---
# Map the ENVI-met internal names (Keys) to the BLUME CSV headers (Values).
POLLUTANT_MAP = {
    "PM10": "Feinstaub (PM10)",
    "PM25": "Feinstaub (PM2,5)",
    "NO2":  "Stickstoffdioxid",
    "NO":   "Stickstoffmonoxid",
    "O3":   "Ozon"
}

# ==========================================
# SCRIPT START
# ==========================================

def get_fox_year(fox_path):
    """Reads the first timestep of the FOX file to determine the simulation year."""
    try:
        with open(fox_path, 'r') as f:
            fox_data = json.load(f)
        
        if not fox_data.get('timestepList'):
            return None
            
        first_date = fox_data['timestepList'][0]['date'] # "YYYY-MM-DD"
        return int(first_date.split('-')[0])
    except Exception as e:
        print(f"Error reading FOX year: {e}")
        return None

def load_and_process_csv(filepath, target_year=None):
    print(f"Reading CSV: {filepath}...")
    
    try:
        # 1. Read Header to find columns
        df_headers = pd.read_csv(filepath, sep=';', encoding='latin1', header=None, nrows=6)
        
        component_row_idx = -1
        for i, row in df_headers.iterrows():
            if row.astype(str).str.contains("Messkomponente").any():
                component_row_idx = i
                break
        
        if component_row_idx == -1:
            raise ValueError("Could not find row starting with 'Messkomponente' in CSV.")

        headers = df_headers.iloc[component_row_idx].values
        skip_rows = component_row_idx + 3 
        
        # 2. Read Data
        df = pd.read_csv(filepath, sep=';', decimal=',', encoding='latin1', 
                         header=None, skiprows=skip_rows, names=headers,
                         na_values=['-', 'nan', '']) 
        
        # 3. Parse Dates
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%Y %H:%M')
        
        # 4. Timezone Handling (Normalize to UTC)
        local_tz = pytz.timezone(CSV_TIMEZONE)
        df = df.set_index(date_col)
        df.index = df.index.tz_localize(local_tz, ambiguous='infer').tz_convert(pytz.UTC)
        
        # 5. Handle Year Mismatch (Project CSV to FOX Year)
        if IGNORE_YEAR_MISMATCH and target_year:
            csv_year = df.index[0].year
            if csv_year != target_year:
                print(f"   -> Adjusting CSV year ({csv_year}) to match FOX file year ({target_year})...")
                
                # Logic to replace year safely (handles Leap Years)
                def replace_year_safe(dt):
                    try:
                        return dt.replace(year=target_year)
                    except ValueError:
                        return None 

                new_index = df.index.map(replace_year_safe)
                
                # Drop invalid dates
                valid_mask = new_index.notnull()
                df = df[valid_mask]
                df.index = new_index[valid_mask]

        # 6. Filter Columns
        cols_to_keep = []
        rename_dict = {}
        found_columns = df.columns.tolist()
        
        for envimet_name, csv_name in POLLUTANT_MAP.items():
            match = next((col for col in found_columns if csv_name in str(col)), None)
            if match:
                cols_to_keep.append(match)
                rename_dict[match] = envimet_name

        df = df[cols_to_keep].rename(columns=rename_dict)
        
        # 7. Resample (Linear Interpolation) to 10 minutes
        print("   -> Resampling to 10-minute intervals (Linear)...")
        df_resampled = df.resample('10min').interpolate(method='linear')
        
        return df_resampled

    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

def update_fox_file(fox_path, output_path, pollutant_df):
    print(f"\nReading FOX file: {fox_path}...")
    
    with open(fox_path, 'r') as f:
        fox_data = json.load(f)
    
    timesteps = fox_data.get('timestepList', [])
    if not timesteps:
        print("Error: FOX file has no timesteps.")
        return

    # FOX Timezone Offset
    fox_offset = timedelta(hours=FOX_OFFSET_HOURS)

    # Calculate ranges for display
    fox_start_utc = datetime.strptime(f"{timesteps[0]['date']} {timesteps[0]['time']}", "%Y-%m-%d %H:%M:%S") - fox_offset
    fox_start_utc = fox_start_utc.replace(tzinfo=pytz.UTC)
    
    fox_end_utc = datetime.strptime(f"{timesteps[-1]['date']} {timesteps[-1]['time']}", "%Y-%m-%d %H:%M:%S") - fox_offset
    fox_end_utc = fox_end_utc.replace(tzinfo=pytz.UTC)

    csv_start = pollutant_df.index[0]
    csv_end = pollutant_df.index[-1]

    print(f"\n--- Coverage Check (UTC) ---")
    print(f"FOX Period: {fox_start_utc} to {fox_end_utc}")
    print(f"CSV Period: {csv_start} to {csv_end}")

    if fox_start_utc < csv_start or fox_end_utc > csv_end:
        print("\n[WARNING] The FOX file extends beyond the CSV data range.")
        print("          Some timesteps will not be updated.")
    
    updated_count = 0
    cleaned_count = 0
    
    for step in timesteps:
        # --- NEW: Set -999 to 0 ---
        if SET_MISSING_TO_ZERO and 'backgrPollutants' in step:
            for key, val in step['backgrPollutants'].items():
                if val == -999:
                    step['backgrPollutants'][key] = 0.0
                    cleaned_count += 1
        
        # --- EXISTING: Update from CSV ---
        # Parse FOX time
        ts_str = f"{step['date']} {step['time']}"
        ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        
        # Convert to UTC
        ts_utc = ts_dt - fox_offset
        ts_utc = ts_utc.replace(tzinfo=pytz.UTC)
        
        # Find match
        try:
            # 1-minute tolerance for floating point matching
            idx_loc = pollutant_df.index.get_indexer([ts_utc], method='nearest', tolerance=timedelta(minutes=1))[0]
            
            if idx_loc != -1:
                row = pollutant_df.iloc[idx_loc]
                
                if 'backgrPollutants' not in step:
                    step['backgrPollutants'] = {}
                
                for pollutant, value in row.items():
                    if pd.notnull(value):
                        step['backgrPollutants'][pollutant] = float(value)
                    
                updated_count += 1
        except KeyError:
            continue

    print(f"\nSaving to: {output_path}")
    if SET_MISSING_TO_ZERO:
        print(f"(Cleaned -999 values in {cleaned_count} total pollutant entries across all timesteps)")
    
    with open(output_path, 'w') as f:
        json.dump(fox_data, f, indent=4)
        
    print(f"Success! Updated {updated_count} timesteps with CSV data.")

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    if IGNORE_YEAR_MISMATCH:
        print("!!! NOTICE: 'IGNORE_YEAR_MISMATCH' is ON. CSV year will be ignored. !!!")

    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Error: CSV file '{INPUT_CSV_FILE}' not found.")
    elif not os.path.exists(INPUT_FOX_FILE):
        print(f"Error: FOX file '{INPUT_FOX_FILE}' not found.")
    else:
        # 1. Determine Target Year from FOX
        target_year = None
        if IGNORE_YEAR_MISMATCH:
            target_year = get_fox_year(INPUT_FOX_FILE)
            if target_year:
                print(f"Detected FOX Simulation Year: {target_year}")
        
        # 2. Process CSV with target year
        df_pollutants = load_and_process_csv(INPUT_CSV_FILE, target_year)
        
        if df_pollutants is not None:
            print("\nSnippet of interpolated data (UTC adjusted):")
            print(df_pollutants.head(3))
            
            update_fox_file(INPUT_FOX_FILE, OUTPUT_FOX_FILE, df_pollutants)
