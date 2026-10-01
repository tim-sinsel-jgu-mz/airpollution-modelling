import json
import math
import pandas as pd
import numpy as np

def safe_float(val, default=0.0):
    """Ensures a safe conversion to float to prevent NaNs in the JSON."""
    try:
        v = float(val)
        if math.isnan(v):
            return default
        return v
    except:
        return default

def calc_specific_humidity(tair_c, rh, p_hpa=1013.25):
    """Calculates specific humidity in g/kg based on air temperature and relative humidity."""
    es = 6.112 * math.exp((17.67 * tair_c) / (tair_c + 243.5))
    e = es * (rh / 100.0)
    q = (0.622 * e) / (p_hpa - (0.378 * e)) * 1000
    return q

def compute_wind_vectors(spd_series, dir_series):
    """Calculates U and V wind components."""
    wd_rad = np.radians(dir_series)
    u = -spd_series * np.sin(wd_rad)
    v = -spd_series * np.cos(wd_rad)
    return u, v

def retrieve_wind_from_vectors(u_mean, v_mean):
    """Recalculates wind speed and direction from mean U and V components."""
    w_spd = np.sqrt(u_mean**2 + v_mean**2)
    w_dir_rad = np.arctan2(-u_mean, -v_mean)
    w_dir = (np.degrees(w_dir_rad) + 360) % 360
    return w_spd, w_dir

def clean_and_parse_numeric(series):
    """Ensures that string numbers with commas are safely converted to floats."""
    if series.dtype == object:
        return series.astype(str).str.replace(',', '.').astype(float)
    return series.astype(float)

def main():
    # ==========================================
    # 1. Define Input and Output File Paths
    # ==========================================
    csv_meteo_path = r'D:\enviprojects\Berlin_Feinstaub_LineSource_newNOsplit\21_Moabit-1_2024_condensed_total_longPeriod_cleaned.csv'
    csv_pollutant_path = r'D:\enviprojects\Berlin_Feinstaub_LineSource_newNOsplit\backgroundConc_median_new.csv'
    csv_solar_path = r'D:\enviprojects\Berlin_Feinstaub_LineSource_newNOsplit\solardata.csv' # Korrigierter Pfad
    output_fox_path = r'D:\enviprojects\Berlin_Pollutants_LeipzigerStr\final.fox'
    
    # ==========================================
    # 2. Process High-Resolution Meteo Data (MEVIS - UTC+1)
    # ==========================================
    print("Reading and processing meteo data...")
    df_meteo = pd.read_csv(csv_meteo_path, sep=';')
    df_meteo.columns = df_meteo.columns.str.strip()
    
    df_meteo['Datetime'] = pd.to_datetime(df_meteo['Date'] + ' ' + df_meteo['Time'], format='%d.%m.%Y %H:%M:%S')
    
    df_meteo['Tair 200 cm'] = clean_and_parse_numeric(df_meteo['Tair 200 cm'])
    df_meteo['RelHum'] = clean_and_parse_numeric(df_meteo['RelHum'])
    df_meteo['WindSpd Mean'] = clean_and_parse_numeric(df_meteo['WindSpd Mean'])
    df_meteo['Wind Dir Mean'] = clean_and_parse_numeric(df_meteo['Wind Dir'])
    
    if 'Precipitation' in df_meteo.columns:
        df_meteo['Precipitation'] = clean_and_parse_numeric(df_meteo['Precipitation']).fillna(0.0)
    else:
        df_meteo['Precipitation'] = 0.0

    # Calculate U/V Vectors prior to aggregation
    df_meteo['U'], df_meteo['V'] = compute_wind_vectors(df_meteo['WindSpd Mean'], df_meteo['Wind Dir Mean'])
    
    # Group into 10-minute intervals
    df_meteo['agg_time'] = (df_meteo['Datetime'] + pd.Timedelta(minutes=5)).dt.floor('10min')
    df_10min = df_meteo.groupby('agg_time').mean(numeric_only=True).reset_index()
    
    # Smooth Wind Vectors
    df_10min['U_smooth'] = df_10min['U'].rolling(window=12, center=True, min_periods=1).mean()
    df_10min['V_smooth'] = df_10min['V'].rolling(window=12, center=True, min_periods=1).mean()
    
    # Retrieve base scalar 10-min speeds & directions
    w_spd_10min, _ = retrieve_wind_from_vectors(df_10min['U'], df_10min['V'])
    _, w_dir_smooth = retrieve_wind_from_vectors(df_10min['U_smooth'], df_10min['V_smooth'])
    
    # Smooth scalar speed and clip to 0.3 m/s minimum
    w_spd_smooth = w_spd_10min.rolling(window=12, center=True, min_periods=1).mean()
    df_10min['Wind Dir Final'] = w_dir_smooth
    df_10min['WindSpd Mean'] = w_spd_smooth.clip(lower=0.3)

    # ==========================================
    # 3. Process Background Pollutants
    # ==========================================
    print("Reading and interpolating background pollutant data...")
    df_poll = pd.read_csv(csv_pollutant_path, sep=';', decimal=',')
    df_poll.columns = df_poll.columns.str.strip()
    df_poll['Datetime'] = pd.to_datetime(df_poll['Datetime'], format='%d.%m.%Y %H:%M')
    
    df_poll.set_index('Datetime', inplace=True)
    df_poll_10min = df_poll.resample('10min').interpolate(method='time')
    
    # ==========================================
    # 4. Read DWD Solar Data CSV & Build Radiation Dictionary (Shift UTC -> UTC+1)
    # ==========================================
    print("Reading solar radiation data and building exact-match lookup...")
    df_solar = pd.read_csv(csv_solar_path, sep=';', skipinitialspace=True)
    df_solar.columns = df_solar.columns.str.strip()
    
    # Datetime parsen aus "YYYYMMDDHHMM" (UTC) und um +1 Stunde verschieben auf MEZ (UTC+1)
    df_solar['Datetime'] = pd.to_datetime(df_solar['MESS_DATUM (YYYYMMDDHHMM)'].astype(str), format='%Y%m%d%H%M') + pd.Timedelta(hours=1)
    
    # Missing values behandeln (-999 durch 0 ersetzen)
    df_solar['DS_10'] = pd.to_numeric(df_solar['DS_10'], errors='coerce').replace(-999, 0).fillna(0)
    df_solar['GS_10'] = pd.to_numeric(df_solar['GS_10'], errors='coerce').replace(-999, 0).fillna(0)
    
    # J/cm^2 in W/m^2 umrechnen (Faktor: 10000 / 600 = 16.6667)
    df_solar['swDif'] = df_solar['DS_10'] * (10000 / 600.0)
    
    # swDir ist Globalstrahlung - Diffusstrahlung
    df_solar['swDir'] = (df_solar['GS_10'] - df_solar['DS_10']) * (10000 / 600.0)
    df_solar['swDir'] = df_solar['swDir'].clip(lower=0) # Absicherung gegen negative Werte
    
    # Dictionary bauen für fehlerfreies String-Matching der Timesteps
    solar_lookup = {}
    for _, row in df_solar.iterrows():
        dt_str = row['Datetime'].strftime("%Y-%m-%d %H:%M:%S")
        solar_lookup[dt_str] = {
            'swDir': row['swDir'],
            'swDif': row['swDif']
        }
    
    # ==========================================
    # 5. Build Output File
    # ==========================================
    print("Building final FOX object...")
    output_timestep_list = []
    
    for idx, row in df_10min.iterrows():
        target_time = row['agg_time']
        
        # --- A. Exact String Match for Radiation Data ---
        target_date_str = target_time.strftime("%Y-%m-%d")
        target_time_str = target_time.strftime("%H:%M:%S")
        target_key = f"{target_date_str} {target_time_str}"
        
        # Look up the radiation data (W/m^2), fallback to 0.0 if timestep is completely missing
        rad_data = solar_lookup.get(target_key, {'swDir': 0.0, 'swDif': 0.0})
        swDir_val = safe_float(rad_data['swDir'])
        swDif_val = safe_float(rad_data['swDif'])
        
        # --- B. Get temporally matched/interpolated pollutant data ---
        poll_idx = df_poll_10min.index.get_indexer([target_time], method='nearest')[0]
        poll_row = df_poll_10min.iloc[poll_idx]
        
        pm10 = safe_float(poll_row.get('Feinstaub (PM10)', 0.0))
        pm25 = safe_float(poll_row.get('Feinstaub (PM2,5)', 0.0))
        no2  = safe_float(poll_row.get('Stickstoffdioxid (NO2)', 0.0))
        no   = safe_float(poll_row.get('Stickstoffmonoxid (NO)', 0.0))
        o3   = safe_float(poll_row.get('Ozon', 0.0))
        
        # --- C. Format Met Profiles ---
        tair_c = safe_float(row['Tair 200 cm'])
        tair_k = tair_c + 273.15 
        q_hum = calc_specific_humidity(tair_c, safe_float(row['RelHum']))
        
        precip = safe_float(row['Precipitation'])
        wspd = safe_float(row['WindSpd Mean'])
        wdir = safe_float(row['Wind Dir Final'])
        
        # --- D. Assemble Timestep Dictionary ---
        step_dict = {
            "date": target_date_str,
            "time": target_time_str,
            "radSurface": 0,
            "dataSrc": 7,
            "swDir": swDir_val,
            "swDif": swDif_val,
            "lwRad": 0,
            "lClouds": -999,
            "mClouds": -999,
            "hClouds": -999,
            "precipitation": precip,
            "backgrPollutants": {
                "User": 0.0,
                "NO": no,
                "NO2": no2,
                "O3": o3,
                "PM10": pm10,
                "PM25": pm25,
                "C5H8": 0.0,
                "RO2": 0.0,
                "HO2": 0.0
            },
            "tProfile": [{"height": 2, "value": tair_k}],
            "qProfile": [{"height": 2, "value": q_hum}],
            "windProfile": [{"height": 3, "wSpdValue": wspd, "wDirValue": wdir}],
            "pProfile": [{"height": 2, "value": 1013.25}]
        }
        
        output_timestep_list.append(step_dict)

    # Build the final top-level structure autonomously
    output_fox_data = {
        "fileType": "ENVI-met JSON Forcing File",
        "metaData": {
            "fileDescription": "Generated merged forcing data with DWD Solar",
            "version": 4
        },
        "locationData": {
            "name": "<Unknown Location>",
            "lat": 0,
            "lon": 0,
            "timezoneLon": 0,
            "altitude": 0
        },
        "timestepList": output_timestep_list
    }
    
    # Save the file
    with open(output_fox_path, 'w', encoding='utf-8') as f:
        json.dump(output_fox_data, f, indent=4)
        
    print(f"Successfully generated {output_fox_path} covering {len(output_timestep_list)} time steps.")

if __name__ == '__main__':
    main()