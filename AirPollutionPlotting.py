import os
import glob
import json
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# version 2.6: 
# Now only selects from hour 01:00 and onward, so that we skip hour 00:00. 
# Hour 00:00, as the initialization stage, can deliver results that differ strongly from the rest of the diurnal profile.
# I have still not fixed the weird r2_score values

# --- PLOTTING STYLE CONFIGURATION ---
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['lines.linewidth'] = 2.5
#plt.rcParams['legend.frameon'] = False     # Clean legend without box
plt.rcParams['savefig.bbox'] = 'tight'     # Ensure nothing is cut off
plt.rcParams['savefig.dpi'] = 300          # High resolution


def load_measurements(csv_path, target_start, target_end):
    """Loads measurements and handles MEZ/MESZ switch by converting to UTC+1."""
    print(f"--- Loading Measurements: {os.path.basename(csv_path)} ---")
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M', errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)
        df.set_index('Datetime', inplace=True)
        
        # Handle Timezones: Berlin (MEZ/MESZ) -> UTC+1 (Fixed)
        # We localize to Berlin then convert to a fixed +01:00 offset
        # Replace the tz_localize block in load_measurements with this:
        try:
            # 'ambiguous=True' assumes the first instance (Daylight Savings) during the fold
            df = df.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('Etc/GMT-1')
            df.index = df.index.tz_localize(None) 
        except Exception as e:
            print(f"Timezone conversion error: {e}")
            # Fallback: if localization fails, treat as naive and force limit
            df.index = df.index.tz_localize(None)

        df.rename(columns={'PM10': 'PM10', 'PM2,5': 'PM2.5'}, inplace=True)
        df = df[['PM10', 'PM2.5']].apply(pd.to_numeric, errors='coerce')
        
        # Limit to Sim Range
        df = df.loc[target_start:target_end]
        return df
    except Exception as e:
        print(f"Error loading Measurements: {e}")
        return pd.DataFrame()

def load_fox_background(fox_path, target_start, target_end):
    """Parses .FOX (JSON) forcing file for background concentrations."""
    print(f"--- Loading FOX Background Data ---")
    try:
        with open(fox_path, 'r') as f:
            data = json.load(f)
        
        records = []
        for ts in data['timestepList']:
            # Fix 2018 year to 2024 (or match sim year)
            dt = pd.to_datetime(ts['date'] + ' ' + ts['time'])
            dt = dt.replace(year=target_start.year) 
            
            pol = ts['backgrPollutants']
            records.append({
                'Datetime': dt,
                'PM10_BG': pol.get('PM10', 0),
                'PM2.5_BG': pol.get('PM25', 0)
            })
            
        df_fox = pd.DataFrame(records).set_index('Datetime').sort_index()
        # Resample to hourly mean
        df_fox = df_fox.resample('1h').mean().loc[target_start:target_end]
        return df_fox
    except Exception as e:
        print(f"Error loading FOX: {e}")
        return pd.DataFrame()

def load_traffic_volume(traffic_path, sim_dates):
    """Loads 24h traffic profile and maps it to all days in the simulation."""
    print(f"--- Loading Traffic Data ---")
    try:
        df_t = pd.read_csv(traffic_path, sep=';')
        df_t['Hour'] = pd.to_datetime(df_t['Time'], format='%H:%M:%S').dt.hour
        df_t.set_index('Hour', inplace=True)
        
        # Expand traffic profile to cover every hour in the sim range
        full_range = pd.date_range(sim_dates.min(), sim_dates.max(), freq='1h')
        traffic_series = pd.DataFrame(index=full_range)
        traffic_series['Traffic'] = traffic_series.index.hour.map(df_t['TrajCount'])
        return traffic_series
    except Exception as e:
        print(f"Error loading Traffic: {e}")
        return pd.DataFrame()

def load_envimet_series(nc_folder_path, x_idx, y_idx, z_idx, cache_dir):
    """Loads NetCDF or Cache, rounding 'crooked' timestamps."""
    nc_files = sorted(glob.glob(os.path.join(nc_folder_path, "*.nc")))
    if not nc_files: return pd.DataFrame(), "Unknown"
    
    sim_name = os.path.splitext(os.path.basename(nc_files[0]))[0]
    cache_file = os.path.join(cache_dir, f"Extracted_{sim_name}_X{x_idx}_Y{y_idx}_Z{z_idx}.csv")
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        # Extraction logic (simplified for brevity, identical to your original logic)
        data_frames = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                pt = ds.isel(GridsI=x_idx, GridsJ=y_idx, GridsK=z_idx)
                chunk = pd.DataFrame({
                    'PM2.5': pt['PM25Conc'].values,
                    'PM10': pt['PM10Conc'].values #ENVIcore added correct PM10Conc variable
                }, index=pd.to_datetime(pt['Time'].values))
                data_frames.append(chunk)
        df = pd.concat(data_frames).sort_index()
        df.to_csv(cache_file)

    # CRITICAL: Round crooked timestamps to nearest minute
    df.index = df.index.round('1min')
    return df, sim_name

def calculate_statistics(x, y):
    """
    Calculates evaluation statistics between measured (x) and modelled (y).
    Returns dictionary of metrics.
    """
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    # Pearson correlation
    r = np.corrcoef(x, y)[0, 1]
    r2_pearson = r**2

    # Regression R² (predictive skill)
    r2_reg = r2_score(x, y)
    
    #Means
    mean_obs = np.mean(x)
    mean_mod = np.mean(y)
    
    # Errors
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))

    # Mean Bias
    mean_bias = np.mean(y - x)

    # Fractional Bias
    fb = 2 * np.mean((y - x) / (y + x))

    return {
        "r": r,
        "r2_pearson": r2_pearson,
        "r2_reg": r2_reg,
        "mean_obs": mean_obs,
        "mean_mod": mean_mod,
        "mae": mae,
        "rmse": rmse,
        "mean_bias": mean_bias,
        "fb": fb
    }

def get_incremented_filename(out_dir, base_name):
    """
    Returns a filename with _001, _002, ... appended
    if the file already exists in out_dir.
    """
    name, ext = os.path.splitext(base_name)
    counter = 1

    new_name = f"{name}_{counter}{ext}"
    full_path = os.path.join(out_dir, new_name)

    while os.path.exists(full_path):
        counter += 1
        new_name = f"{name}_{counter}{ext}"
        full_path = os.path.join(out_dir, new_name)

    return full_path

def plot_final_results(df_meas, df_model, df_fox, df_traffic, pollutants, out_dir, sim_name, coords):
    """Generates 2x1 Diurnal and 2x1 Regression plots with stats boxes and fixed gaps."""
    date_str = df_model.index[0].strftime('%d.%m.%Y')
    x_idx, y_idx, z_idx = coords
    header_str = f"Simulation: {sim_name} | Grid: X={x_idx}, Y={y_idx}, Z={z_idx}"
    
    # 1. DIURNAL PLOT (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(header_str, fontsize=14, fontweight='bold', y=0.98) # Global Title
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        
        # Calculate Stats for the box (on the full hourly time series)
        stats = calculate_statistics(df_meas[pol], df_model[pol])

        if stats is not None:
            stats_str = (
                f"$r = {stats['r']:.2f}$\n"
                f"$r^2_{{corr}} = {stats['r2_pearson']:.2f}$\n"
                f"$R^2_{{reg}} = {stats['r2_reg']:.2f}$\n"
                f"$Mean_{{obs}} = {stats['mean_obs']:.2f}$\n"
                f"$Mean_{{mod}} = {stats['mean_mod']:.2f}$\n"
                f"MAE = {stats['mae']:.2f}\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"FB = {stats['fb']:.2f}"
            )
        else:
            stats_str = "No Data"

        # Diurnal Averages - REINDEX to range(24) to prevent lines connecting across gaps
        m_diurnal = df_meas.groupby(df_meas.index.hour)[pol].mean().reindex(range(24))
        s_diurnal = df_model.groupby(df_model.index.hour)[pol].mean().reindex(range(24))
        f_diurnal = df_fox.groupby(df_fox.index.hour)[f"{pol}_BG"].mean().reindex(range(24))
        t_diurnal = df_traffic.groupby(df_traffic.index.hour)['Traffic'].mean().reindex(range(24))

        # Traffic Fill (Secondary Axis)
        ax2 = ax.twinx()
        ax2.fill_between(t_diurnal.index, 0, t_diurnal, color='gray', alpha=0.15, label='Traffic')
        ax2.set_ylabel("Traffic Vol. [Veh./h]", color='gray', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, None)

        # Pollution Lines
        ax.plot(m_diurnal.index, m_diurnal, 'k-', label='Measured', zorder=5)
        ax.plot(s_diurnal.index, s_diurnal, '#D62728', label='Modelled', zorder=6)
        ax.plot(f_diurnal.index, f_diurnal, color='gray', linestyle='--', label='Background (FOX)', zorder=4)

        # Stats Box (Diurnal)
        ax.text(0.03, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(f"Diurnal Cycle: {pol}", fontweight='bold')
        ax.set_ylabel("Conc. [µg m$^{-3}$]")
        ax.set_xlabel("Hour of Day")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_ylim(0, None) # Force Y-axis to start at 0
        ### if i == 0: ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, which='major', linestyle=':', alpha=0.6)
        
        # Collect legend handles from first axis
        if i == 0:
            handles_main, labels_main = ax.get_legend_handles_labels()
            handles_sec, labels_sec = ax2.get_legend_handles_labels()

    # Combine (avoid duplicates if needed)
    handles = handles_main + handles_sec
    labels = labels_main + labels_sec

    # Global legend underneath both subplots
    fig.legend(handles, labels,
               loc='lower center',
               ncol=len(labels),
               frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    base_filename = f"Diurnal_Final_{sim_name}.png"
    save_path = get_incremented_filename(out_dir, base_filename)
    plt.savefig(save_path)
    base_filename = f"Diurnal_Final_{sim_name}.svg"
    save_path = get_incremented_filename(out_dir, base_filename)
    plt.savefig(save_path)
    
    # 2. REGRESSION PLOT (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    fig.suptitle(header_str, fontsize=12, fontweight='bold', y=0.98) # Global Title
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        x_raw, y_raw = df_meas[pol], df_model[pol]
        mask = x_raw.notna() & y_raw.notna()
        x, y = x_raw[mask], y_raw[mask]
        
        if len(x) > 1:
            stats = calculate_statistics(x, y)
            m, b = np.polyfit(x, y, 1)
            
            # Scatter, 1:1 Line, and Regression Line
            ax.scatter(x, y, alpha=0.5, s=20, edgecolors='none')
            lim = max(x.max(), y.max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            ax.plot(x, m*x + b, color='#D62728', linewidth=1.5, label='Fit')
            
            # Stats Box (Regression)
            stats_str = (
                f"$y = {m:.2f}x + {b:.2f}$\n"
                f"$r = {stats['r']:.2f}$\n"
                f"$r^2_{{corr}} = {stats['r2_pearson']:.2f}$\n"
                f"$R^2_{{reg}} = {stats['r2_reg']:.2f}$\n"
                f"$Mean_{{obs}} = {stats['mean_obs']:.2f}$\n"
                f"$Mean_{{mod}} = {stats['mean_mod']:.2f}$\n"
                f"MAE = {stats['mae']:.2f}\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"FB = {stats['fb']:.2f}"
            )
            
            ax.text(0.95, 0.05, stats_str, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')

        ax.set_title(f"{pol} Regression", fontweight='bold')
        ax.set_xlabel("Measured [µg m$^{-3}$]")
        ax.set_ylabel("Modelled [µg m$^{-3}$]")

    plt.tight_layout()
    base_filename = f"Regression_Final_{sim_name}.png"
    save_path = get_incremented_filename(out_dir, base_filename)
    plt.savefig(save_path)
    base_filename = f"Regression_Final_{sim_name}.svg"
    save_path = get_incremented_filename(out_dir, base_filename)
    plt.savefig(save_path)
    
if __name__ == "__main__":
    # Paths (Update these)
    csv_file = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\Berlin_Feinstaub_Messdaten.csv"
    fox_file = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\merge7_clean_2024_Jun_Nov_smthWind_realBG2.FOX"
    traffic_file = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\TrafficVolume_LEIPZ1_lineCount2.CSV"
    netcdf_folder = r"Z:\Linde\Pascal\20241106_messstation3_3m_realBG2_lineSrc3\NetCDF"
    base_out_dir = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting"
    cache_dir = os.path.join(base_out_dir, "Data_Cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define coordinates once here
    target_coords = (134, 104, 3) #area5_4m: 101, 92, 3      #messstation3_3m: 134, 104, 3
    
    # 1. Load Model Data first to define time range
    model_df, sim_name = load_envimet_series(netcdf_folder, *target_coords, cache_dir) 
    # --- UPDATED SECTION ---
    # Define start as the second hour (index.min + 1 hour) to skip initialization
    t_start = model_df.index.min() + pd.Timedelta(hours=1) 
    t_end = model_df.index.max()
    # Slice the model dataframe immediately to the new range
    model_df = model_df.loc[t_start:t_end]
    # -----------------------

    # 2. Load other files limited to sim range
    meas_df = load_measurements(csv_file, t_start, t_end)
    fox_df = load_fox_background(fox_file, t_start, t_end)
    traffic_df = load_traffic_volume(traffic_file, model_df.index)

    # 3. Align all to hourly snapshots
    # Resample model to hourly to match measurements
    model_hourly = model_df.resample('1h').mean()
    
    # Final cleanup: Ensure all dataframes have the exact same index
    common_idx = model_hourly.index.intersection(meas_df.index)

    if common_idx.empty:
        print("Error: No overlapping timestamps found between measurements and model!")
    else:
        plot_final_results(meas_df.loc[common_idx], model_hourly.loc[common_idx], 
                           fox_df.loc[common_idx], traffic_df.loc[common_idx], 
                           ['PM2.5', 'PM10'], base_out_dir, sim_name, target_coords)
        print("Processing Complete.")