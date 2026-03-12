import os
import glob
import json
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr

# version 2.8.1:
# Now optionally also outputs a CSV table with correlation values for the relationships of sim result concentrations with: measured concentrations, background concentrations, input wind speed, input wind direction.
# Metrics calculated for that: Pearson r, R-squared, Spearmans Rank Correlation

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
            # Extract Wind from the first profile entry (usually 10m)
            wind_prof = ts.get('windProfile', [{}])[0]
            ws = wind_prof.get('wSpdValue', np.nan)
            wd = wind_prof.get('wDirValue', np.nan)

            
            records.append({
                'Datetime': dt,
                'PM10_BG': pol.get('PM10', 0),
                'PM2.5_BG': pol.get('PM25', 0),
                'WindSpeed': float(ws),
                'WindDir': float(wd)
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

    # Correlations
    r = np.corrcoef(x, y)[0, 1]
    r2_pearson = r**2
    rho, _ = spearmanr(x, y)

    # Regression R² (predictive skill)
    r2_reg = r2_score(x, y)
    
    #Means
    mean_obs = np.mean(x)
    mean_mod = np.mean(y)
    
    # Errors
    mae = mean_absolute_error(x, y)
    nmse = np.mean((x - y)**2) / (mean_obs * mean_mod) if (mean_obs * mean_mod) != 0 else np.nan
    rmse = np.sqrt(mean_squared_error(x, y))
    nrmse = rmse / mean_obs if mean_obs != 0 else np.nan

    # Mean Bias
    mean_bias = np.mean(y - x)
    nmb = np.sum(y - x) / np.sum(x) if np.sum(x) != 0 else np.nan

    # Fractional Bias
    fb = 2 * np.mean((y - x) / (y + x))
    
    # FAC2
    ratio = y / x
    fac2 = np.mean((ratio >= 0.5) & (ratio <= 2.0))
    
    # Index of Agreement (Willmott)
    denominator = np.sum((np.abs(y - mean_obs) + np.abs(x - mean_obs))**2)
    ioa = 1 - (np.sum((y - x)**2) / denominator) if denominator != 0 else np.nan

    return {
        "r": r,
        "r2_pearson": r2_pearson,
        "r2_reg": r2_reg,
        "rho": rho,
        "mean_obs": mean_obs,
        "mean_mod": mean_mod,
        "mae": mae,
        "nmse": nmse,
        "rmse": rmse,
        "nrmse": nrmse,
        "mean_bias": mean_bias,
        "nmb": nmb,
        "fb": fb,
        "fac2": fac2,
        "ioa": ioa
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
                f"$r_{{pearson}} = {stats['r']:.2f}$\n"
                f"$r^2_{{pearson}} = {stats['r2_pearson']:.2f}$\n"
                f"Spearman = {stats['rho']:.2f}\n\n"
               # f"$R^2_{{reg}} = {stats['r2_reg']:.2f}$\n"
                f"$Mean_{{obs}} = {stats['mean_obs']:.2f}$\n"
                f"$Mean_{{mod}} = {stats['mean_mod']:.2f}$\n\n"
                f"MAE = {stats['mae']:.2f}\n"
                f"NMSE = {stats['nmse']:.2f}\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"NRMSE = {stats['nrmse']:.2f}\n\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"NMB = {stats['nmb']:.2f}\n"
                f"FB = {stats['fb']:.2f}\n\n"
                f"FAC2 = {stats['fac2']:.2f}\n"
                f"d = {stats['ioa']:.2f}\n"
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
        ax.text(1.25, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top',
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
               bbox_to_anchor=(0.4, -0.02))
    
    # Adjust layout to make room for stats on the right (right=0.75)
    plt.subplots_adjust(right=0.75, left=0.1, bottom=0.1, top=0.92, hspace=0.3)
    
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
            
            # Scatter, 1:1 Line, and Regression Line. And now FAC2 lines
            ax.scatter(x, y, alpha=0.5, s=20, edgecolors='none')
            lim = max(x.max(), y.max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            ax.plot(x, m*x + b, color='#D62728', linewidth=1.5, label='Fit')
            ax.plot([0, lim], [0, 0.5*lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2', zorder=1)
            ax.plot([0, lim], [0, 2*lim], 'k--', alpha=0.2, linewidth=0.8, zorder=1)
            
            # Stats Box (Regression)
            stats_str = (
                f"$y = {m:.2f}x + {b:.2f}$\n"
                f"$r_{{pearson}} = {stats['r']:.2f}$\n"
                f"$r^2_{{pearson}} = {stats['r2_pearson']:.2f}$\n"
                f"Spearman = {stats['rho']:.2f}\n\n"
               # f"$R^2_{{reg}} = {stats['r2_reg']:.2f}$\n"
                f"$Mean_{{obs}} = {stats['mean_obs']:.2f}$\n"
                f"$Mean_{{mod}} = {stats['mean_mod']:.2f}$\n\n"
                f"MAE = {stats['mae']:.2f}\n"
                f"NMSE = {stats['nmse']:.2f}\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"NRMSE = {stats['nrmse']:.2f}\n\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"NMB = {stats['nmb']:.2f}\n"
                f"FB = {stats['fb']:.2f}\n\n"
                f"FAC2 = {stats['fac2']:.2f}\n"
                f"d = {stats['ioa']:.2f}\n"
            )
            
            ax.text(1.1, 0.5, stats_str, transform=ax.transAxes, verticalalignment='center', horizontalalignment='left',
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

    plt.subplots_adjust(right=0.7, left=0.15, wspace=0.4, hspace=0.3)
    
    base_filename = f"Regression_Final_{sim_name}.png"
    save_path = get_incremented_filename(out_dir, base_filename)
    plt.savefig(save_path)
    base_filename = f"Regression_Final_{sim_name}.svg"
    save_path = get_incremented_filename(out_dir, base_filename)
    plt.savefig(save_path)
   
def export_extended_correlations(df_model, df_meas, df_fox, out_csv_path):
    """Calculates Pearson r, R2, and Spearman rho for pollutants vs multiple variables."""
    print("--- Calculating Extended Correlations ---")
    
    # Decompose circular Wind Direction into linear U/V components
    wd_rad = np.radians(df_fox['WindDir'])
    df_fox_ext = df_fox.copy()
    df_fox_ext['WindDir_sin'] = np.sin(wd_rad) # East-West 
    df_fox_ext['WindDir_cos'] = np.cos(wd_rad) # North-South
    
    # Combine everything into one DF for aligned row-by-row calculations
    df_all = pd.concat([
        df_model[['PM10', 'PM2.5']].add_prefix('Mod_'),
        df_meas[['PM10', 'PM2.5']].add_prefix('Meas_'),
        df_fox_ext[['PM10_BG', 'PM2.5_BG', 'WindSpeed', 'WindDir_sin', 'WindDir_cos']]
    ], axis=1)
    
    results = []
    targets = [
        ('Meas_PM10', 'Measured PM10'), ('PM10_BG', 'Background PM10'),
        ('Meas_PM2.5', 'Measured PM2.5'), ('PM2.5_BG', 'Background PM2.5'),
        ('WindSpeed', 'Wind Speed'), 
        ('WindDir_sin', 'Wind Dir (Sine/EW)'), ('WindDir_cos', 'Wind Dir (Cosine/NS)')
    ]

    for pol in ['PM10', 'PM2.5']:
        for target_col, target_label in targets:
            valid = df_all[[f'Mod_{pol}', target_col]].dropna()
            if len(valid) > 1:
                x, y = valid[f'Mod_{pol}'], valid[target_col]
                r = np.corrcoef(x, y)[0, 1]
                rho, _ = spearmanr(x, y)
                results.append({
                    'Sim_Variable': pol,
                    'Correlated_Against': target_label,
                    'Pearson_r': r,
                    'R_Squared': r**2,
                    'Spearman_rho': rho
                })

    pd.DataFrame(results).to_csv(out_csv_path, index=False, sep=';', decimal=',')
    print(f"Stats saved to: {out_csv_path}")
   
if __name__ == "__main__":
    # Paths (Update these)
    csv_file = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\Berlin_Feinstaub_Messdaten.csv"
    fox_file = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\merge7_clean_2024_Jun_Nov_smthWind_realBG2.FOX"
    traffic_file = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\TrafficVolume_LEIPZ1_lineCount2.CSV"
    netcdf_folder = r"Z:\Linde\Pascal\20241106_messstation3_3m_realBG2_lineSrc3\NetCDF"
    base_out_dir = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting"
    cache_dir = os.path.join(base_out_dir, "Data_Cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # --- OPTIONAL EXTENDED CORRELATION CONFIG ---
    CALC_EXTENDED_CORR = True
    ext_corr_filename = f"Extendend_Correlation_Stats_{os.path.basename(netcdf_folder.replace(r"\NetCDF", ""))}.csv"
    ext_corr_path = os.path.join(base_out_dir, ext_corr_filename)
    # --------------------------------------------
    
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
        
        # --- NEW TRIGGER FOR EXTENDED CORRELATIONS ---
        if CALC_EXTENDED_CORR:
            export_extended_correlations(
                model_hourly.loc[common_idx], 
                meas_df.loc[common_idx], 
                fox_df.loc[common_idx], 
                ext_corr_path
            )
        # ---------------------------------------------
        
        print("Processing Complete.")