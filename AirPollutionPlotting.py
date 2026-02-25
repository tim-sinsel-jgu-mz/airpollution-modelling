import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# --- PLOTTING STYLE CONFIGURATION ---
# rigorous publication quality settings
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
plt.rcParams['legend.frameon'] = False     # Clean legend without box
plt.rcParams['savefig.bbox'] = 'tight'     # Ensure nothing is cut off
plt.rcParams['savefig.dpi'] = 300          # High resolution

def load_measurements(csv_path):
    """
    Loads and parses the measurement CSV file.
    Expects format: Datetime;PM10;PM2,5;NO2;NO;NOX
    """
    print(f"--- Loading Measurements: {os.path.basename(csv_path)} ---")
    try:
        # Load CSV (handle semicolon delimiter and comma decimal if present)
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        
        # Parse Datetime (Format: 01.06.2024 00:00)
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M', errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)
        df.set_index('Datetime', inplace=True)
        
        # Rename columns to standard internal names
        rename_map = {
            'PM10': 'PM10',
            'PM2,5': 'PM2.5',
            'NO2': 'NO2',
            'NO': 'NO',
            'NOX': 'NOx'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        print(f"Loaded {len(df)} timestamps. Range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def load_envimet_netcdf_series(nc_folder_path, x_idx, y_idx, z_idx, cache_dir):
    """
    Loads specific grid point data. 
    Checks for Cached CSV first. If not found, loads from NetCDF and saves to Cache.
    """
    print(f"--- Loading Model Data ---")
    
    # 1. Identify the Simulation Name (based on the first .nc file name)
    nc_files = sorted(glob.glob(os.path.join(nc_folder_path, "*.nc")))
    if not nc_files:
        print("Error: No .nc files found in directory.")
        return pd.DataFrame(), "Unknown"
        
    # Base name for the simulation series
    sim_name = os.path.splitext(os.path.basename(nc_files[0]))[0]
    
    # Construct Cache Filename
    cache_file = os.path.join(cache_dir, f"Extracted_{sim_name}_X{x_idx}_Y{y_idx}_Z{z_idx}.csv")
    
    # 2. CHECK CACHE
    if os.path.exists(cache_file):
        print(f"Cache found! Loading from: {cache_file}")
        df_model = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df_model, sim_name
    
    # 3. EXTRACT FROM NETCDF (if no cache)
    print(f"No cache found. Extracting from NetCDF files... (This may take a moment)")
    print(f"Target Grid: X={x_idx}, Y={y_idx}, Z={z_idx}")
    
    data_frames = []
    
    # Map NetCDF var name -> Internal DataFrame name
    var_map = {
        'PM25Conc': 'PM2.5',
        'PMCoarseConc': 'PM_Coarse',
        'NO2Conc': 'NO2',
        'NOConc': 'NO',
        'NOxConc': 'NOx'
    }
    
    for f in nc_files:
        try:
            with xr.open_dataset(f, decode_times=True) as ds:
                # Check dims
                if 'GridsI' not in ds.dims or 'GridsJ' not in ds.dims:
                    continue
                
                # Select point
                point_ds = ds.isel(GridsI=x_idx, GridsJ=y_idx, GridsK=z_idx)
                
                # Extract data
                data = {}
                data['Datetime'] = point_ds['Time'].values
                
                for nc_var, df_var in var_map.items():
                    if nc_var in point_ds:
                        data[df_var] = point_ds[nc_var].values
                
                df_chunk = pd.DataFrame(data)
                df_chunk.set_index('Datetime', inplace=True)
                data_frames.append(df_chunk)
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not data_frames:
        return pd.DataFrame(), sim_name
        
    # Concatenate and Sort
    df_model = pd.concat(data_frames).sort_index()
    
    # Calculate PM10
    if 'PM2.5' in df_model.columns and 'PM_Coarse' in df_model.columns:
        df_model['PM10'] = df_model['PM2.5'] + df_model['PM_Coarse']
        
    # 4. SAVE TO CACHE
    print(f"Saving extracted data to: {cache_file}")
    df_model.to_csv(cache_file)
    
    return df_model, sim_name

def align_to_hourly_snapshots(df_meas, df_model):
    """
    Aligns Model data to Measurement timestamps (Hourly).
    Disregards intermediate model steps (e.g. 10:10, 10:20), taking only the snapshot
    closest to the measurement time (e.g. 10:00).
    """
    if df_meas.empty or df_model.empty:
        return None, None
        
    print("--- Aligning Data (Hourly Snapshots) ---")
    
    # 1. Sort Indices
    df_meas = df_meas.sort_index()
    df_model = df_model.sort_index()
    
    # 2. Restrict Measurement data to the Model's general time range first
    # (Improves performance before reindexing)
    t_min = df_model.index.min() - pd.Timedelta(minutes=15)
    t_max = df_model.index.max() + pd.Timedelta(minutes=15)
    
    mask_time = (df_meas.index >= t_min) & (df_meas.index <= t_max)
    df_meas_cut = df_meas.loc[mask_time].copy()
    
    if df_meas_cut.empty:
        print("Error: No measurements found in model time range.")
        return None, None

    # 3. ALIGNMENT
    # Reindex the MODEL to match the MEASUREMENT timestamps.
    # method='nearest': Picks the closest model timestamp (e.g. Model 10:00 for Meas 10:00)
    # tolerance='5min': Ensures we don't match if gaps are too huge.
    try:
        df_model_aligned = df_model.reindex(
            df_meas_cut.index, 
            method='nearest', 
            tolerance=pd.Timedelta('5min')
        )
    except Exception as e:
        print(f"Alignment Error: {e}")
        return None, None

    # 4. Remove rows where alignment found no data (NaNs)
    # Check a key column like 'PM2.5'
    valid_mask = df_model_aligned['PM2.5'].notna()
    
    df_meas_final = df_meas_cut.loc[valid_mask]
    df_model_final = df_model_aligned.loc[valid_mask]
    
    print(f"Alignment matched {len(df_meas_final)} hourly points.")
    
    return df_meas_final, df_model_final

def plot_diurnal_2x2(df_meas, df_model, pollutants, out_dir, sim_name):
    """
    Generates a 2x2 Grid of Diurnal Cycle Plots with Embedded Stats.
    Saves directly to out_dir.
    """
    # Setup 2x2 Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Get Date for Title
    date_str = df_model.index[0].strftime('%d.%m.%Y')
    fig.suptitle(f"Diurnal Cycles - {date_str} - {sim_name}", fontsize=16, y=0.95)
    
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        
        if pol not in df_meas.columns or pol not in df_model.columns:
            ax.text(0.5, 0.5, f"{pol} missing", ha='center', va='center')
            continue
        
        # --- CALCULATE STATS (on the full hourly time series) ---
        # We calculate stats on the aligned data points, not the diurnal average curves
        x = df_meas[pol]
        y = df_model[pol]
        mask = ~np.isnan(x) & ~np.isnan(y)
        
        if np.sum(mask) > 1:
            x_valid, y_valid = x[mask], y[mask]
            rmse = np.sqrt(mean_squared_error(x_valid, y_valid))
            # Pearson R2
            r_sq = np.corrcoef(x_valid, y_valid)[0, 1]**2
            stats_str = f"$R^2={r_sq:.2f}$\n$RMSE={rmse:.1f}$"
        else:
            stats_str = "No Data"

        # --- PREPARE DIURNAL DATA ---
        meas_diurnal = df_meas.groupby(df_meas.index.hour)[pol].mean()
        model_diurnal = df_model.groupby(df_model.index.hour)[pol].mean()
        
        # --- PLOT ---
        ax.plot(meas_diurnal.index, meas_diurnal, 
                label='Measured', color='black', linewidth=1, linestyle='-')
        
        ax.plot(model_diurnal.index, model_diurnal, 
                label='Modelled', color='#D62728', linewidth=1, linestyle='-')
        
        # Add Stats Box
        ax.text(0.04, 0.94, stats_str, transform=ax.transAxes, 
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
        
        # Styling
        ax.set_title(pol, fontsize=14, pad=10)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Conc. [µg m$^{{-3}}$]")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which='major', linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=10)

    # Hide unused subplots
    for j in range(len(pollutants), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save directly to output directory
    filename = f"Diurnal_Matrix_{sim_name}.png"
    plt.savefig(os.path.join(out_dir, filename))
    filename = f"Diurnal_Matrix_{sim_name}.svg"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_regression_2x2(df_meas, df_model, pollutants, out_dir, sim_name):
    """
    Generates a 2x2 Grid of Regression Plots.
    Saves directly to out_dir.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    date_str = df_model.index[0].strftime('%d.%m.%Y')
    fig.suptitle(f"Model Evaluation - {date_str} - {sim_name}", fontsize=16, y=0.95)
    
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        
        if pol not in df_meas.columns or pol not in df_model.columns:
            ax.text(0.5, 0.5, f"{pol} missing", ha='center')
            continue
            
        x_full = df_meas[pol]
        y_full = df_model[pol]
        mask = ~np.isnan(x_full) & ~np.isnan(y_full)
        x = x_full[mask]
        y = y_full[mask]
        
        if len(x) < 2:
            ax.text(0.5, 0.5, "Not enough data", ha='center')
            continue

        # Stats
        rmse = np.sqrt(mean_squared_error(x, y))
        r_sq = np.corrcoef(x, y)[0, 1]**2
        
        # Limits
        limit = max(x.max(), y.max()) * 1.1
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect('equal')
        
        # 1:1 Line
        ax.plot([0, limit], [0, limit], color='gray', linestyle=':', linewidth=1.5, zorder=1)
        
        # Scatter
        ax.scatter(x, y, alpha=0.5, c='#1f77b4', s=15, edgecolors='none', zorder=2)
        
        # Regression Line
        try:
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m*x + b, color='#D62728', linewidth=1, linestyle='-', zorder=3)
            
            stats_text = (f"$y = {m:.2f}x + {b:.2f}$\n"
                          f"$R^2 = {r_sq:.2f}$\n"
                          f"RMSE $= {rmse:.2f}$")
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        except:
            pass
        
        ax.set_title(pol, fontsize=14)
        ax.set_xlabel("Measured [µg m$^{{-3}}$]")
        ax.set_ylabel("Modelled [µg m$^{{-3}}$]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    for j in range(len(pollutants), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save directly to output directory
    filename = f"Regression_Matrix_{sim_name}.png"
    plt.savefig(os.path.join(out_dir, filename))
    filename = f"Regression_Matrix_{sim_name}.svg"
    plt.savefig(os.path.join(out_dir, filename))    
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # --- CONFIGURATION ---
    # Update these paths as needed
    csv_file = r"D:\enviprojects\Berlin_Mehringdamm_Base\Berlin_Feinstaub_Messdaten.csv"
    netcdf_folder = r"X:\Linde\Pascal\20240715_messstation3_3m_realBG\NetCDF"
    
    base_out_dir = r"D:\Berlin_Mehringdamm_CompResults\Base"
    os.makedirs(base_out_dir, exist_ok=True)
    
    cache_dir = os.path.join(base_out_dir, "Data_Cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    target_x, target_y, target_z = 134, 104, 3
    
    # 1. Load Measurements
    meas_df = load_measurements(csv_file)
    
    # 2. Load Model Data
    model_df, sim_name = load_envimet_netcdf_series(netcdf_folder, target_x, target_y, target_z, cache_dir)
    
    if not meas_df.empty and not model_df.empty:
        
        # 3. Align Data (HOURLY SNAPSHOTS ONLY)
        # We now match Model strictly to Measurement timestamps
        aligned_meas, aligned_model = align_to_hourly_snapshots(meas_df, model_df)
        
        if aligned_meas is not None and not aligned_meas.empty:
            pollutants = ['PM2.5', 'PM10', 'NO2', 'NO']
            
            print(f"\nGenerating Matrix plots for: {sim_name}")
            
            # 4. Generate Plots
            plot_diurnal_2x2(aligned_meas, aligned_model, pollutants, base_out_dir, sim_name)
            plot_regression_2x2(aligned_meas, aligned_model, pollutants, base_out_dir, sim_name)
            
            print(f"Done. Results in: {os.path.abspath(base_out_dir)}")
        else:
            print("Alignment failed. No overlapping data found.")
    else:
        print("Data loading failed.")