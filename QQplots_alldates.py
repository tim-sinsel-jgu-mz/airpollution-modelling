import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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
plt.rcParams['legend.frameon'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300


def load_measurements(csv_path):
    """Loads measurements and handles MEZ/MESZ switch by converting to UTC+1."""
    print(f"--- Loading Measurements: {os.path.basename(csv_path)} ---")
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        
        # Check if the time column is named 'Zeit' or 'Datetime'
        time_col = 'Zeit' if 'Zeit' in df.columns else 'Datetime'
        
        df[time_col] = pd.to_datetime(df[time_col], format='%d.%m.%Y %H:%M', errors='coerce')
        df.dropna(subset=[time_col], inplace=True)
        df.set_index(time_col, inplace=True)
        
        # Handle Timezones: Berlin (MEZ/MESZ) -> UTC+1 (Fixed)
        try:
            df = df.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('Etc/GMT-1')
            df.index = df.index.tz_localize(None) 
        except Exception as e:
            print(f"Timezone conversion error: {e}")
            df.index = df.index.tz_localize(None)

        # Standardize column names
        rename_dict = {}
        if 'PM10' in df.columns: rename_dict['PM10'] = 'PM10'
        if 'PM2_5' in df.columns: rename_dict['PM2_5'] = 'PM2.5'
        if 'PM2,5' in df.columns: rename_dict['PM2,5'] = 'PM2.5'
        
        df.rename(columns=rename_dict, inplace=True)
        df = df[['PM10', 'PM2.5']].apply(pd.to_numeric, errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error loading Measurements: {e}")
        return pd.DataFrame()


def load_model_caches(cache_dir, cache_pattern="*.csv"):
    """Loads and combines all cache files matching the pattern."""
    print(f"--- Loading Model Data Caches from: {cache_dir} ---")
    cache_files = glob.glob(os.path.join(cache_dir, cache_pattern))
    
    if not cache_files:
        print("No cache files found!")
        return pd.DataFrame()

    all_dfs = []
    for file in cache_files:
        try:
            # Read cache file. Assumes datetime is in the first column (index 0)
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df.index = df.index.round('1min')
            
            # Resample to hourly to match measurement timestamps
            df_hourly = df.resample('1h').mean()
            all_dfs.append(df_hourly)
            print(f"Loaded {len(df_hourly)} hourly steps from {os.path.basename(file)}")
        except Exception as e:
            print(f"Failed to load {os.path.basename(file)}: {e}")
            
    if not all_dfs:
        return pd.DataFrame()

    # Combine all loaded dataframes and sort by time
    df_combined = pd.concat(all_dfs).sort_index()
    
    # Drop any duplicate timestamps, keeping the first occurrence 
    # (in case caches have overlapping simulation periods)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    return df_combined


def get_incremented_filename(out_dir, base_name):
    """
    Returns a filename with _1, _2, ... appended
    if the file already exists in out_dir.
    """
    name, ext = os.path.splitext(base_name)
    full_path = os.path.join(out_dir, base_name)
    
    if not os.path.exists(full_path):
        return full_path

    counter = 1
    new_name = f"{name}_{counter}{ext}"
    full_path = os.path.join(out_dir, new_name)

    while os.path.exists(full_path):
        counter += 1
        new_name = f"{name}_{counter}{ext}"
        full_path = os.path.join(out_dir, new_name)

    return full_path


def plot_combined_qq(df_meas, df_model, pollutants, out_dir):
    """Generates a Q-Q plot comparing all provided measurements against modeled data."""
    print("--- Generating Q-Q Plots ---")
    
    # Align data by finding intersecting timestamps
    common_idx = df_meas.index.intersection(df_model.index)
    
    # Drop 00:00 hours if necessary (keeping consistent with your previous script)
    if not common_idx.empty:
        common_idx = common_idx[common_idx.hour != 0]

    if common_idx.empty:
        print("Error: No overlapping timestamps found between combined measurements and model caches!")
        print(f"Model Date Range: {df_model.index.min()} to {df_model.index.max()}")
        print(f"Meas Date Range: {df_meas.index.min()} to {df_meas.index.max()}")
        return

    df_meas_aligned = df_meas.loc[common_idx]
    df_model_aligned = df_model.loc[common_idx]
    
    # Create Figure
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    fig.suptitle("Q-Q Plot: Model vs Measured (All Dates)", fontsize=14, fontweight='bold', y=0.98) 
    
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        
        x_raw = df_meas_aligned[pol]
        y_raw = df_model_aligned[pol]
        
        # Mask out NaNs
        mask = x_raw.notna() & y_raw.notna()
        x = x_raw[mask]
        y = y_raw[mask]
        
        if len(x) > 1:
            # Sort arrays to create empirical quantiles
            ax.scatter(np.sort(x), np.sort(y), alpha=0.5, s=20, edgecolors='none', color='#2ca02c')
            
            lim = max(x.max(), y.max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            ax.text(0.5, 0.5, "Insufficient Overlapping Data", ha='center')

        ax.set_title(f"{pol} Q-Q Plot (N={len(x)})", fontweight='bold')
        ax.set_xlabel("Measured Quantiles [µg m$^{-3}$]")
        ax.set_ylabel("Modelled Quantiles [µg m$^{-3}$]")

    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.4, hspace=0.3)
    
    # Save the plots
    png_path = get_incremented_filename(out_dir, "QQplot_alldates.png")
    svg_path = get_incremented_filename(out_dir, "QQplot_alldates.svg")
    
    plt.savefig(png_path)
    plt.savefig(svg_path)
    print(f"Saved Q-Q plots to:\n  - {png_path}\n  - {svg_path}")


if __name__ == "__main__":
    # ==========================================
    # USER INPUTS: Update these paths
    # ==========================================
    
    # The continuous measurement CSV covering all dates
    MEASUREMENT_CSV = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\Berlin_Feinstaub_Messdaten.csv"
    
    # Directory containing all the Data Cache CSVs you want to combine
    CACHE_DIRECTORY = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\QQplots_alldates\area5_4m"
    
    # Where to save the final QQ plot
    OUTPUT_DIRECTORY = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\QQplots_alldates"
    
    # ==========================================
    
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # 1. Load Combined Model Data
    model_combined = load_model_caches(CACHE_DIRECTORY, cache_pattern="*.csv")
    
    # 2. Load Measurements
    meas_df = load_measurements(MEASUREMENT_CSV)
    
    # 3. Generate Q-Q Plots
    if not model_combined.empty and not meas_df.empty:
        plot_combined_qq(meas_df, model_combined, ['PM2.5', 'PM10'], OUTPUT_DIRECTORY)
        print("Script finished successfully.")
    else:
        print("Script failed: Either model caches or measurement data could not be loaded.")