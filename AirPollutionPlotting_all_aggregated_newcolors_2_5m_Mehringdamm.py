import os
import glob
import json
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import skill_metrics as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress
import cmcrameri.cm as cmc 

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
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

# --- HELPER FUNCTIONS ---
def smooth_data(series, num_points=300, allow_negative=False):
    series = series.dropna()
    if len(series) < 4:
        return series.index.values, series.values
    x = series.index.values
    y = series.values
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    
    # Only clip at zero if it's an absolute concentration plot
    if not allow_negative:
        y_smooth = np.maximum(y_smooth, 0)
        
    return x_smooth, y_smooth

def calculate_statistics(base_df, method_df, pol):
    # FAST: Removed sort_values! Data is inherently aligned by extraction.
    x = base_df[pol].values
    y = method_df[pol].values
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    
    if len(x) < 2: return None
    r = np.corrcoef(x, y)[0, 1]
    return {
        "r2_pearson": r2_score(x, y),
        "rmse": np.sqrt(mean_squared_error(x, y)),
        "mae": mean_absolute_error(x, y)
    }


# --- DATA EXTRACTION: JSON MASKING & NETCDF ---
def load_mask_from_inx(inx_path):
    print(f"   -> Parsing JSON INX for mask: {os.path.basename(inx_path)}")
    try:
        with open(inx_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {inx_path}: {e}")
        return None

    try:
        soil_data = data['envimetDatafile']['spatialData2D']['soilProfiles']['data']
        bldg_data = data['envimetDatafile']['spatialData2D']['buildings']['top']
        
        size_x = len(soil_data)
        size_y = len(soil_data[0]) if size_x > 0 else 0
        mask_array = np.zeros((size_y, size_x), dtype=bool)
        
        for x in range(size_x):
            for y in range(size_y):
                if 10 <= x < (size_x - 10) and 10 <= y < (size_y - 10):
                    soil_id = str(soil_data[x][y]).strip()
                    bldg_height = float(bldg_data[x][y])
                    if soil_id == "AAAAAA" and bldg_height == 0.0:
                        mask_array[y, x] = True
                        
        valid_cells = np.sum(mask_array)
        print(f"      Mask generated successfully. Found {valid_cells} valid horizontal grid cells.")
        return mask_array
    except KeyError as e:
        print(f"KeyError: Could not find tag {e} in JSON.")
        return None

def load_envimet_mask_series(nc_folder_path, boolean_mask, z_idx, cache_file):
    # Force a new cache file name to prevent loading the old, bugged data
    cache_pkl = cache_file.replace('.pkl', '_filtered.pkl')
    if os.path.exists(cache_pkl):
        return pd.read_pickle(cache_pkl)

    nc_files = sorted(glob.glob(os.path.join(nc_folder_path, "*.nc")))
    if not nc_files: 
        return pd.DataFrame()
    
    data_frames = []
    num_valid_cells = np.sum(boolean_mask)
    
    var_mapping = {'PM25Conc': 'PM2.5', 'PM10Conc': 'PM10', 'NOConc': 'NO', 'NO2Conc': 'NO2'}
    vars_to_keep = list(var_mapping.keys())
    
    for f in nc_files:
        print(f"      -> Reading {os.path.basename(f)} (Ultra-Fast Vectorized)...")
        with xr.open_dataset(f) as ds:
            avail_vars = [v for v in vars_to_keep if v in ds.data_vars]
            if not avail_vars: continue
                
            ds_z = ds[avail_vars].isel(GridsK=z_idx).compute()
            times = np.atleast_1d(ds['Time'].values)
            num_times = len(times)
            
            time_col = np.repeat(times, num_valid_cells)
            cell_col = np.tile(np.arange(num_valid_cells), num_times)
            
            df_step = pd.DataFrame({'Time': time_col, 'Cell_ID': cell_col})
            
            for nc_var, df_col in var_mapping.items():
                if nc_var in ds_z:
                    da = ds_z[nc_var]
                    dims_order = [d for d in ['Time', 'GridsJ', 'GridsI'] if d in da.dims]
                    da = da.transpose(*dims_order, ...)
                    arr = da.values
                    
                    if 'Time' in da.dims:
                        df_step[df_col] = arr[:, boolean_mask].flatten()
                    else:
                        df_step[df_col] = np.tile(arr[boolean_mask], num_times)
                else:
                    df_step[df_col] = np.nan
                    
            data_frames.append(df_step)
                
    if not data_frames: return pd.DataFrame()
    df = pd.concat(data_frames, ignore_index=True)
    
    # --- NEW: Filter out spin-up time to protect statistics and plots ---
    df['Time'] = pd.to_datetime(df['Time'])
    min_time = df['Time'].min()
    
    # Find the start of the main 24h day (first midnight)
    if min_time.hour == 0 and min_time.minute == 0:
        main_day_start = min_time
    else:
        main_day_start = (min_time + pd.Timedelta(days=1)).normalize()
        
    main_day_end = main_day_start + pd.Timedelta(days=1)
    
    # Keep strictly the 24-hour focus period (00:00 to 00:00 next day)
    df = df[(df['Time'] >= main_day_start) & (df['Time'] <= main_day_end)]
    
    df.to_pickle(cache_pkl) 
    return df

# --- PLOTTING FUNCTIONS ---

def get_diurnal_series(df, pol):
    """
    Calculates continuous elapsed hours (0 to 24), automatically resetting
    the clock for different seasons so they perfectly superimpose when aggregated.
    """
    spatial_avg = df.groupby('Time')[pol].mean().reset_index()
    
    # CRITICAL FIX 1: Round datetime to the nearest minute to eliminate simulation drift
    spatial_avg['Time'] = pd.to_datetime(spatial_avg['Time']).dt.round('min')
    
    # Sort chronologically to properly detect breaks between seasons
    spatial_avg = spatial_avg.sort_values('Time')
    
    # Identify separate seasons by looking for time gaps larger than 12 hours
    spatial_avg['Season_Block'] = (spatial_avg['Time'].diff() > pd.Timedelta(hours=12)).cumsum()
    
    # Calculate elapsed hours
    block_starts = spatial_avg.groupby('Season_Block')['Time'].transform('min')
    spatial_avg['Elapsed_Hours'] = (spatial_avg['Time'] - block_starts).dt.total_seconds() / 3600.0
    
    # CRITICAL FIX 2: Round the resulting float to 3 decimal places.
    # This forces 17.000000 and 17.000001 to both become 17.000, ensuring 
    # datasets align perfectly for heatmaps and average cleanly for splines.
    spatial_avg['Elapsed_Hours'] = spatial_avg['Elapsed_Hours'].round(3)
    
    # Group by the rounded elapsed hours, averaging everything cleanly together
    return spatial_avg.groupby('Elapsed_Hours')[pol].mean().sort_index()


def plot_comparative_diurnal(base_df, methods_dict, title_suffix, file_suffix, pollutants, out_dir, smooth_lines=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_flat = axes.flatten()
    method_names = list(methods_dict.keys())
    colors = cmc.batlow(np.linspace(0.1, 0.9, len(method_names)))

    for i, pol in enumerate(pollutants):
        if i >= len(ax_flat): break
        ax = ax_flat[i]
        
        b_diurnal = get_diurnal_series(base_df, pol)
        
        if smooth_lines:
            x_b, y_b = smooth_data(b_diurnal)
            ax.plot(x_b, y_b, color='black', linewidth=3.0, label='Base Scenario', zorder=10)
        else:
            ax.plot(b_diurnal.index, b_diurnal, color='black', linewidth=3.0, label='Base Scenario', zorder=10)

        for j, m_name in enumerate(method_names):
            m_diurnal = get_diurnal_series(methods_dict[m_name], pol)
            
            if smooth_lines:
                x_m, y_m = smooth_data(m_diurnal)
                ax.plot(x_m, y_m, color=colors[j], linestyle='--', alpha=0.8, label=m_name)
            else:
                ax.plot(m_diurnal.index, m_diurnal, color=colors[j], linestyle='--', alpha=0.8, label=m_name)

        ax.set_title(f"{pol} - {title_suffix}", fontweight='bold')
        ax.set_ylabel("Spatial Mean Concentration [µg m$^{-3}$]")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(True, linestyle=':', alpha=0.6)

    handles, labels = ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.92, wspace=0.25, hspace=0.3)
    plt.savefig(os.path.join(out_dir, f"Diurnal_Absolute_{file_suffix}.png"))
    plt.close()

def plot_difference_diurnal(base_df, methods_dict, title_suffix, file_suffix, pollutants, out_dir, smooth_lines=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_flat = axes.flatten()
    method_names = list(methods_dict.keys())
    colors = cmc.batlow(np.linspace(0.1, 0.9, len(method_names)))

    for i, pol in enumerate(pollutants):
        if i >= len(ax_flat): break
        ax = ax_flat[i]
        
        b_diurnal = get_diurnal_series(base_df, pol)
        ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=1)

        for j, m_name in enumerate(method_names):
            m_diurnal = get_diurnal_series(methods_dict[m_name], pol)
            
            # Calculate delta and drop any potential NaNs from missing timesteps
            diff_diurnal = (m_diurnal - b_diurnal).dropna()
            
            if smooth_lines:
                # PASS ALLOW_NEGATIVE=TRUE SO WE CAN SEE REDUCTIONS IN POLLUTION
                x_d, y_d = smooth_data(diff_diurnal, allow_negative=True)
                ax.plot(x_d, y_d, color=colors[j], alpha=0.8, label=m_name)
            else:
                ax.plot(diff_diurnal.index, diff_diurnal, color=colors[j], alpha=0.8, label=m_name)

        ax.set_title(rf"$\Delta$ {pol} - {title_suffix}", fontweight='bold')
        ax.set_ylabel(rf"$\Delta$ Concentration [µg m$^{{-3}}$]")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(True, linestyle=':', alpha=0.6)

    handles, labels = ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.92, wspace=0.25, hspace=0.3)
    plt.savefig(os.path.join(out_dir, f"Diurnal_Delta_{file_suffix}.png"))
    plt.close()

def plot_regression_scatter(base_df, methods_dict, title_suffix, file_suffix, pollutants, out_dir):
    method_names = list(methods_dict.keys())
    colors = cmc.batlow(np.linspace(0.1, 0.9, len(method_names)))

    # --- NEW: Dynamic Grid Calculation ---
    n_methods = len(method_names)
    cols = 4 if n_methods >= 4 else n_methods
    rows = int(np.ceil(n_methods / cols))

    for pol in pollutants:
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 5 * rows))
        
        # Ensure axes is iterable and flatten it
        if n_methods == 1:
            ax_flat = [axes]
        else:
            ax_flat = axes.flatten()
        
        # FAST: Direct value extraction
        x_full = base_df[pol].values
        
        for j, m_name in enumerate(method_names):
            ax = ax_flat[j]
            y_full = methods_dict[m_name][pol].values
            
            mask = ~np.isnan(x_full) & ~np.isnan(y_full)
            x, y = x_full[mask], y_full[mask]
            
            if len(x) > 2:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                rmse = np.sqrt(mean_squared_error(x, y))
                
                # FAST: Limit point rendering to 100k
                if len(x) > 100000:
                    idx = np.random.choice(len(x), 100000, replace=False)
                    x_plot, y_plot = x[idx], y[idx]
                else:
                    x_plot, y_plot = x, y
                    
                axis_max = max(x_plot.max(), y_plot.max()) * 1.05
                dynamic_alpha = max(0.01, min(0.3, 20000 / len(x_plot)))
                
                ax.scatter(x_plot, y_plot, alpha=dynamic_alpha, color=colors[j], edgecolors='none', s=2, rasterized=True)
                ax.plot([0, axis_max], [0, axis_max], 'k--', lw=1.5, zorder=10, label='1:1 Line')
                ax.plot(np.array([0, axis_max]), intercept + slope * np.array([0, axis_max]), color='red', lw=2, label='Fit')
                
                stats_text = f"$R^2$ = {r_value**2:.2f}\nRMSE = {rmse:.2f}"
                ax.text(0.05, 0.85, stats_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                ax.set_title(f"{m_name}", fontweight='bold')
                ax.set_xlabel("Base [µg m$^{-3}$]")
                ax.set_ylabel(f"{m_name} [µg m$^{-3}$]")
                ax.set_xlim(0, axis_max); ax.set_ylim(0, axis_max)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='lower right', frameon=True)

        # --- NEW: Hide any unused subplots in the grid ---
        for j in range(n_methods, len(ax_flat)):
            ax_flat[j].set_visible(False)

        plt.suptitle(f"Scatter: {pol} (Cell-by-Cell) - {title_suffix}", fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"Scatter_{pol}_{file_suffix}.png"), bbox_inches='tight')
        plt.close()
        
def plot_ridge_distribution(base_df, methods_dict, title_suffix, file_suffix, pollutants, out_dir):
    combined_data = []
    SAMPLE_LIMIT = 50000 
    
    for pol in pollutants:
        b_vals = base_df[pol].dropna()
        if len(b_vals) > SAMPLE_LIMIT: b_vals = b_vals.sample(SAMPLE_LIMIT, random_state=42)
        
        temp_df = pd.DataFrame({'Concentration': b_vals})
        temp_df['Method'], temp_df['Pollutant'] = 'Base Scenario', pol
        combined_data.append(temp_df)
        
    for m_name, m_df in methods_dict.items():
        for pol in pollutants:
            m_vals = m_df[pol].dropna()
            if len(m_vals) > SAMPLE_LIMIT: m_vals = m_vals.sample(SAMPLE_LIMIT, random_state=42)
            
            temp_df = pd.DataFrame({'Concentration': m_vals})
            temp_df['Method'], temp_df['Pollutant'] = m_name, pol
            combined_data.append(temp_df)
            
    plot_df = pd.concat(combined_data)
    method_names = ['Base Scenario'] + list(methods_dict.keys())
    colors = cmc.batlow(np.linspace(0.1, 0.9, len(method_names)))
    
    for pol in pollutants:
        subset = plot_df[plot_df['Pollutant'] == pol].copy()
        
        # Calculate x_max based on ALL data in the subset
        if len(subset) > 0:
            x_max_cutoff = subset['Concentration'].quantile(0.995) * 1.2
        else:
            x_max_cutoff = 1.0

        # CRITICAL FIX: Physically drop extreme outliers from the dataframe.
        # This guarantees Seaborn cannot draw an invisible tail that expands the canvas.
        subset = subset[subset['Concentration'] <= x_max_cutoff]

        # Aspect=5 keeps them wide and readable
        g = sns.FacetGrid(subset, row="Method", hue="Method", aspect=5, height=1.5, 
                          palette=colors, sharex=True)
        
        # Add clip=(0, x_max_cutoff) to forcefully bound the KDE calculation
        g.map(sns.kdeplot, "Concentration", bw_adjust=1.0, clip=(0, x_max_cutoff), clip_on=False, fill=True, alpha=0.7, linewidth=1.5)
        g.map(sns.kdeplot, "Concentration", clip=(0, x_max_cutoff), clip_on=False, color="w", lw=2, bw_adjust=1.0)
        g.map(plt.axhline, y=0, lw=1.5, color='black', clip_on=False)
        
        g.set(xlim=(0, x_max_cutoff))
        
        g.set_titles("") 
        g.set(yticks=[], ylabel="")
        g.despine(left=True)
        
        # Place labels cleanly in the top right corner
        for ax, m_name, color in zip(g.axes.flat, method_names, colors):
            ax.text(0.98, 0.8, m_name, fontweight="bold", color=color, 
                    ha="right", va="top", transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        
        # Positive hspace keeps the plots from crushing each other
        g.fig.subplots_adjust(hspace=0.2)
        g.fig.suptitle(f"{pol} Spread (Cell-by-Cell) - {title_suffix}", fontweight='bold', y=1.02)
        
        plt.savefig(os.path.join(out_dir, f"RidgePlot_{pol}_{file_suffix}.png"), bbox_inches='tight')
        plt.close()

def plot_taylor_diagram(base_df, methods_dict, title_suffix, file_suffix, pollutants, out_dir):
    method_names = list(methods_dict.keys())
    all_colors = cmc.batlow(np.linspace(0.1, 0.9, len(method_names)))
    color_map = {m: mcolors.to_hex(c) for m, c in zip(method_names, all_colors)}
    
    for pol in pollutants:
        sdev, crmse, ccoef = [], [], []
        valid_methods = []
        
        x_full = base_df[pol].values
        mask_b = ~np.isnan(x_full)
        if np.sum(mask_b) < 2: continue
            
        std_base = np.std(x_full[mask_b], ddof=0)
        sdev.append(std_base); crmse.append(0.0); ccoef.append(1.0)
        
        for m_name in method_names:
            y_full = methods_dict[m_name][pol].values
            mask = ~np.isnan(x_full) & ~np.isnan(y_full)
            x, y = x_full[mask], y_full[mask]
            
            if len(x) > 1:
                valid_methods.append(m_name)
                std_m = np.std(y, ddof=0)
                r = np.corrcoef(x, y)[0, 1]
                exact_crmse = np.sqrt(max(0.0, std_m**2 + std_base**2 - 2 * std_m * std_base * r))
                
                sdev.append(std_m); ccoef.append(r); crmse.append(exact_crmse)
        
        if len(sdev) <= 1: continue 
            
        fig = plt.figure(figsize=(10, 8))
        sm.taylor_diagram(np.array(sdev), np.array(crmse), np.array(ccoef),
                          markerLegend='off', styleOBS='-', colOBS='black', titleOBS='Base', checkstats='on')
        
        X = np.array(sdev) * np.array(ccoef)
        Y = np.array(sdev) * np.sqrt(np.maximum(0.0, 1.0 - np.array(ccoef)**2))
        ax = plt.gca()
        
        ax.scatter(X[0], Y[0], c='#000000', s=150, label='Base Scenario', zorder=10, edgecolors='w', linewidths=1.5)
        for i, m_name in enumerate(valid_methods):
            ax.scatter(X[i+1], Y[i+1], c=color_map[m_name], s=150, label=m_name, zorder=10, edgecolors='w', linewidths=1.5)
            
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), frameon=False, title='Scenarios', title_fontproperties={'weight':'bold'})
        plt.title(f"Taylor Diagram (Cell-by-Cell): {pol} - {title_suffix}", y=1.08, fontweight='bold')
        plt.subplots_adjust(right=0.75) 
        plt.savefig(os.path.join(out_dir, f"TaylorDiagram_{pol}_{file_suffix}.png"), bbox_inches='tight')
        plt.close()

def plot_temporal_heatmap(base_df, methods_dict, title_suffix, file_suffix, pollutants, out_dir):
    for pol in pollutants:
        b_diurnal = get_diurnal_series(base_df, pol)
        
        delta_data = {}
        for m_name, m_df in methods_dict.items():
            m_diurnal = get_diurnal_series(m_df, pol)
            delta_data[m_name] = m_diurnal - b_diurnal
            
        delta_df = pd.DataFrame(delta_data).T.interpolate(axis=1).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 4))
        max_val = np.nanmax(np.abs(delta_df.values))
        max_val = max_val if max_val > 0 else 1 
        
        sns.heatmap(delta_df, cmap=cmc.vik, center=0, vmin=-max_val, vmax=max_val,
                    cbar_kws={'label': rf'$\Delta$ {pol} [µg m$^{{-3}}$]'}, ax=ax)
        
        ax.set_title(rf"Hourly $\Delta$ from Base: {pol} - {title_suffix}", fontweight='bold', pad=15)
        ax.set_xlabel("Time of Day"); ax.set_ylabel("") 
        
        # SMART LABEL STEPPING - FIXING THE 23:60 BUG
        hours = delta_df.columns
        step = max(1, len(hours) // 12) 
        tick_locs = np.arange(0, len(hours), step)
        ax.set_xticks(tick_locs + 0.5)
        
        labels = []
        for h in hours:
            hh = int(h)
            mm = int(round((h % 1) * 60))
            # Catch the rounding overflow
            if mm == 60:
                hh += 1
                mm = 0
            # Format cleanly, using modulo 24 to wrap 24:00 back to 00:00
            labels.append(f"{hh % 24:02d}:{mm:02d}")
            
        ax.set_xticklabels([labels[i] for i in tick_locs], rotation=45, ha='right')
        ax.tick_params(axis='y', rotation=0) 
        
        plt.savefig(os.path.join(out_dir, f"Heatmap_{pol}_{file_suffix}.png"), bbox_inches='tight')
        plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    SMOOTH_PLOTS = True
    Z_INDEX = 2 
    
    # --- DIRECTORIES ---
    base_out_dir = r"D:\Berlin_Anonymization_Study_Results"
    stats_dir = os.path.join(base_out_dir, "Stats")
    os.makedirs(stats_dir, exist_ok=True)
        
    # --- SEASONAL INPUT PATHS CONFIGURATION ---
    seasons_config = {
        "Summer": {
            "Base": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_Base_comp\NetCDF",
            "Methods": {
                "K2": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_K2_comp\NetCDF",
                "K5": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_K5_comp\NetCDF",
                "K7": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_K7_comp\NetCDF",
                "dptraj20": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_dptraj20_comp\NetCDF",
                "dptraj22": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_dptraj22_comp\NetCDF",
                "dptraj50": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_dptraj50_comp\NetCDF",
                "adatrace_g6": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_adatrace_g6_comp\NetCDF",
                "adatrace_g12": r"Y:\Berlin_Mehringdamm\Mehringdamm_Summer_adatrace_g12_comp\NetCDF"
            }
        },
        "Autumn": {
            "Base": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_Base_comp\NetCDF",
            "Methods": {
                "K2": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_K2_comp\NetCDF",
                "K5": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_K5_comp\NetCDF",
                "K7": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_K7_comp\NetCDF",
                "dptraj20": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_dptraj20_comp\NetCDF",
                "dptraj22": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_dptraj22_comp\NetCDF",
                "dptraj50": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_dptraj50_comp\NetCDF",
                "adatrace_g6": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_adatrace_g6_comp\NetCDF",
                "adatrace_g12": r"Y:\Berlin_Mehringdamm\Mehringdamm_Autumn_adatrace_g12_comp\NetCDF"                
            }
        }
    }

    mask_files = {
        "Entire_Area": r"D:\Masks\entire_area_mask_JSON.inx",
        "Main_Road": r"D:\Masks\main_road_mask_JSON.inx",
        "Secondary_Roads": r"D:\Masks\secondary_roads_mask_JSON.inx"
    }
    
    pollutants_to_plot = ['PM2.5', 'PM10', 'NO', 'NO2']
    all_stats = []

    for mask_name, mask_path in mask_files.items():
        print(f"\n================ Processing Mask: {mask_name} ================")
        
        mask_boolean = load_mask_from_inx(mask_path)
        if mask_boolean is None:
            continue
            
        agg_base_frames = []
        agg_method_frames = {m: [] for m in seasons_config["Summer"]["Methods"].keys()}

        # 1. PROCESS EACH SEASON SEPARATELY
        for season, paths in seasons_config.items():
            print(f"\n  --- Extracting {season} ---")
            
            season_cache = os.path.join(base_out_dir, "Cache", season)
            season_plot = os.path.join(base_out_dir, "Plots", season)
            os.makedirs(season_cache, exist_ok=True); os.makedirs(season_plot, exist_ok=True)
            
            b_cache_file = os.path.join(season_cache, f"Base_{mask_name}.pkl")
            df_base = load_envimet_mask_series(paths["Base"], mask_boolean, Z_INDEX, b_cache_file)
            
            if df_base.empty:
                print(f"      No base data for {season}. Skipping season.")
                continue
            
            agg_base_frames.append(df_base)
            
            methods_data = {}
            for m_name, m_folder in paths["Methods"].items():
                m_cache_file = os.path.join(season_cache, f"{m_name}_{mask_name}.pkl")
                df_method = load_envimet_mask_series(m_folder, mask_boolean, Z_INDEX, m_cache_file)
                
                if not df_method.empty:
                    methods_data[m_name] = df_method
                    agg_method_frames[m_name].append(df_method)
                    
                    for pol in pollutants_to_plot:
                        stats = calculate_statistics(df_base, df_method, pol)
                        if stats:
                            stats_row = {'Mask': mask_name, 'Season': season, 'Method': m_name, 'Pollutant': pol}
                            stats_row.update(stats)
                            all_stats.append(stats_row)

            title_suf = f"{mask_name} ({season})"
            file_suf = f"{mask_name}_{season}"
            
            print(f"      Plotting {season} visuals...")
            plot_comparative_diurnal(df_base, methods_data, title_suf, file_suf, pollutants_to_plot, season_plot, SMOOTH_PLOTS)
            plot_difference_diurnal(df_base, methods_data, title_suf, file_suf, pollutants_to_plot, season_plot, SMOOTH_PLOTS)
            plot_regression_scatter(df_base, methods_data, title_suf, file_suf, pollutants_to_plot, season_plot)
            plot_ridge_distribution(df_base, methods_data, title_suf, file_suf, pollutants_to_plot, season_plot)
            plot_taylor_diagram(df_base, methods_data, title_suf, file_suf, pollutants_to_plot, season_plot)
            plot_temporal_heatmap(df_base, methods_data, title_suf, file_suf, pollutants_to_plot, season_plot)

        # 2. PROCESS AGGREGATED SEASONS
        print(f"\n  --- Processing Aggregated (Summer + Autumn) for {mask_name} ---")
        if agg_base_frames:
            agg_plot_dir = os.path.join(base_out_dir, "Plots", "Aggregated")
            os.makedirs(agg_plot_dir, exist_ok=True)
            
            df_base_agg = pd.concat(agg_base_frames, ignore_index=True)
            methods_data_agg = {}
            
            for m_name, frames in agg_method_frames.items():
                if frames:
                    df_method_agg = pd.concat(frames, ignore_index=True)
                    methods_data_agg[m_name] = df_method_agg
                    
                    for pol in pollutants_to_plot:
                        stats = calculate_statistics(df_base_agg, df_method_agg, pol)
                        if stats:
                            stats_row = {'Mask': mask_name, 'Season': 'Aggregated', 'Method': m_name, 'Pollutant': pol}
                            stats_row.update(stats)
                            all_stats.append(stats_row)
                            
            title_suf = f"{mask_name} (Aggregated)"
            file_suf = f"{mask_name}_Aggregated"
            
            print(f"      Plotting Aggregated visuals...")
            plot_comparative_diurnal(df_base_agg, methods_data_agg, title_suf, file_suf, pollutants_to_plot, agg_plot_dir, SMOOTH_PLOTS)
            plot_difference_diurnal(df_base_agg, methods_data_agg, title_suf, file_suf, pollutants_to_plot, agg_plot_dir, SMOOTH_PLOTS)
            plot_regression_scatter(df_base_agg, methods_data_agg, title_suf, file_suf, pollutants_to_plot, agg_plot_dir)
            plot_ridge_distribution(df_base_agg, methods_data_agg, title_suf, file_suf, pollutants_to_plot, agg_plot_dir)
            plot_taylor_diagram(df_base_agg, methods_data_agg, title_suf, file_suf, pollutants_to_plot, agg_plot_dir)
            plot_temporal_heatmap(df_base_agg, methods_data_agg, title_suf, file_suf, pollutants_to_plot, agg_plot_dir)

    # 4. EXPORT GLOBAL STATISTICS
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df = stats_df.sort_values(by=['Mask', 'Season', 'Pollutant', 'Method'])
        stats_export_path = os.path.join(stats_dir, "Anonymization_Statistics.csv")
        stats_df.to_csv(stats_export_path, index=False, sep=';', decimal=',')
        print(f"\nStatistics successfully exported to {stats_export_path}")

    print("\nAll Processing Complete!")