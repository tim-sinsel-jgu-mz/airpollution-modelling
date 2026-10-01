import os
import glob
import json
import pandas as pd
import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
from scipy.interpolate import make_interp_spline
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
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

# --- COLOR FAMILIES (Crameri-inspired) ---

# Using the official cmcrameri library for exact color matching.
# Values are spaced out to maximize contrast against a white background and between model/meas.
COLORS = {
    # Particulates using 'hawaii' (Dark Purple to Bright Orange)
    'PM2.5': {'meas': cmc.hawaii(0.1), 'mod': cmc.hawaii(0.3), 'bg': cmc.hawaii(0.2)}, 
    'PM10':  {'meas': cmc.hawaii(0.6), 'mod': cmc.hawaii(0.8), 'bg': cmc.hawaii(0.7)}, 
    
    # Nitrogen Oxides using 'batlow' (Dark Blue/Green to Peach) to keep distinct from PMs
    'NO':    {'meas': cmc.batlow(0.1), 'mod': cmc.batlow(0.3), 'bg': cmc.batlow(0.2)}, 
    'NO2':   {'meas': cmc.batlow(0.6), 'mod': cmc.batlow(0.8), 'bg': cmc.batlow(0.7)}  
}

# --- NEW HELPER FUNCTION FOR SMOOTHING ---
def smooth_data(series, num_points=300):
    """Applies cubic spline interpolation to smooth a pandas Series."""
    series = series.dropna()
    if len(series) < 4:  # Splines require at least 4 points
        return series.index.values, series.values
    
    x = series.index.values
    y = series.values
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    
    # Prevent the mathematical curve from dipping below zero
    y_smooth = np.maximum(y_smooth, 0)
    
    return x_smooth, y_smooth

def load_measurements(csv_path, target_start, target_end):
    print(f"--- Loading Measurements: {os.path.basename(csv_path)} ---")
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M', errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)
        df.set_index('Datetime', inplace=True)
        
        try:
            df = df.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('Etc/GMT-1')
            df.index = df.index.tz_localize(None) 
        except Exception as e:
            print(f"Timezone conversion error: {e}")
            df.index = df.index.tz_localize(None)

        df.rename(columns={'PM10': 'PM10', 'PM2,5': 'PM2.5', 'NO': 'NO', 'NO2': 'NO2'}, inplace=True)
        df = df[['PM10', 'PM2.5', 'NO', 'NO2']].apply(pd.to_numeric, errors='coerce')
        
        df = df.loc[target_start:target_end]
        return df
    except Exception as e:
        print(f"Error loading Measurements: {e}")
        return pd.DataFrame()

def load_fox_background(fox_path, target_start, target_end):
    print(f"--- Loading FOX Background Data ---")
    try:
        with open(fox_path, 'r') as f:
            data = json.load(f)
        
        records = []
        for ts in data['timestepList']:
            dt = pd.to_datetime(ts['date'] + ' ' + ts['time'])
            dt = dt.replace(year=target_start.year) 
            
            pol = ts['backgrPollutants']
            
            records.append({
                'Datetime': dt,
                'PM10_BG': pol.get('PM10', 0),
                'PM2.5_BG': pol.get('PM25', 0),
                'NO_BG': pol.get('NO', 0),
                'NO2_BG': pol.get('NO2', 0),
            })
            
        df_fox = pd.DataFrame(records).set_index('Datetime').sort_index()
        df_fox = df_fox.resample('1h').mean().loc[target_start:target_end]
        return df_fox
    except Exception as e:
        print(f"Error loading FOX: {e}")
        return pd.DataFrame()

def load_traffic_volume(traffic_path, sim_dates):
    print(f"--- Loading Traffic Data ---")
    try:
        df_t = pd.read_csv(traffic_path, sep=';')
        df_t['Hour'] = pd.to_datetime(df_t['Time'], format='%H:%M:%S').dt.hour
        df_t.set_index('Hour', inplace=True)
        
        full_range = pd.date_range(sim_dates.min(), sim_dates.max(), freq='1h')
        traffic_series = pd.DataFrame(index=full_range)
        traffic_series['Traffic'] = traffic_series.index.hour.map(df_t['TrajCount'])
        return traffic_series
    except Exception as e:
        print(f"Error loading Traffic: {e}")
        return pd.DataFrame()

def load_envimet_series(nc_folder_path, x_idx, y_idx, z_idx, cache_dir):
    nc_files = sorted(glob.glob(os.path.join(nc_folder_path, "*.nc")))
    if not nc_files: return pd.DataFrame(), "Unknown"
    
    sim_name = os.path.basename(os.path.normpath(nc_folder_path.replace(r"\NetCDF", "")))
    cache_file = os.path.join(cache_dir, f"Extracted_{sim_name}_X{x_idx}_Y{y_idx}_Z{z_idx}.csv")
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        data_frames = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                pt = ds.isel(GridsI=x_idx, GridsJ=y_idx, GridsK=z_idx)
                
                chunk = pd.DataFrame({
                    'PM2.5': pt['PM25Conc'].values if 'PM25Conc' in ds else np.nan,
                    'PM10': pt['PM10Conc'].values if 'PM10Conc' in ds else np.nan,
                    'NO': pt['NOConc'].values if 'NOConc' in ds else np.nan,
                    'NO2': pt['NO2Conc'].values if 'NO2Conc' in ds else np.nan
                }, index=pd.to_datetime(pt['Time'].values))
                data_frames.append(chunk)
        df = pd.concat(data_frames).sort_index()
        df.to_csv(cache_file)

    df.index = df.index.round('1min')
    return df, sim_name

def calculate_statistics(x, y):
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    r = np.corrcoef(x, y)[0, 1]
    r2_pearson = r**2
    rmse = np.sqrt(mean_squared_error(x, y))

    return {
        "r2_pearson": r2_pearson,
        "rmse": rmse,
    }

# --- ADDED smooth_lines ARGUMENT ---
def plot_final_results(df_meas, df_model, df_fox, df_traffic, pollutants, out_dir_dict, sim_name, coords, smooth_lines=False):
    stats_export_list = [] 
    
    # 1. DIURNAL PLOT (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_flat = axes.flatten()

    for i, pol in enumerate(pollutants):
        ax = ax_flat[i]
        c = COLORS[pol]
        
        stats = calculate_statistics(df_meas[pol], df_model[pol])
        if stats is not None:
            stats_str = f"$R^2 = {stats['r2_pearson']:.2f}$\nRMSE = {stats['rmse']:.2f}"
        else:
            stats_str = "No Data"

        m_diurnal = df_meas.groupby(df_meas.index.hour)[pol].mean().reindex(range(0, 24))
        s_diurnal = df_model.groupby(df_model.index.hour)[pol].mean().reindex(range(0, 24))
        f_diurnal = df_fox.groupby(df_fox.index.hour)[f"{pol}_BG"].mean().reindex(range(0, 24))
        t_diurnal = df_traffic.groupby(df_traffic.index.hour)['Traffic'].mean().reindex(range(0, 24))

        ax2 = ax.twinx()
        
        # --- APPLY SMOOTHING LOGIC ---
        if smooth_lines:
            x_t, y_t = smooth_data(t_diurnal)
            ax2.fill_between(x_t, 0, y_t, color='gray', alpha=0.15, label='Traffic')
            
            x_m, y_m = smooth_data(m_diurnal)
            x_s, y_s = smooth_data(s_diurnal)
            x_f, y_f = smooth_data(f_diurnal)
            
            ax.plot(x_m, y_m, color=c['meas'], linestyle='-', label=f'Meas {pol}', zorder=5)
            ax.plot(x_s, y_s, color=c['mod'], linestyle='-', label=f'Mod {pol}', zorder=6)
            ax.plot(x_f, y_f, color=c['bg'], linestyle='--', alpha=0.6, label=f'BG {pol}', zorder=4)
        else:
            ax2.fill_between(t_diurnal.index, 0, t_diurnal, color='gray', alpha=0.15, label='Traffic')
            ax.plot(m_diurnal.index, m_diurnal, color=c['meas'], linestyle='-', label=f'Meas {pol}', zorder=5)
            ax.plot(s_diurnal.index, s_diurnal, color=c['mod'], linestyle='-', label=f'Mod {pol}', zorder=6)
            ax.plot(f_diurnal.index, f_diurnal, color=c['bg'], linestyle='--', alpha=0.6, label=f'BG {pol}', zorder=4)
        # ------------------------------

        if i % 2 == 1:  
            ax2.set_ylabel("Traffic Volume [Veh/h]", color='black', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, None)

        ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(f"Diurnal Cycle: {pol}", fontweight='bold')
        ax.set_ylabel("Concentration [µg m$^{-3}$]")
        ax.set_xlabel("Hour")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        
        max_y = max(m_diurnal.max(), s_diurnal.max(), f_diurnal.max())
        ax.set_ylim(0, max_y * 1.20 if not np.isnan(max_y) else None) 
        ax.grid(True, which='major', linestyle=':', alpha=0.6)
        
        if i == 0:
            handles_main, labels_main = ax.get_legend_handles_labels()
            handles_sec, labels_sec = ax2.get_legend_handles_labels()

    handles = handles_main + handles_sec
    labels = labels_main + labels_sec
    fig.legend(handles, labels, loc='lower center', ncol=len(labels)//2 + 1, frameon=False, bbox_to_anchor=(0.5, -0.02))
    
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.92, wspace=0.3, hspace=0.3)
    
    plt.savefig(os.path.join(out_dir_dict['Diurnal'], f"Diurnal_{sim_name}.png"))
    plt.savefig(os.path.join(out_dir_dict['Diurnal'], f"Diurnal_{sim_name}.svg"))
    plt.close()
    
    # 2. REGRESSION PLOT (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax_flat = axes.flatten()

    for i, pol in enumerate(pollutants):
        ax = ax_flat[i]
        c = COLORS[pol]
        x_raw, y_raw = df_meas[pol], df_model[pol]
        mask = x_raw.notna() & y_raw.notna()
        x, y = x_raw[mask], y_raw[mask]
        
        if len(x) > 1:
            stats = calculate_statistics(x, y)
            m, b = np.polyfit(x, y, 1)
            
            stats_row = {'Pollutant': pol, 'Comparison': 'Model vs Measured', 'm': m, 'b': b}
            stats_row.update(stats)
            stats_export_list.append(stats_row)
            
            ax.scatter(x, y, color=c['mod'], alpha=0.5, s=20, edgecolors='none')
            lim = max(x.max(), y.max()) * 1.1 if not x.empty else 100
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            ax.plot(x, m*x + b, color=c['meas'], linewidth=1.5, label='Fit')
            ax.plot([0, lim], [0, 0.5*lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2', zorder=1)
            ax.plot([0, lim], [0, 2*lim], 'k--', alpha=0.2, linewidth=0.8, zorder=1)
            
            stats_str = f"$R^2 = {stats['r2_pearson']:.2f}$\nRMSE = {stats['rmse']:.2f}"
            ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                    fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
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

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(out_dir_dict['Regression'], f"Regression_{sim_name}.png"))
    plt.savefig(os.path.join(out_dir_dict['Regression'], f"Regression_{sim_name}.svg"))
    plt.close()
   
    # 3. BACKGROUND REGRESSION PLOT (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax_flat = axes.flatten()

    for i, pol in enumerate(pollutants):
        ax = ax_flat[i]
        c = COLORS[pol]
        x_raw, y_raw = df_fox[f"{pol}_BG"], df_model[pol]
        mask = x_raw.notna() & y_raw.notna()
        x, y = x_raw[mask], y_raw[mask]
        
        if len(x) > 1:
            stats = calculate_statistics(x, y)
            m, b = np.polyfit(x, y, 1)
            
            stats_row = {'Pollutant': pol, 'Comparison': 'Model vs Background', 'm': m, 'b': b}
            stats_row.update(stats)
            stats_export_list.append(stats_row)
            
            ax.scatter(x, y, color=c['mod'], alpha=0.5, s=20, edgecolors='none')
            lim = max(x.max(), y.max()) * 1.1 if not x.empty else 100
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            ax.plot(x, m*x + b, color=c['meas'], linewidth=1.5, label='Fit')
            ax.plot([0, lim], [0, 0.5*lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2', zorder=1)
            ax.plot([0, lim], [0, 2*lim], 'k--', alpha=0.2, linewidth=0.8, zorder=1)
            
            stats_str = f"$R^2 = {stats['r2_pearson']:.2f}$\nRMSE = {stats['rmse']:.2f}"
            ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                    fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')

        ax.set_title(f"{pol} Regression (Background)", fontweight='bold')
        ax.set_xlabel("Background [µg m$^{-3}$]")
        ax.set_ylabel("Modelled [µg m$^{-3}$]")

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(out_dir_dict['Regression_BG'], f"Regression_BG_{sim_name}.png"))
    plt.savefig(os.path.join(out_dir_dict['Regression_BG'], f"Regression_BG_{sim_name}.svg"))
    plt.close()
    
    if stats_export_list:
        save_path = os.path.join(out_dir_dict['Stats'], f"Stats_{sim_name}.csv")
        pd.DataFrame(stats_export_list).to_csv(save_path, index=False, sep=';', decimal=',')


# --- ADDED smooth_lines ARGUMENT ---
def plot_aggregated_diurnal(all_sim_data, pollutant_groups, out_dir, smooth_lines=False):
    """Generates grouped diurnal plots per pollutant category."""
    print("--- Generating Aggregated Diurnal Plots ---")
    n_sims = len(all_sim_data)
    if n_sims == 0:
        return

    cols = 2 if n_sims > 1 else 1
    rows = int(np.ceil(n_sims / cols)) if n_sims > 0 else 1

    for group_name, pols in pollutant_groups.items():
        max_val_conc = 0
        max_val_traf = 0
        
        # Calculate Y limits dynamically across the entire group
        for sim in all_sim_data:
            t_diurnal = sim['traffic'].groupby(sim['traffic'].index.hour)['Traffic'].mean()
            local_max_traf = t_diurnal.max()
            if not np.isnan(local_max_traf) and local_max_traf > max_val_traf:
                max_val_traf = local_max_traf
                
            for pol in pols:
                m_diurnal = sim['meas'].groupby(sim['meas'].index.hour)[pol].mean()
                s_diurnal = sim['model'].groupby(sim['model'].index.hour)[pol].mean()
                f_diurnal = sim['fox'].groupby(sim['fox'].index.hour)[f"{pol}_BG"].mean()
                local_max_conc = np.nanmax([m_diurnal.max(), s_diurnal.max(), f_diurnal.max()])
                if not np.isnan(local_max_conc) and local_max_conc > max_val_conc:
                    max_val_conc = local_max_conc

        y_lim_conc = max_val_conc * 1.20 if max_val_conc > 0 else 45
        y_lim_traf = max_val_traf * 1.20 if max_val_traf > 0 else 100

        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4.5))
        
        if n_sims == 1:
            ax_flat = [axes]
        elif rows == 1 or cols == 1:
            ax_flat = axes.flatten()
        else:
            ax_flat = axes.flatten()

        for i, sim in enumerate(all_sim_data):
            ax = ax_flat[i]
            date_str = sim['date_str']
            
            t_diurnal = sim['traffic'].groupby(sim['traffic'].index.hour)['Traffic'].mean().reindex(range(0, 24))

            ax2 = ax.twinx()
            
            # --- APPLY SMOOTHING LOGIC ---
            if smooth_lines:
                x_t, y_t = smooth_data(t_diurnal)
                ax2.fill_between(x_t, 0, y_t, color='gray', alpha=0.15, label='Traffic')
            else:
                ax2.fill_between(t_diurnal.index, 0, t_diurnal, color='gray', alpha=0.15, label='Traffic')
            # -----------------------------
            
            ax2.set_ylim(0, y_lim_traf)
            
            # --- Y-Axis Layout Management ---
            if (i % cols) == 0:
                ax.set_ylabel("Concentration [µg m$^{-3}$]")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if (i % cols) == (cols - 1) or i == (n_sims - 1):
                ax2.set_ylabel("Traffic Volume [Veh/h]", color='black', fontsize=11)
            else:
                ax2.yaxis.set_visible(False) 
            # --------------------------------

            stats_texts = []
            for pol in pols:
                c = COLORS[pol]
                m_diurnal = sim['meas'].groupby(sim['meas'].index.hour)[pol].mean().reindex(range(0, 24))
                s_diurnal = sim['model'].groupby(sim['model'].index.hour)[pol].mean().reindex(range(0, 24))
                f_diurnal = sim['fox'].groupby(sim['fox'].index.hour)[f"{pol}_BG"].mean().reindex(range(0, 24))

                # --- APPLY SMOOTHING LOGIC ---
                if smooth_lines:
                    x_m, y_m = smooth_data(m_diurnal)
                    x_s, y_s = smooth_data(s_diurnal)
                    x_f, y_f = smooth_data(f_diurnal)
                    
                    ax.plot(x_m, y_m, color=c['meas'], linestyle='-', label=f'Meas {pol}', zorder=5)
                    ax.plot(x_s, y_s, color=c['mod'], linestyle='-', label=f'Mod {pol}', zorder=6)
                    ax.plot(x_f, y_f, color=c['bg'], linestyle='--', alpha=0.6, label=f'BG {pol}', zorder=4)
                else:
                    ax.plot(m_diurnal.index, m_diurnal, color=c['meas'], linestyle='-', label=f'Meas {pol}', zorder=5)
                    ax.plot(s_diurnal.index, s_diurnal, color=c['mod'], linestyle='-', label=f'Mod {pol}', zorder=6)
                    ax.plot(f_diurnal.index, f_diurnal, color=c['bg'], linestyle='--', alpha=0.6, label=f'BG {pol}', zorder=4)
                # -----------------------------

                stats = calculate_statistics(sim['meas'][pol], sim['model'][pol])
                if stats:
                    stats_texts.append(f"{pol}: $R^2 = {stats['r2_pearson']:.2f}$, RMSE = {stats['rmse']:.1f}")

            stats_str = "\n".join(stats_texts) if stats_texts else "No Data"
            ax.text(0.03, 0.96, stats_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                    fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            ax.set_title(date_str, fontweight='bold')
            ax.set_xlabel("Hour of the Day")
            ax.set_xlim(0, 23)
            ax.set_xticks(range(0, 24, 4))
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))

            # Check if the current subplot is in the bottom row
            if i >= (rows - 1) * cols:
                ax.set_xlabel("Hour of the Day")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False) # Hides the tick numbers for upper rows            
            
            ax.set_ylim(0, y_lim_conc)
            ax.grid(True, which='major', linestyle=':', alpha=0.6)

        for j in range(n_sims, rows * cols):
            fig.delaxes(ax_flat[j])

        handles_main, labels_main = ax_flat[0].get_legend_handles_labels()
        handles_sec, labels_sec = [], []
        
        for a in fig.axes:
            if a.get_ylabel() == "Traffic Volume [Veh/h]":
                handles_sec, labels_sec = a.get_legend_handles_labels()
                break
                
        by_label = dict(zip(labels_main + labels_sec, handles_main + handles_sec))
        
        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.94, wspace=0.08, hspace=0.13)
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=len(by_label), frameon=False, bbox_to_anchor=(0.5, 0.02))
        
        plt.savefig(os.path.join(out_dir, f"Aggregated_Diurnal_{group_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f"Aggregated_Diurnal_{group_name}.svg"), bbox_inches='tight')
        plt.close()

def plot_aggregated_regression(all_sim_data, pollutant_groups, out_dir):
    # (Unchanged from original script)
    print("--- Generating Aggregated Regression Plot ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_flat = axes.flatten()

    for i, (group_name, pols) in enumerate(pollutant_groups.items()):
        ax = ax_flat[i]
        
        global_max = 0
        stats_texts = []

        for pol in pols:
            c = COLORS[pol]
            x_all, y_all = [], []
            for sim in all_sim_data:
                x_raw, y_raw = sim['meas'][pol], sim['model'][pol]
                mask = x_raw.notna() & y_raw.notna()
                x_all.extend(x_raw[mask].values)
                y_all.extend(y_raw[mask].values)
                
            x_arr, y_arr = pd.Series(x_all), pd.Series(y_all)
            if len(x_arr) > 1:
                stats = calculate_statistics(x_arr, y_arr)
                m, b = np.polyfit(x_arr, y_arr, 1)
                
                ax.scatter(x_arr, y_arr, color=c['meas'], alpha=0.5, s=20, edgecolors='none', label=f'{pol} Data')
                ax.plot(x_arr, m*x_arr + b, color=c['meas'], linewidth=2.0, linestyle='-', label=f'{pol} Fit')
                
                local_max = max(x_arr.max(), y_arr.max())
                if local_max > global_max:
                    global_max = local_max
                
                stats_texts.append(f"{pol}: $R^2 = {stats['r2_pearson']:.2f}$, RMSE = {stats['rmse']:.1f}")

        lim = global_max * 1.1 if global_max > 0 else 100
        
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1' if i==0 else "")
        ax.plot([0, lim], [0, 0.5*lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2' if i==0 else "")
        ax.plot([0, lim], [0, 2*lim], 'k--', alpha=0.2, linewidth=0.8)

        stats_str = "\n".join(stats_texts) if stats_texts else "Insufficient Data"
        ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        title_name = group_name.replace('_', ' ')
        ax.set_title(f"{title_name} Aggregated", fontweight='bold')
        ax.set_xlabel("Measured [µg m$^{-3}$]")
        ax.set_ylabel("Modelled [µg m$^{-3}$]")

    # Collect handles and labels from ALL subplots
    handles, labels = [], []
    for ax in ax_flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=11)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.15)
    
    plt.savefig(os.path.join(out_dir, "Aggregated_Regression_Combined.png"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Aggregated_Regression_Combined.svg"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    
    # ==========================================
    # TOGGLE SMOOTH PLOTS ON / OFF HERE
    # ==========================================
    SMOOTH_PLOTS = True
    
    # Paths
    csv_file = r"D:\enviprojects\Berlin_Mehringdamm_Base\Berlin_Feinstaub_Messdaten.csv"
    fox_file = r"D:\enviprojects\Berlin_Feinstaub_Friedrichstr\new.fox"
    traffic_file = r"D:\enviprojects\Berlin_Feinstaub_Friedrichstr\trafficvolume.CSV"
    base_out_dir = r"D:\Berlin_Friedrichstr_CompResults_corrected_new_2_5m"
    
    # Define Subdirectories
    out_dirs = {
        'Diurnal': os.path.join(base_out_dir, "Diurnal"),
        'Regression': os.path.join(base_out_dir, "Regression"),
        'Regression_BG': os.path.join(base_out_dir, "Regression_BG"),
        'Stats': os.path.join(base_out_dir, "Stats"),
        'Aggregated': os.path.join(base_out_dir, "Aggregated"),
        'Cache': os.path.join(base_out_dir, "Data_Cache")
    }
    for directory in out_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # List all NetCDF folders to loop through
  #  netcdf_folders = [
  #      r"X:\Linde\Pascal\20240626_messstation3_3m_realBG2_lineSrc2_SourcesFix\NetCDF",
  #      r"X:\Linde\Pascal\20240708_messstation3_3m_realBG2_lineSrc2_SourcesFix\NetCDF",
  #      r"X:\Linde\Pascal\20240715_messstation3_3m_realBG2_lineSrc2_SourcesFix\NetCDF",
  #      r"X:\Linde\Pascal\20241106_messstation3_3m_realBG2_lineSrc3_SourcesFix\NetCDF",
  #      r"X:\Linde\Pascal\20241111_messstation3_3m_realBG2_lineSrc2\NetCDF",
  #      r"X:\Linde\Pascal\20241122_messstation3_3m_realBG2_lineSrc2\NetCDF"
  #  ]

    netcdf_folders = [
        r"X:\Linde\Pascal\20240626_05wind\NetCDF",
        r"X:\Linde\Pascal\20240708_05wind\NetCDF",
        r"X:\Linde\Pascal\20240715_05wind\NetCDF",
        r"X:\Linde\Pascal\20241106_05wind\NetCDF",
        r"X:\Linde\Pascal\20241111_05wind\NetCDF",
        r"X:\Linde\Pascal\20241122_05wind\NetCDF"
    ]    

    # X:\Linde\Pascal\20240626_05wind_newExe are wrong
    
    target_coords = (162, 126, 3)
    pollutants_to_plot = ['PM2.5', 'PM10', 'NO', 'NO2']
    
    # Defining plotting groupings
    pollutant_groups = {
        'Particulates': ['PM2.5', 'PM10'],
        'Nitrogen_Oxides': ['NO', 'NO2']
    }
    
    all_simulation_data = [] 
    
    for folder in netcdf_folders:
        print(f"\n================ Processing: {os.path.basename(os.path.normpath(folder.replace('NetCDF', '')))} ================")
        if not os.path.exists(folder):
            print(f"Skipping - Directory not found: {folder}")
            continue

        model_df, sim_name = load_envimet_series(folder, *target_coords, out_dirs['Cache']) 
        if model_df.empty:
            continue
            
        t_start = model_df.index.min()
        t_end = model_df.index.max()

        meas_df = load_measurements(csv_file, t_start, t_end)
        fox_df = load_fox_background(fox_file, t_start, t_end)
        traffic_df = load_traffic_volume(traffic_file, model_df.index)

        model_hourly = model_df.resample('1h').mean()
        common_idx = model_hourly.index.intersection(meas_df.index)

        if not common_idx.empty:
            common_idx = common_idx[common_idx > common_idx.min()]

        if common_idx.empty:
            print(f"Error: No overlapping timestamps found for {sim_name}!")
        else:
            c_meas = meas_df.loc[common_idx]
            c_mod = model_hourly.loc[common_idx]
            c_fox = fox_df.loc[common_idx]
            c_traf = traffic_df.loc[common_idx]
            
            all_simulation_data.append({
                'name': sim_name,
                'date_str': c_mod.index[0].strftime('%d.%m.%Y'),
                'meas': c_meas,
                'model': c_mod,
                'fox': c_fox,
                'traffic': c_traf
            })
            
            # --- PASSING SMOOTH TOGGLE ---
            plot_final_results(c_meas, c_mod, c_fox, c_traf, pollutants_to_plot, out_dirs, sim_name, target_coords, smooth_lines=SMOOTH_PLOTS)
            
    if all_simulation_data:
        plot_aggregated_regression(all_simulation_data, pollutant_groups, out_dirs['Aggregated'])
        
        # --- PASSING SMOOTH TOGGLE ---
        plot_aggregated_diurnal(all_simulation_data, pollutant_groups, out_dirs['Aggregated'], smooth_lines=SMOOTH_PLOTS)
        print("\nAll Processing Complete!")
    else:
        print("\nNo data processed to aggregate.")