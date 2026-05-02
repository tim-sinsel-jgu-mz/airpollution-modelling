import os
import glob
import json
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

#
# ### This script was fully recreated in claude.ai based on AirPollutionPlotting2.9.2_f1_f1.3.py
#


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


# ============================================================
# DATA LOADING FUNCTIONS  (unchanged from original)
# ============================================================

def load_measurements(csv_path, target_start, target_end):
    """Loads measurements and handles MEZ/MESZ switch by converting to UTC+1."""
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

        df.rename(columns={'PM10': 'PM10', 'PM2,5': 'PM2.5'}, inplace=True)
        df = df[['PM10', 'PM2.5']].apply(pd.to_numeric, errors='coerce')
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
            dt = pd.to_datetime(ts['date'] + ' ' + ts['time'])
            dt = dt.replace(year=target_start.year)
            pol = ts['backgrPollutants']
            records.append({
                'Datetime': dt,
                'PM10_BG': pol.get('PM10', 0),
                'PM2.5_BG': pol.get('PM25', 0),
            })

        df_fox = pd.DataFrame(records).set_index('Datetime').sort_index()
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
    if not nc_files:
        return pd.DataFrame(), "Unknown"

    sim_name = os.path.splitext(os.path.basename(nc_files[0]))[0]
    cache_file = os.path.join(cache_dir, f"Extracted_{sim_name}_X{x_idx}_Y{y_idx}_Z{z_idx}.csv")

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        data_frames = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                pt = ds.isel(GridsI=x_idx, GridsJ=y_idx, GridsK=z_idx)
                chunk = pd.DataFrame({
                    'PM2.5': pt['PM25Conc'].values,
                    'PM10': pt['PM10Conc'].values
                }, index=pd.to_datetime(pt['Time'].values))
                data_frames.append(chunk)
        df = pd.concat(data_frames).sort_index()
        df.to_csv(cache_file)

    df.index = df.index.round('1min')
    return df, sim_name


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(x, y, increment_mode=False):
    """
    Calculates evaluation statistics between observed (x) and modelled (y).

    Parameters
    ----------
    x, y : pd.Series
        Observed and modelled values on a common index.
    increment_mode : bool
        When True, FAC2 and FB are set to NaN because both metrics are
        undefined / unreliable when values can be near-zero or negative
        (as is the case for traffic increments).

    Returns dict of metrics, or None if fewer than 2 valid pairs.
    """
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 2:
        return None

    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b

    r_pearson, p_pearson = pearsonr(x, y)
    r2_pearson = r_pearson ** 2
    rho_spearman, p_spearman = spearmanr(x, y)
    r2_reg = r2_score(x, y)

    mean_obs = np.mean(x)
    mean_mod = np.mean(y)

    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    rmse_s = np.sqrt(np.mean((y_hat - x) ** 2))
    rmse_u = np.sqrt(np.mean((y - y_hat) ** 2))
    nrmse = rmse / mean_obs if mean_obs != 0 else np.nan
    nmse = np.mean((x - y) ** 2) / (mean_obs * mean_mod) if (mean_obs * mean_mod) != 0 else np.nan

    mean_bias = np.mean(y - x)
    nmb = mean_bias / mean_obs if mean_obs != 0 else np.nan

    # FAC2 and FB are meaningless for near-zero / negative increment values
    if increment_mode:
        fb = np.nan
        fac2 = np.nan
    else:
        fb = 2 * np.mean((y - x) / (y + x))
        ratio = y / x
        fac2 = np.mean((ratio >= 0.5) & (ratio <= 2.0))

    denominator = np.sum((np.abs(y - mean_obs) + np.abs(x - mean_obs)) ** 2)
    ioa = 1 - (np.sum((y - x) ** 2) / denominator) if denominator != 0 else np.nan

    return {
        "r": r_pearson,
        "p_pearson": p_pearson,
        "r2_pearson": r2_pearson,
        "r2_reg": r2_reg,
        "rho": rho_spearman,
        "p_spearman": p_spearman,
        "n": n,
        "mean_obs": mean_obs,
        "mean_mod": mean_mod,
        "mae": mae,
        "nmse": nmse,
        "rmse": rmse,
        "rmse_s": rmse_s,
        "rmse_u": rmse_u,
        "nrmse": nrmse,
        "mean_bias": mean_bias,
        "nmb": nmb,
        "fb": fb,
        "fac2": fac2,
        "ioa": ioa,
        "m": m,
        "b": b
    }


# ============================================================
# HELPERS
# ============================================================

def get_incremented_filename(out_dir, base_name):
    """Returns a non-colliding filename by appending _001, _002, …"""
    name, ext = os.path.splitext(base_name)
    counter = 1
    new_name = f"{name}_{counter}{ext}"
    full_path = os.path.join(out_dir, new_name)
    while os.path.exists(full_path):
        counter += 1
        new_name = f"{name}_{counter}{ext}"
        full_path = os.path.join(out_dir, new_name)
    return full_path


def compute_traffic_increments(df_meas, df_model, df_fox, pollutants):
    """
    Subtracts the Wedding-station background (df_fox) from both the
    Leipziger Straße measurements and the ENVI-met model output to isolate
    the local traffic signal.

    Returns
    -------
    df_meas_inc, df_model_inc : pd.DataFrame
        DataFrames with the same columns as df_meas / df_model but containing
        only the traffic increment (concentration above background).
    """
    df_meas_inc = pd.DataFrame(index=df_meas.index)
    df_model_inc = pd.DataFrame(index=df_model.index)

    for pol in pollutants:
        bg_col = f"{pol}_BG"
        if bg_col not in df_fox.columns:
            print(f"  WARNING: Background column '{bg_col}' not found – skipping {pol}.")
            continue

        bg = df_fox[bg_col]

        # Align background to the common index before subtracting
        bg_aligned = bg.reindex(df_meas.index)

        df_meas_inc[pol] = df_meas[pol] - bg_aligned
        df_model_inc[pol] = df_model[pol] - bg_aligned

    print(f"\n--- Traffic Increments computed for: {pollutants} ---")
    print(f"    Measured increment preview:\n{df_meas_inc.describe().round(2)}\n")
    print(f"    Modelled increment preview:\n{df_model_inc.describe().round(2)}\n")
    return df_meas_inc, df_model_inc


# ============================================================
# ORIGINAL PLOTS  (identical to original script)
# ============================================================

def plot_final_results(df_meas, df_model, df_fox, df_traffic, pollutants,
                       out_dir, sim_name, coords):
    """Generates the original diurnal and regression plots (unchanged)."""
    stats_export_list = []
    x_idx, y_idx, z_idx = coords
    header_str = f"Simulation: {sim_name} | Grid: X={x_idx}, Y={y_idx}, Z={z_idx}"

    # ------------------------------------------------------------------
    # 1. DIURNAL PLOT
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(header_str, fontsize=14, fontweight='bold', y=0.98)

    for i, pol in enumerate(pollutants):
        ax = axes[i]
        stats = calculate_statistics(df_meas[pol], df_model[pol])

        if stats is not None:
            p_val_str = f"{stats['p_pearson']:.3f}" if stats['p_pearson'] >= 0.001 else "< 0.001"
            stats_str = (
                f"$\\bar{{O}} = {stats['mean_obs']:.2f}$\n"
                f"$\\bar{{P}} = {stats['mean_mod']:.2f}$\n\n"
                f"$r = {stats['r']:.2f}$ ($p={p_val_str}$)\n"
                f"$r^2 = {stats['r2_pearson']:.2f}$\n"
                f"ρ = {stats['rho']:.2f}\n\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"$RMSE_s$ = {stats['rmse_s']:.2f}\n"
                f"$RMSE_u$ = {stats['rmse_u']:.2f}\n"
                f"NRMSE = {stats['nrmse']:.2f}\n"
                f"MAE = {stats['mae']:.2f}\n\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"NMB = {stats['nmb']:.2f}\n"
                f"FB = {stats['fb']:.2f}\n\n"
                f"FAC2 = {stats['fac2']:.2f}\n"
                f"d = {stats['ioa']:.2f}\n"
            )
        else:
            stats_str = "No Data"

        m_diurnal = df_meas.groupby(df_meas.index.hour)[pol].mean().reindex(range(1, 24))
        s_diurnal = df_model.groupby(df_model.index.hour)[pol].mean().reindex(range(1, 24))
        f_diurnal = df_fox.groupby(df_fox.index.hour)[f"{pol}_BG"].mean().reindex(range(1, 24))
        t_diurnal = df_traffic.groupby(df_traffic.index.hour)['Traffic'].mean().reindex(range(1, 24))

        ax2 = ax.twinx()
        ax2.fill_between(t_diurnal.index, 0, t_diurnal, color='gray', alpha=0.15, label='Traffic')
        ax2.set_ylabel("Traffic Vol. [Veh./h]", color='gray', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, None)

        ax.plot(m_diurnal.index, m_diurnal, 'k-', label='Measured', zorder=5)
        ax.plot(s_diurnal.index, s_diurnal, '#D62728', label='Modelled', zorder=6)
        ax.plot(f_diurnal.index, f_diurnal, color='gray', linestyle='--', label='Background', zorder=4)

        ax.text(1.25, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(f"Diurnal Cycle: {pol}", fontweight='bold')
        ax.set_ylabel("Conc. [µg m$^{-3}$]")
        ax.set_xlabel("Hour of Day")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_ylim(0, None)
        ax.grid(True, which='major', linestyle=':', alpha=0.6)

        if i == 0:
            handles_main, labels_main = ax.get_legend_handles_labels()
            handles_sec, labels_sec = ax2.get_legend_handles_labels()

    handles = handles_main + handles_sec
    labels = labels_main + labels_sec
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.4, -0.02))
    plt.subplots_adjust(right=0.75, left=0.1, bottom=0.1, top=0.92, hspace=0.3)

    for ext in ('png', 'svg'):
        save_path = get_incremented_filename(out_dir, f"Diurnal_{sim_name}.{ext}")
        plt.savefig(save_path)

    # ------------------------------------------------------------------
    # 2. REGRESSION PLOT  (Model vs Measured)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(header_str, fontsize=14, fontweight='bold', y=0.98)

    for i, pol in enumerate(pollutants):
        ax = axes[i]
        x_raw, y_raw = df_meas[pol], df_model[pol]
        mask = x_raw.notna() & y_raw.notna()
        x, y = x_raw[mask], y_raw[mask]

        if len(x) > 1:
            stats = calculate_statistics(x, y)
            m, b = np.polyfit(x, y, 1)
            stats_row = {'Pollutant': pol, 'Comparison': 'Model vs Measured', 'm': m, 'b': b}
            stats_row.update(stats)
            stats_export_list.append(stats_row)

            ax.scatter(x, y, alpha=0.5, s=20, edgecolors='none')
            lim = max(x.max(), y.max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            ax.plot(x, m * x + b, color='#D62728', linewidth=1.5, label='Fit')
            ax.plot([0, lim], [0, 0.5 * lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2', zorder=1)
            ax.plot([0, lim], [0, 2 * lim], 'k--', alpha=0.2, linewidth=0.8, zorder=1)

            p_val_str = f"{stats['p_pearson']:.3f}" if stats['p_pearson'] >= 0.001 else "< 0.001"
            stats_str = (
                f"$y = {m:.2f}x + {b:.2f}$\n"
                f"$\\bar{{O}} = {stats['mean_obs']:.2f}$\n"
                f"$\\bar{{P}} = {stats['mean_mod']:.2f}$\n\n"
                f"$r = {stats['r']:.2f}$ ($p={p_val_str}$)\n"
                f"$r^2 = {stats['r2_pearson']:.2f}$\n"
                f"ρ = {stats['rho']:.2f}\n\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"$RMSE_s$ = {stats['rmse_s']:.2f}\n"
                f"$RMSE_u$ = {stats['rmse_u']:.2f}\n"
                f"NRMSE = {stats['nrmse']:.2f}\n"
                f"MAE = {stats['mae']:.2f}\n\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"NMB = {stats['nmb']:.2f}\n"
                f"FB = {stats['fb']:.2f}\n\n"
                f"FAC2 = {stats['fac2']:.2f}\n"
                f"d = {stats['ioa']:.2f}\n"
            )

            ax.text(1.1, 0.5, stats_str, transform=ax.transAxes, verticalalignment='center',
                    horizontalalignment='left', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
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

    plt.subplots_adjust(right=0.75, left=0.1, bottom=0.1, top=0.92, hspace=0.3)
    for ext in ('png', 'svg'):
        save_path = get_incremented_filename(out_dir, f"Regression_{sim_name}.{ext}")
        plt.savefig(save_path)

    # ------------------------------------------------------------------
    # 3. BACKGROUND REGRESSION PLOT
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(f"{header_str}\nModel vs Background", fontsize=14, fontweight='bold', y=0.98)

    for i, pol in enumerate(pollutants):
        ax = axes[i]
        x_raw, y_raw = df_fox[f"{pol}_BG"], df_model[pol]
        mask = x_raw.notna() & y_raw.notna()
        x, y = x_raw[mask], y_raw[mask]

        if len(x) > 1:
            stats = calculate_statistics(x, y)
            m, b = np.polyfit(x, y, 1)
            stats_row = {'Pollutant': pol, 'Comparison': 'Model vs Background', 'm': m, 'b': b}
            stats_row.update(stats)
            stats_export_list.append(stats_row)

            ax.scatter(x, y, alpha=0.5, s=20, edgecolors='none')
            lim = max(x.max(), y.max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            ax.plot(x, m * x + b, color='#D62728', linewidth=1.5, label='Fit')
            ax.plot([0, lim], [0, 0.5 * lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2', zorder=1)
            ax.plot([0, lim], [0, 2 * lim], 'k--', alpha=0.2, linewidth=0.8, zorder=1)

            p_val_str = f"{stats['p_pearson']:.3f}" if stats['p_pearson'] >= 0.001 else "< 0.001"
            stats_str = (
                f"$y = {m:.2f}x + {b:.2f}$\n"
                f"$\\bar{{O}} = {stats['mean_obs']:.2f}$\n"
                f"$\\bar{{P}} = {stats['mean_mod']:.2f}$\n\n"
                f"$r = {stats['r']:.2f}$ ($p={p_val_str}$)\n"
                f"$r^2 = {stats['r2_pearson']:.2f}$\n"
                f"ρ = {stats['rho']:.2f}\n\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"$RMSE_s$ = {stats['rmse_s']:.2f}\n"
                f"$RMSE_u$ = {stats['rmse_u']:.2f}\n"
                f"NRMSE = {stats['nrmse']:.2f}\n"
                f"MAE = {stats['mae']:.2f}\n\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"NMB = {stats['nmb']:.2f}\n"
                f"FB = {stats['fb']:.2f}\n\n"
                f"FAC2 = {stats['fac2']:.2f}\n"
                f"d = {stats['ioa']:.2f}\n"
            )

            ax.text(1.1, 0.5, stats_str, transform=ax.transAxes, verticalalignment='center',
                    horizontalalignment='left', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
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

    plt.subplots_adjust(right=0.75, left=0.1, bottom=0.1, top=0.92, hspace=0.3)
    for ext in ('png', 'svg'):
        save_path = get_incremented_filename(out_dir, f"Regression_BG_{sim_name}.{ext}")
        plt.savefig(save_path)

    # --- Measured vs Background stats (no plot, stats only) ---
    for pol in pollutants:
        x_raw, y_raw = df_fox[f"{pol}_BG"], df_meas[pol]
        mask = x_raw.notna() & y_raw.notna()
        x, y = x_raw[mask], y_raw[mask]
        if len(x) > 1:
            stats = calculate_statistics(x, y)
            m, b = np.polyfit(x, y, 1)
            stats_row = {'Pollutant': pol, 'Comparison': 'Measured vs Background', 'm': m, 'b': b}
            stats_row.update(stats)
            stats_export_list.append(stats_row)

    # --- Export original stats ---
    if stats_export_list:
        base_filename = f"Stats_{os.path.basename(netcdf_folder.replace(r'/NetCDF', '').replace(r'\\NetCDF', ''))}.csv"
        save_path = get_incremented_filename(out_dir, base_filename)
        pd.DataFrame(stats_export_list).to_csv(save_path, index=False, sep=';', decimal=',')
        print(f"Original regression stats saved to: {save_path}")


# ============================================================
# NEW: TRAFFIC INCREMENT PLOTS
# ============================================================

def plot_increment_results(df_meas_inc, df_model_inc, df_traffic, pollutants,
                           out_dir, sim_name, coords):
    """
    Generates diurnal and scatter/regression plots for the traffic increment
    (concentration above background).

    The traffic increment is defined as:
        increment = raw_concentration - background_concentration

    FAC2 and FB are omitted for increment data because both metrics require
    strictly positive values and are undefined / misleading when increments
    can be near-zero or negative.
    """
    stats_export_list = []
    x_idx, y_idx, z_idx = coords
    header_str = (
        f"Simulation: {sim_name} | Grid: X={x_idx}, Y={y_idx}, Z={z_idx}\n"
        f"Traffic Increment (Conc. − Background)"
    )

    # ------------------------------------------------------------------
    # 1. INCREMENT DIURNAL PLOT
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(header_str, fontsize=13, fontweight='bold', y=0.99)

    for i, pol in enumerate(pollutants):
        ax = axes[i]

        # Stats on the full hourly increment time series
        stats = calculate_statistics(df_meas_inc[pol], df_model_inc[pol],
                                     increment_mode=True)

        if stats is not None:
            p_val_str = f"{stats['p_pearson']:.3f}" if stats['p_pearson'] >= 0.001 else "< 0.001"
            # FB and FAC2 are NaN in increment mode — show a note instead of the value
            fac2_str = "N/A*" if np.isnan(stats['fac2']) else f"{stats['fac2']:.2f}"
            fb_str   = "N/A*" if np.isnan(stats['fb'])   else f"{stats['fb']:.2f}"

            stats_str = (
                f"$\\bar{{O_{{inc}}}} = {stats['mean_obs']:.2f}$\n"
                f"$\\bar{{P_{{inc}}}} = {stats['mean_mod']:.2f}$\n\n"
                f"$r = {stats['r']:.2f}$ ($p={p_val_str}$)\n"
                f"$r^2 = {stats['r2_pearson']:.2f}$\n"
                f"ρ = {stats['rho']:.2f}\n\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"$RMSE_s$ = {stats['rmse_s']:.2f}\n"
                f"$RMSE_u$ = {stats['rmse_u']:.2f}\n"
                f"NRMSE = {stats['nrmse']:.2f}\n"
                f"MAE = {stats['mae']:.2f}\n\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"NMB = {stats['nmb']:.2f}\n"
                f"FB = {fb_str}\n\n"
                f"FAC2 = {fac2_str}\n"
                f"d = {stats['ioa']:.2f}\n"
                f"\n* N/A: undefined for\n  near-zero/negative\n  increments"
            )
        else:
            stats_str = "No Data"

        # Diurnal averages of increments
        m_diurnal = df_meas_inc.groupby(df_meas_inc.index.hour)[pol].mean().reindex(range(1, 24))
        s_diurnal = df_model_inc.groupby(df_model_inc.index.hour)[pol].mean().reindex(range(1, 24))
        t_diurnal = df_traffic.groupby(df_traffic.index.hour)['Traffic'].mean().reindex(range(1, 24))

        # Traffic fill (secondary axis)
        ax2 = ax.twinx()
        ax2.fill_between(t_diurnal.index, 0, t_diurnal, color='gray', alpha=0.15, label='Traffic')
        ax2.set_ylabel("Traffic Vol. [Veh./h]", color='gray', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, None)

        # Increment lines
        ax.plot(m_diurnal.index, m_diurnal, 'k-', label='Meas. Increment', zorder=5)
        ax.plot(s_diurnal.index, s_diurnal, '#D62728', label='Model Increment', zorder=6)

        # Zero reference line — visually marks the background level
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6,
                   label='Background (0)', zorder=3)

        ax.text(1.25, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top',
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(f"Diurnal Traffic Increment: {pol}", fontweight='bold')
        ax.set_ylabel("Increment [µg m$^{-3}$]")
        ax.set_xlabel("Hour of Day")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which='major', linestyle=':', alpha=0.6)
        # NOTE: y-axis is NOT forced to 0 — negative increments are physically valid

        if i == 0:
            handles_main, labels_main = ax.get_legend_handles_labels()
            handles_sec, labels_sec = ax2.get_legend_handles_labels()

    handles = handles_main + handles_sec
    labels = labels_main + labels_sec
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.4, -0.02))
    plt.subplots_adjust(right=0.75, left=0.1, bottom=0.1, top=0.92, hspace=0.3)

    for ext in ('png', 'svg'):
        save_path = get_incremented_filename(out_dir, f"Diurnal_Increment_{sim_name}.{ext}")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    # ------------------------------------------------------------------
    # 2. INCREMENT SCATTER / REGRESSION PLOT
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(header_str, fontsize=13, fontweight='bold', y=0.99)

    for i, pol in enumerate(pollutants):
        ax = axes[i]
        x_raw = df_meas_inc[pol]
        y_raw = df_model_inc[pol]
        mask = x_raw.notna() & y_raw.notna()
        x, y = x_raw[mask], y_raw[mask]

        if len(x) > 1:
            stats = calculate_statistics(x, y, increment_mode=True)
            m, b = np.polyfit(x, y, 1)

            stats_row = {'Pollutant': pol, 'Comparison': 'Model vs Measured (Increment)',
                         'm': m, 'b': b}
            stats_row.update(stats)
            stats_export_list.append(stats_row)

            ax.scatter(x, y, alpha=0.5, s=20, edgecolors='none')

            # Axis limits: symmetric around the data range, not forced to 0
            all_vals = pd.concat([x, y])
            val_min = all_vals.min()
            val_max = all_vals.max()
            pad = (val_max - val_min) * 0.1
            lim_lo = val_min - pad
            lim_hi = val_max + pad

            # 1:1 line spanning the full range
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.3, label='1:1')
            # Regression line
            x_fit = np.array([lim_lo, lim_hi])
            ax.plot(x_fit, m * x_fit + b, color='#D62728', linewidth=1.5, label='Fit')
            # Zero crosshairs — visual anchor for the background level
            ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

            p_val_str = f"{stats['p_pearson']:.3f}" if stats['p_pearson'] >= 0.001 else "< 0.001"
            fac2_str = "N/A*" if np.isnan(stats['fac2']) else f"{stats['fac2']:.2f}"
            fb_str   = "N/A*" if np.isnan(stats['fb'])   else f"{stats['fb']:.2f}"

            stats_str = (
                f"$y = {m:.2f}x + {b:.2f}$\n"
                f"$\\bar{{O_{{inc}}}} = {stats['mean_obs']:.2f}$\n"
                f"$\\bar{{P_{{inc}}}} = {stats['mean_mod']:.2f}$\n\n"
                f"$r = {stats['r']:.2f}$ ($p={p_val_str}$)\n"
                f"$r^2 = {stats['r2_pearson']:.2f}$\n"
                f"ρ = {stats['rho']:.2f}\n\n"
                f"RMSE = {stats['rmse']:.2f}\n"
                f"$RMSE_s$ = {stats['rmse_s']:.2f}\n"
                f"$RMSE_u$ = {stats['rmse_u']:.2f}\n"
                f"NRMSE = {stats['nrmse']:.2f}\n"
                f"MAE = {stats['mae']:.2f}\n\n"
                f"MB = {stats['mean_bias']:.2f}\n"
                f"NMB = {stats['nmb']:.2f}\n"
                f"FB = {fb_str}\n\n"
                f"FAC2 = {fac2_str}\n"
                f"d = {stats['ioa']:.2f}\n"
                f"\n* N/A: undefined for\n  near-zero/negative\n  increments"
            )

            ax.text(1.1, 0.5, stats_str, transform=ax.transAxes, verticalalignment='center',
                    horizontalalignment='left', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_aspect('equal')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')

        ax.set_title(f"{pol} Increment Regression", fontweight='bold')
        ax.set_xlabel("Measured Increment [µg m$^{-3}$]")
        ax.set_ylabel("Modelled Increment [µg m$^{-3}$]")
        ax.legend(fontsize=9, loc='upper left')

    plt.subplots_adjust(right=0.75, left=0.1, bottom=0.1, top=0.90, hspace=0.3)

    for ext in ('png', 'svg'):
        save_path = get_incremented_filename(out_dir, f"Regression_Increment_{sim_name}.{ext}")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    # ------------------------------------------------------------------
    # Export increment stats to CSV
    # ------------------------------------------------------------------
    if stats_export_list:
        base_filename = (
            f"Stats_Increment_"
            f"{os.path.basename(netcdf_folder.replace(r'/NetCDF', '').replace(r'\\NetCDF', ''))}.csv"
        )
        save_path = get_incremented_filename(out_dir, base_filename)
        pd.DataFrame(stats_export_list).to_csv(save_path, index=False, sep=';', decimal=',')
        print(f"Increment stats saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # --- Paths (update these) ---
    csv_file      = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\Berlin_Feinstaub_Messdaten.csv"
    fox_file      = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\merge7_clean_2024_Jun_Nov_smthWind_realBG2_fix0626.FOX"
    traffic_file  = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting\TrafficVolume_messstation.CSV"
    netcdf_folder = r"Z:\Linde\Pascal\20241122_messstation3_3m_realBG2_lineSrc2\NetCDF"
    base_out_dir  = r"C:\Users\silik\OneDrive\JGU MAINZ\BACHELORARBEIT\THEMA Feinstaub Berlin\Phyton Scripts\Plotting"
    cache_dir     = os.path.join(base_out_dir, "Data_Cache")
    os.makedirs(cache_dir, exist_ok=True)

    target_coords = (134, 104, 3)  # area5_4m: 101,92,3  |  messstation3_3m: 134,104,3

    pollutants = ['PM2.5', 'PM10']

    # 1. Load model data first to define the time range
    model_df, sim_name = load_envimet_series(netcdf_folder, *target_coords, cache_dir)

    t_start = model_df.index.min()
    t_end   = model_df.index.max()

    # 2. Load remaining data limited to the simulation range
    meas_df    = load_measurements(csv_file, t_start, t_end)
    fox_df     = load_fox_background(fox_file, t_start, t_end)
    traffic_df = load_traffic_volume(traffic_file, model_df.index)

    # 3. Align everything to hourly snapshots
    model_hourly = model_df.resample('1h').mean()
    common_idx   = model_hourly.index.intersection(meas_df.index)

    # Drop the first aligned hour to skip model initialisation
    if not common_idx.empty:
        common_idx = common_idx[common_idx > common_idx.min()]

    if common_idx.empty:
        print("Error: No overlapping timestamps found between measurements and model!")
    else:
        df_meas_aligned    = meas_df.loc[common_idx]
        df_model_aligned   = model_hourly.loc[common_idx]
        df_fox_aligned     = fox_df.loc[common_idx]
        df_traffic_aligned = traffic_df.loc[common_idx]

        # --- A. Original plots (unchanged) ---
        plot_final_results(
            df_meas_aligned, df_model_aligned,
            df_fox_aligned, df_traffic_aligned,
            pollutants, base_out_dir, sim_name, target_coords
        )

        # --- B. Compute traffic increments ---
        # Measured increment  = Leipziger Str. measurement  − Wedding background
        # Modelled increment  = ENVI-met simulation output  − Wedding background
        # This isolates the local traffic signal from the background noise,
        # giving a true evaluation of the PyQGIS traffic aggregation tool.
        df_meas_inc, df_model_inc = compute_traffic_increments(
            df_meas_aligned, df_model_aligned, df_fox_aligned, pollutants
        )

        # --- C. Increment plots ---
        plot_increment_results(
            df_meas_inc, df_model_inc, df_traffic_aligned,
            pollutants, base_out_dir, sim_name, target_coords
        )

        print("\nProcessing Complete.")