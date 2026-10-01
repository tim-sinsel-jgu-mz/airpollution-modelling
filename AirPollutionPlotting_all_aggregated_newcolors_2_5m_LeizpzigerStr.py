import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import PchipInterpolator
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

COLORS = {
    # Particulates using 'hawaii' (Dark Purple to Bright Orange)
    'PM2.5': {'meas': cmc.hawaii(0.1), 'mod': cmc.hawaii(0.3), 'bg': cmc.hawaii(0.2)},
    'PM10':  {'meas': cmc.hawaii(0.6), 'mod': cmc.hawaii(0.8), 'bg': cmc.hawaii(0.7)},

    # Nitrogen Oxides using 'batlow' (Dark Blue/Green to Peach) to keep distinct from PMs
    'NO':    {'meas': cmc.batlow(0.1), 'mod': cmc.batlow(0.3), 'bg': cmc.batlow(0.2)},
    'NO2':   {'meas': cmc.batlow(0.6), 'mod': cmc.batlow(0.8), 'bg': cmc.batlow(0.7)}
}

# NetCDF variable and FOX background key per pollutant
NC_VARS = {'PM2.5': 'PM25Conc', 'PM10': 'PM10Conc', 'NO': 'NOConc', 'NO2': 'NO2Conc'}
FOX_KEYS = {'PM2.5': 'PM25', 'PM10': 'PM10', 'NO': 'NO', 'NO2': 'NO2'}

TRAFFIC_LABEL = "Traffic Volume [Veh/h]"

# Evaluated hours after the spin-up: EVAL_HOURS + 1 outputs, midnight to midnight
EVAL_HOURS = 24


# --- HELPERS ---
def smooth_data(series, num_points=300):
    """Shape-preserving (PCHIP) interpolation of a pandas Series for display.
    The curve passes through every point and never overshoots, so it adds no
    peaks, dips or negative values that are not in the data."""
    series = series.dropna()
    if len(series) < 3:
        return series.index.values, series.values
    x = series.index.values.astype(float)
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    return x_smooth, PchipInterpolator(x, series.values)(x_smooth)

def hours_since(series, t0):
    """The series against hours elapsed since t0: 00:00 of the evaluated day is 0, the next midnight 24."""
    out = series.copy()
    out.index = (series.index - t0) / pd.Timedelta(hours=1)
    return out

def hourly_curve(series, t0):
    """Hourly values at 0..EVAL_HOURS, NaN where an hour is missing."""
    return hours_since(series, t0).reindex(np.arange(EVAL_HOURS + 1, dtype=float))

def calculate_statistics(obs, mod):
    """Paired statistics of mod against obs, aligned on time."""
    obs, mod = obs.align(mod, join='inner')
    mask = obs.notna() & mod.notna()
    obs, mod = obs[mask].astype(float), mod[mask].astype(float)
    if len(obs) < 2:
        return None

    ratio = mod[obs > 0] / obs[obs > 0]
    m, b = np.polyfit(obs, mod, 1)
    return {
        "n": len(obs),
        "mean_meas": obs.mean(),
        "mean_mod": mod.mean(),
        "mb": (mod - obs).mean(),                       # mean bias
        "nmb": (mod - obs).sum() / obs.sum(),           # normalised mean bias
        "rmse": np.sqrt(((mod - obs) ** 2).mean()),
        "r2_pearson": np.corrcoef(obs, mod)[0, 1] ** 2, # squared Pearson r, blind to bias
        "fac2": ((ratio >= 0.5) & (ratio <= 2.0)).mean(),
        "m": m,
        "b": b,
    }

def stats_text(s, digits=2):
    return f"$R^2 = {s['r2_pearson']:.2f}$, RMSE = {s['rmse']:.{digits}f}, MB = {s['mb']:+.{digits}f}"


# --- LOADERS (each input file is read once) ---
def load_measurements(csv_path):
    print(f"--- Loading Measurements: {os.path.basename(csv_path)} ---")
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Datetime']).set_index('Datetime')

    # The CSV is in local civil time (CET/CEST); the model runs on fixed UTC+1.
    df.index = (df.index.tz_localize('Europe/Berlin', ambiguous=True, nonexistent='NaT')
                        .tz_convert('Etc/GMT-1').tz_localize(None))
    df = df[df.index.notna()]

    df = df.rename(columns={'PM2,5': 'PM2.5'})
    return df[['PM10', 'PM2.5', 'NO', 'NO2']].apply(pd.to_numeric, errors='coerce')

def load_fox_background(fox_path):
    print(f"--- Loading FOX Background Data ---")
    with open(fox_path, 'r') as f:
        steps = json.load(f)['timestepList']

    idx = pd.to_datetime([s['date'] + ' ' + s['time'] for s in steps], format='ISO8601')
    df = pd.DataFrame({f"{pol}_BG": [s['backgrPollutants'].get(key, np.nan) for s in steps]
                       for pol, key in FOX_KEYS.items()}, index=idx)
    df = df.where(df > -998)  # FOX no-data value -999
    return df.groupby(level=0).mean().sort_index()

def load_traffic_profile(traffic_path):
    print(f"--- Loading Traffic Data ---")
    df_t = pd.read_csv(traffic_path, sep=';')
    hours = pd.to_datetime(df_t['Time'], format='%H:%M:%S').dt.hour
    return pd.Series(df_t['TrajCount'].values, index=hours.values, name='Traffic')

def find_nc_files(source):
    """A scenario source is either a NetCDF folder or a single .nc file."""
    if source.lower().endswith('.nc'):
        return [source] if os.path.isfile(source) else []
    return sorted(glob.glob(os.path.join(source, "*.nc")))

def load_envimet_series(nc_files, sim_name, x_idx, y_idx, z_idx, cache_dir):
    """Receptor-cell time series. The cache name carries size and date of the source
    files, so a re-run simulation is extracted again instead of read from a stale cache."""
    signature = "|".join(f"{os.path.basename(f)}:{os.path.getsize(f)}:{os.path.getmtime(f):.0f}" for f in nc_files)
    tag = hashlib.md5(signature.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"Extracted_{sim_name}_X{x_idx}_Y{y_idx}_Z{z_idx}_{tag}.csv")

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        chunks = []
        for f in nc_files:
            print(f"    extracting cell from {os.path.basename(f)} (slow on first run, then cached)")
            with xr.open_dataset(f) as ds:
                present = {pol: var for pol, var in NC_VARS.items() if var in ds}
                pt = ds[list(present.values())].isel(GridsI=x_idx, GridsJ=y_idx, GridsK=z_idx).load()
                chunk = pd.DataFrame({pol: pt[var].values for pol, var in present.items()},
                                     index=pd.to_datetime(pt['Time'].values))
            chunks.append(chunk.reindex(columns=list(NC_VARS)))
        df = pd.concat(chunks).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df.to_csv(cache_file)

    df.index = df.index.round('1min')
    return df

def evaluation_window(model_df, spinup_hours):
    """The hourly outputs after the spin-up, midnight to midnight. The first NetCDF
    record is the initial state at the simulation start, so a 23:00 start with 1 h
    spin-up evaluates 00:00 of the following day to 00:00 of the day after (the
    final output at 23:59:58 is rounded to that midnight)."""
    start = model_df.index.min() + pd.Timedelta(hours=spinup_hours)
    return pd.date_range(start, periods=EVAL_HOURS + 1, freq='1h')


# --- PER-SCENARIO PLOTS ---
def plot_regression_grid(sim, pollutants, against, out_path, stats_rows):
    """2x2 scatter of modelled values against the measurements ('meas') or the forced background ('bg')."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, pol in zip(axes.flatten(), pollutants):
        c = COLORS[pol]
        x_raw = sim['meas'][pol] if against == 'meas' else sim['bg'][f"{pol}_BG"]
        y_raw = sim['model'][pol]
        s = calculate_statistics(x_raw, y_raw)

        if s is not None:
            row = {'Pollutant': pol, 'Comparison': 'Model vs Measured' if against == 'meas' else 'Model vs Background'}
            row.update(s)
            stats_rows.append(row)

            x, y = x_raw.align(y_raw, join='inner')
            mask = x.notna() & y.notna()
            x, y = x[mask], y[mask]
            ax.scatter(x, y, color=c['mod'], alpha=0.5, s=20, edgecolors='none')
            lim = max(x.max(), y.max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
            ax.plot(x, s['m'] * x + s['b'], color=c['meas'], linewidth=1.5, label='Fit')
            ax.plot([0, lim], [0, 0.5 * lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2', zorder=1)
            ax.plot([0, lim], [0, 2 * lim], 'k--', alpha=0.2, linewidth=0.8, zorder=1)

            ax.text(0.05, 0.95, stats_text(s).replace(', ', '\n'), transform=ax.transAxes, verticalalignment='top',
                    horizontalalignment='left', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')

        if against == 'meas':
            ax.set_title(f"{pol} Regression", fontweight='bold')
            ax.set_xlabel("Measured [µg m$^{-3}$]")
        else:
            ax.set_title(f"{pol} Regression (Background)", fontweight='bold')
            ax.set_xlabel("Background [µg m$^{-3}$]")
        ax.set_ylabel("Modelled [µg m$^{-3}$]")

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(out_path + ".png")
    plt.savefig(out_path + ".svg")
    plt.close()

def plot_diurnal_lines(ax, sim, pol, smooth_lines):
    """Measured, modelled and background diurnal lines of one pollutant; returns their maximum."""
    c = COLORS[pol]
    t0 = sim['t0']
    curves = [(hourly_curve(sim['meas'][pol], t0), c['meas'], '-', 1.0, f'Meas {pol}', 5),
              (hourly_curve(sim['model'][pol], t0), c['mod'], '-', 1.0, f'Mod {pol}', 6),
              (hours_since(sim['fox'][f"{pol}_BG"], t0), c['bg'], '--', 0.6, f'BG {pol}', 4)]
    for series, color, ls, alpha, label, z in curves:
        x, y = smooth_data(series) if smooth_lines else (series.index.values, series.values)
        ax.plot(x, y, color=color, linestyle=ls, alpha=alpha, label=label, zorder=z)
    return np.nanmax(np.concatenate([s.values for s, *_ in curves]))

def plot_traffic(ax2, sim, smooth_lines):
    t_diurnal = hourly_curve(sim['traffic']['Traffic'], sim['t0'])
    x, y = smooth_data(t_diurnal) if smooth_lines else (t_diurnal.index.values, t_diurnal.values)
    ax2.fill_between(x, 0, y, color='gray', alpha=0.15, label='Traffic')
    return np.nanmax(t_diurnal.values)

def plot_final_results(sim, pollutants, out_dir_dict, smooth_lines=False):
    sim_name = sim['name']
    stats_export_list = []

    # 1. DIURNAL PLOT (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (ax, pol) in enumerate(zip(axes.flatten(), pollutants)):
        s = calculate_statistics(sim['meas'][pol], sim['model'][pol])
        ax2 = ax.twinx()
        plot_traffic(ax2, sim, smooth_lines)
        max_y = plot_diurnal_lines(ax, sim, pol, smooth_lines)

        if i % 2 == 1:
            ax2.set_ylabel(TRAFFIC_LABEL, color='black', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, None)

        ax.text(0.05, 0.95, stats_text(s).replace(', ', '\n') if s else "No Data", transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.set_title(f"Diurnal Cycle: {pol}", fontweight='bold')
        ax.set_ylabel("Concentration [µg m$^{-3}$]")
        ax.set_xlabel("Hour")
        ax.set_xlim(0, EVAL_HOURS)
        ax.set_xticks(range(0, EVAL_HOURS + 1, 4))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_ylim(0, max_y * 1.20 if np.isfinite(max_y) else None)
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

    # 2. REGRESSION PLOTS (2x2): against the measurements and against the forced background
    plot_regression_grid(sim, pollutants, 'meas', os.path.join(out_dir_dict['Regression'], f"Regression_{sim_name}"), stats_export_list)
    plot_regression_grid(sim, pollutants, 'bg', os.path.join(out_dir_dict['Regression_BG'], f"Regression_BG_{sim_name}"), stats_export_list)

    if stats_export_list:
        save_path = os.path.join(out_dir_dict['Stats'], f"Stats_{sim_name}.csv")
        pd.DataFrame(stats_export_list).to_csv(save_path, index=False, sep=';', decimal=',')


# --- AGGREGATED PLOTS ---
def plot_aggregated_diurnal(all_sim_data, scenario_names, pollutant_groups, out_dir, smooth_lines=False):
    """One panel per configured scenario, in the configured order; scenarios without data stay empty."""
    print("--- Generating Aggregated Diurnal Plots ---")
    if not all_sim_data:
        return
    sims = {sim['name']: sim for sim in all_sim_data}
    n_panels = len(scenario_names)
    cols = 2 if n_panels > 1 else 1
    rows = int(np.ceil(n_panels / cols))

    traffic_max = np.nanmax([np.nanmax(s['traffic']['Traffic'].values) for s in all_sim_data])
    y_lim_traf = traffic_max * 1.20 if traffic_max > 0 else 100

    for group_name, pols in pollutant_groups.items():
        conc_max = np.nanmax([np.nanmax(np.r_[s['meas'][p].values, s['model'][p].values, s['fox'][f"{p}_BG"].values])
                              for s in all_sim_data for p in pols])
        y_lim_conc = conc_max * 1.20 if conc_max > 0 else 45

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.5), squeeze=False)
        ax_flat = axes.flatten()
        legend_ax = None

        for i, name in enumerate(scenario_names):
            ax = ax_flat[i]
            ax.set_title(pd.to_datetime(name, format='%Y%m%d').strftime('%d.%m.%Y'), fontweight='bold')
            ax.set_xlim(0, EVAL_HOURS)
            ax.set_ylim(0, y_lim_conc)
            ax.set_xticks(range(0, EVAL_HOURS + 1, 4))
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.grid(True, which='major', linestyle=':', alpha=0.6)

            ax2 = ax.twinx()
            ax2.set_ylim(0, y_lim_traf)
            if (i % cols) == (cols - 1):
                ax2.set_ylabel(TRAFFIC_LABEL, color='black', fontsize=11)
            else:
                ax2.tick_params(labelright=False)

            if (i % cols) == 0:
                ax.set_ylabel("Concentration [µg m$^{-3}$]")
            else:
                ax.tick_params(labelleft=False)

            if i >= (rows - 1) * cols:
                ax.set_xlabel("Hour of the Day")
            else:
                ax.tick_params(labelbottom=False)

            sim = sims.get(name)
            if sim is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center', color='gray')
                continue

            plot_traffic(ax2, sim, smooth_lines)
            stats_texts = []
            for pol in pols:
                plot_diurnal_lines(ax, sim, pol, smooth_lines)
                s = calculate_statistics(sim['meas'][pol], sim['model'][pol])
                if s:
                    stats_texts.append(f"{pol}: {stats_text(s, digits=1)}")
            ax.text(0.03, 0.96, "\n".join(stats_texts) if stats_texts else "No Data", transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            if legend_ax is None:
                legend_ax = (ax, ax2)

        for j in range(n_panels, rows * cols):
            fig.delaxes(ax_flat[j])

        handles_main, labels_main = legend_ax[0].get_legend_handles_labels()
        handles_sec, labels_sec = legend_ax[1].get_legend_handles_labels()
        by_label = dict(zip(labels_main + labels_sec, handles_main + handles_sec))

        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.94, wspace=0.08, hspace=0.13)
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=len(by_label), frameon=False, bbox_to_anchor=(0.5, 0.02))
        plt.savefig(os.path.join(out_dir, f"Aggregated_Diurnal_{group_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f"Aggregated_Diurnal_{group_name}.svg"), bbox_inches='tight')
        plt.close()

def pooled(all_sim_data, frame, column):
    """One series over all scenarios; the timestamps of the scenarios do not overlap."""
    return pd.concat([sim[frame][column] for sim in all_sim_data])

def plot_aggregated_regression(all_sim_data, pollutant_groups, out_dir):
    print("--- Generating Aggregated Regression Plot ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_flat = axes.flatten()

    for i, (group_name, pols) in enumerate(pollutant_groups.items()):
        ax = ax_flat[i]
        global_max = 0
        stats_texts = []

        for pol in pols:
            c = COLORS[pol]
            x_all, y_all = pooled(all_sim_data, 'meas', pol), pooled(all_sim_data, 'model', pol)
            s = calculate_statistics(x_all, y_all)
            if s is None:
                continue
            x, y = x_all.align(y_all, join='inner')
            mask = x.notna() & y.notna()
            x, y = x[mask], y[mask]

            ax.scatter(x, y, color=c['meas'], alpha=0.5, s=20, edgecolors='none', label=f'{pol} Data')
            ax.plot(x, s['m'] * x + s['b'], color=c['meas'], linewidth=2.0, linestyle='-', label=f'{pol} Fit')
            global_max = max(global_max, x.max(), y.max())
            stats_texts.append(f"{pol}: {stats_text(s, digits=1)}")

        lim = global_max * 1.1 if global_max > 0 else 100
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1' if i == 0 else "")
        ax.plot([0, lim], [0, 0.5 * lim], 'k--', alpha=0.2, linewidth=0.8, label='FAC2' if i == 0 else "")
        ax.plot([0, lim], [0, 2 * lim], 'k--', alpha=0.2, linewidth=0.8)

        ax.text(0.05, 0.95, "\n".join(stats_texts) if stats_texts else "Insufficient Data", transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(f"{group_name.replace('_', ' ')} Aggregated", fontweight='bold')
        ax.set_xlabel("Measured [µg m$^{-3}$]")
        ax.set_ylabel("Modelled [µg m$^{-3}$]")

    handles, labels = [], []
    for ax in ax_flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    by_label.pop("", None)
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=11)

    plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.15)
    plt.savefig(os.path.join(out_dir, "Aggregated_Regression_Combined.png"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Aggregated_Regression_Combined.svg"), bbox_inches='tight')
    plt.close()

def export_aggregated_statistics(all_sim_data, pollutants, out_dir):
    """Pooled statistics of the model, of the forced background alone (the reference the model
    has to beat), and of the local increment above the background (model - BG vs measured - BG)."""
    rows = []
    for pol in pollutants:
        meas = pooled(all_sim_data, 'meas', pol)
        mod = pooled(all_sim_data, 'model', pol)
        bg = pooled(all_sim_data, 'bg', f"{pol}_BG")
        for comparison, obs, pred in [('Model vs Measured', meas, mod),
                                      ('Background vs Measured', meas, bg),
                                      ('Increment: Model-BG vs Measured-BG', meas - bg, mod - bg)]:
            s = calculate_statistics(obs, pred)
            if s is not None:
                rows.append({'Pollutant': pol, 'Comparison': comparison, **s})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "Stats_Aggregated.csv"), index=False, sep=';', decimal=',')
    print("\n--- Pooled statistics over all scenarios ---")
    print(df[['Pollutant', 'Comparison', 'n', 'mean_meas', 'mean_mod', 'mb', 'nmb', 'rmse', 'r2_pearson', 'fac2']]
          .to_string(index=False, float_format=lambda v: f"{v:7.2f}"))


if __name__ == "__main__":

    # TOGGLE SMOOTH PLOTS ON / OFF HERE
    SMOOTH_PLOTS = True
    SPINUP_HOURS = 1

    # Paths
    csv_file = r"D:\enviprojects\Berlin_Mehringdamm_Base\Berlin_Feinstaub_Messdaten.csv" 
    fox_file = r"D:\enviprojects\Berlin_Pollutants_LeipzigerStr\final.fox"
    traffic_file = r"D:\enviprojects\Berlin_Pollutants_LeipzigerStr\trafficvolume.CSV"
    base_out_dir = r"D:\Berlin_Friedrichstr_CompResults_corrected_new_2_5m_V6"

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

    # Scenario name (evaluated day) -> NetCDF folder or single .nc file
    scenarios = {
        '20240626': r"Y:\BerlinLeipzigerStr\20240626\NetCDF",
        '20240708': r"Y:\BerlinLeipzigerStr\20240708\NetCDF",
        '20240715': r"Y:\BerlinLeipzigerStr\20240715\NetCDF",
        '20241106': r"Y:\BerlinLeipzigerStr\20241106\NetCDF",
        '20241111': r"Y:\BerlinLeipzigerStr\20241111\NetCDF",
        '20241122': r"Y:\BerlinLeipzigerStr\20241122\NetCDF",
    }
    # TEMPORARY (other machine): only the NetCDF file of 15.07. is reachable
    #scenarios['20240715'] = r"G:\Meine Ablage\enviprojects\20240715_001.nc"

    target_coords = (162, 125, 3)
    pollutants_to_plot = ['PM2.5', 'PM10', 'NO', 'NO2']

    pollutant_groups = {
        'Particulates': ['PM2.5', 'PM10'],
        'Nitrogen_Oxides': ['NO', 'NO2']
    }

    meas_all = load_measurements(csv_file)
    fox_all = load_fox_background(fox_file)
    traffic_profile = load_traffic_profile(traffic_file)

    all_simulation_data = []

    for sim_name, source in scenarios.items():
        print(f"\n================ Processing: {sim_name} ================")
        nc_files = find_nc_files(source)
        if not nc_files:
            print(f"Skipping - no NetCDF found: {source}")
            continue

        model_df = load_envimet_series(nc_files, sim_name, *target_coords, out_dirs['Cache'])
        if model_df.empty:
            continue

        # resample labels each hour at its start; with 60-min output it keeps the snapshots
        model_hourly = model_df.resample('1h').mean()
        window = evaluation_window(model_df, SPINUP_HOURS)
        common_idx = window.intersection(model_hourly.index).intersection(meas_all.index)

        if common_idx.empty:
            print(f"Error: No overlapping timestamps found for {sim_name}!")
            continue
        if len(common_idx) < len(window):
            print(f"Warning: {sim_name} has {len(common_idx)} of {len(window)} hours of the evaluation window")
        print(f"    evaluated {common_idx.min()} - {common_idx.max()} ({len(common_idx)} outputs)")

        sim = {
            'name': sim_name,
            't0': window[0],  # 00:00 of the evaluated day, hour 0 of the diurnal plots
            'meas': meas_all.loc[common_idx],
            'model': model_hourly.loc[common_idx],
            # 10-min background for the diurnal line, and its value at each model output for the statistics
            'fox': fox_all.loc[common_idx.min():common_idx.max()],
            'bg': fox_all.reindex(common_idx, method='ffill'),
            'traffic': pd.DataFrame({'Traffic': traffic_profile.reindex(common_idx.hour).values}, index=common_idx),
        }
        all_simulation_data.append(sim)
        plot_final_results(sim, pollutants_to_plot, out_dirs, smooth_lines=SMOOTH_PLOTS)

    if all_simulation_data:
        plot_aggregated_regression(all_simulation_data, pollutant_groups, out_dirs['Aggregated'])
        plot_aggregated_diurnal(all_simulation_data, list(scenarios), pollutant_groups, out_dirs['Aggregated'], smooth_lines=SMOOTH_PLOTS)
        export_aggregated_statistics(all_simulation_data, pollutants_to_plot, out_dirs['Stats'])
        print("\nAll Processing Complete!")
    else:
        print("\nNo data processed to aggregate.")
