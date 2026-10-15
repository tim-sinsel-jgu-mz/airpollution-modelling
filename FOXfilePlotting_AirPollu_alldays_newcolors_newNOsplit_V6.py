"""Boundary conditions of the FOX forcing file per season: meteorology and background concentrations.

One figure per season, one column per day and one row per quantity group:
air temperature and specific humidity, shortwave radiation, wind, background PM,
background NO and NO2, background ozone. One legend below the panels holds one column per row.

Usage: python FOXfilePlotting_AirPollu_alldays_newcolors_newNOsplit_V7.py [fox_file] [output_folder]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator
import cmcrameri.cm as cmc

# --------------------------
# Config
# --------------------------
FILE_PATH = Path(r'D:\enviprojects\Berlin_Pollutants_LeipzigerStr\final.fox')
OUT_DIR = None  # None: next to the FOX file

# Days per season (model time) - add or remove days here
SEASONS = {
    "Summer": [
        ("26.06.2024 00:00:00", "27.06.2024 00:00:00"),
        ("08.07.2024 00:00:00", "09.07.2024 00:00:00"),
        ("15.07.2024 00:00:00", "16.07.2024 00:00:00")
    ],
    "Autumn": [
        ("06.11.2024 00:00:00", "07.11.2024 00:00:00"),
        ("11.11.2024 00:00:00", "12.11.2024 00:00:00"),
        ("22.11.2024 00:00:00", "23.11.2024 00:00:00")
    ]
}

# Rows with two y-axes keep fixed limits so that both axes share their grid lines
# (same number of intervals); all other rows scale to the data of the season.
TWIN_AXIS_LIMITS = {
    "Summer": {'at': (15, 35), 'q': (0, 16), 'ws': (0, 1.2)},
    "Autumn": {'at': (0, 8), 'q': (0, 8), 'ws': (0, 2.0)},
}
TWIN_AXIS_INTERVALS = 4

# Shape-preserving smoothing (PCHIP, no overshoot) of the line series; the FOX holds 10-min values
SMOOTH = False

# Plot styling
FONT_SIZE = 14
AXIS_LABEL_SIZE = 15
TITLE_SIZE = 16
LINEWIDTH = 2.0
PANEL_WIDTH, PANEL_HEIGHT = 4.2, 2.5  # inches per day column and per row
MARGINS = {'left': 1.35, 'right': 1.15, 'top': 0.55, 'bottom': 1.75}  # inches: axis labels, titles, legend
GRID_STYLE = {'color': '#DDDDDD', 'linestyle': '--', 'linewidth': 0.5}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': FONT_SIZE,
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#444444',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'savefig.dpi': 300,
})

# Colours: background concentrations use the background colours of the diurnal plots
COLORS = {
    'at': cmc.roma(0.1), 'q': cmc.roma(0.9),
    'directrad': cmc.lajolla(0.45), 'diffuserad': cmc.lajolla(0.7),
    'ws': cmc.broc(0.1), 'wd': cmc.broc(0.9),
    'PM25': cmc.hawaii(0.2), 'PM10': cmc.hawaii(0.7),
    'NO': cmc.batlow(0.2), 'NO2': cmc.batlow(0.7),
    'O3': cmc.roma(0.75),
}

# Rows: left-axis series, right-axis series, axis labels. Series are (column, legend label).
ROWS = [
    {'left': [('at', 'Air temperature')], 'right': [('q', 'Specific humidity')],
     'ylabel': "Air Temperature\n[°C]", 'ylabel_right': "Specific Humidity\n[g/kg]"},
    {'left': [('directrad', 'Direct radiation'), ('diffuserad', 'Diffuse radiation')],
     'ylabel': "Shortwave Radiation\n[W/m²]"},
    {'left': [('ws', 'Wind speed')], 'right': [('wd', 'Wind direction')],
     'ylabel': "Wind Speed\n[m/s]", 'ylabel_right': "Wind Direction"},
    {'left': [('PM25', 'PM$_{2.5}$'), ('PM10', 'PM$_{10}$')],
     'ylabel': "Background PM\n[µg/m³]"},
    {'left': [('NO', 'NO'), ('NO2', 'NO$_2$')],
     'ylabel': "Background NO$_x$\n[µg/m³]"},
    {'left': [('O3', 'O$_3$')],
     'ylabel': "Background O$_3$\n[µg/m³]"},
]


# --------------------------
# Data
# --------------------------
def load_data(file_path):
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading file: {file_path.name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        timestep_list = json.load(f).get('timestepList', [])
    if not timestep_list:
        return None

    records = []
    for item in timestep_list:
        record = {'Date': item.get('date'), 'Time': item.get('time'),
                  'directrad': item.get('swDir', 0), 'diffuserad': item.get('swDif', 0)}
        t_prof = item.get('tProfile', [])
        if t_prof:
            record['at'] = t_prof[0].get('value') - 273.15
        q_prof = item.get('qProfile', [])
        if q_prof:
            record['q'] = q_prof[0].get('value')
        w_prof = item.get('windProfile', [])
        if w_prof:
            record['ws'] = w_prof[0].get('wSpdValue')
            record['wd'] = w_prof[0].get('wDirValue')
        bg = item.get('backgrPollutants', {})
        for key in ('NO', 'NO2', 'PM10', 'PM25', 'O3'):
            record[key] = bg.get(key, np.nan)
        records.append(record)

    df = pd.DataFrame(records)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['DateTime'])
    for col in ('NO', 'NO2', 'PM10', 'PM25', 'O3'):
        df[col] = df[col].where(df[col] > -998)  # FOX no-data value -999
    return df


def filter_data(df, start_dt, end_dt):
    return df[(df['DateTime'] >= start_dt) & (df['DateTime'] <= end_dt)]


def hours(df, start_dt):
    return ((df['DateTime'] - start_dt) / pd.Timedelta(hours=1)).to_numpy()


def line_data(x, y):
    """The series as plotted: raw, or PCHIP-smoothed when SMOOTH is set."""
    mask = np.isfinite(y)
    x, y = x[mask], y[mask]
    if not SMOOTH or len(x) < 3:
        return x, y
    x_s = np.linspace(x.min(), x.max(), 600)
    return x_s, PchipInterpolator(x, y)(x_s)


def nice_limit(vmax, intervals=4):
    """Upper axis limit on a 1-2-2.5-5 grid with roughly the given number of intervals above zero."""
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    raw = vmax * 1.05 / intervals
    mag = 10 ** np.floor(np.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    return step * np.ceil(vmax * 1.05 / step)


# --------------------------
# Plotting
# --------------------------
def plot_series(ax, x, y, column):
    """Wind direction as dots (a line would jump across the 0/360 wrap), everything else as a line."""
    if column == 'wd':
        mask = np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=7, color=COLORS[column], linewidths=0, zorder=3)
    else:
        ax.plot(*line_data(x, y), color=COLORS[column], linewidth=LINEWIDTH, zorder=3)


def legend_handle(column, label):
    if column == 'wd':
        return Line2D([], [], color=COLORS[column], marker='o', linestyle='', markersize=5, label=label)
    return Line2D([], [], color=COLORS[column], linewidth=LINEWIDTH, label=label)


def format_x(ax, is_bottom):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['', '06:00', '12:00', '18:00', ''])
    ax.tick_params(axis='x', labelbottom=is_bottom, labelsize=FONT_SIZE)
    ax.grid(True, **GRID_STYLE)


def row_limits(season_name, days, df):
    """Left-axis limits per row; rows without a twin axis scale to the maximum of the season."""
    fixed = TWIN_AXIS_LIMITS[season_name]
    season_df = pd.concat([filter_data(df, s, e) for s, e in days])
    limits = []
    for row in ROWS:
        first = row['left'][0][0]
        if 'right' in row:
            limits.append(fixed[first])
        else:
            vmax = np.nanmax([season_df[col].max() for col, _ in row['left'] if col in season_df])
            limits.append((0, nice_limit(vmax)))
    return limits


def plot_season(df, season_name, days, out_dir):
    n_days = len(days)
    n_rows = len(ROWS)
    fig_w = PANEL_WIDTH * n_days + MARGINS['left'] + MARGINS['right']
    fig_h = PANEL_HEIGHT * n_rows + MARGINS['top'] + MARGINS['bottom']
    fig, axes = plt.subplots(n_rows, n_days, sharex=True, sharey='row', squeeze=False, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=MARGINS['left'] / fig_w, right=1 - MARGINS['right'] / fig_w,
                        top=1 - MARGINS['top'] / fig_h, bottom=MARGINS['bottom'] / fig_h, wspace=0.05, hspace=0.12)

    periods = [(pd.to_datetime(s, format="%d.%m.%Y %H:%M:%S"), pd.to_datetime(e, format="%d.%m.%Y %H:%M:%S"))
               for s, e in days]
    limits = row_limits(season_name, periods, df)
    fixed = TWIN_AXIS_LIMITS[season_name]

    for c, (s_dt, e_dt) in enumerate(periods):
        sub = filter_data(df, s_dt, e_dt)
        if sub.empty:
            print(f"No data found for {s_dt:%d.%m.%Y}")
            continue
        x = hours(sub, s_dt)
        axes[0, c].set_title(s_dt.strftime("%d.%m.%Y"), fontsize=TITLE_SIZE, fontweight='bold', pad=10)

        for r, row in enumerate(ROWS):
            ax = axes[r, c]
            for col, _ in row['left']:
                if col in sub:
                    plot_series(ax, x, sub[col].to_numpy(dtype=float), col)
            ax.set_ylim(limits[r])
            if 'right' in row:
                ax.set_yticks(np.linspace(*limits[r], TWIN_AXIS_INTERVALS + 1))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(axis='y', labelsize=FONT_SIZE)
            format_x(ax, r == n_rows - 1)
            if c == 0:
                ax.set_ylabel(row['ylabel'], fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=8)

            if 'right' in row:
                twin = ax.twinx()
                for col, _ in row['right']:
                    if col in sub:
                        plot_series(twin, x, sub[col].to_numpy(dtype=float), col)
                if row['right'][0][0] == 'wd':
                    twin.set_ylim(0, 360)
                    twin.set_yticks([0, 90, 180, 270, 360])
                    twin.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
                else:
                    lim = fixed[row['right'][0][0]]
                    twin.set_ylim(lim)
                    twin.set_yticks(np.linspace(*lim, TWIN_AXIS_INTERVALS + 1))
                twin.tick_params(axis='y', labelsize=FONT_SIZE, labelright=(c == n_days - 1))
                if c == n_days - 1:
                    twin.set_ylabel(row['ylabel_right'], fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=8)

    for c in range(n_days):
        axes[-1, c].set_xlabel("Time (UTC+1)", fontsize=AXIS_LABEL_SIZE)

    # One legend below the panels: column k lists the series of row k (legends fill column by column)
    groups = [[legend_handle(col, label) for col, label in row['left'] + row.get('right', [])] for row in ROWS]
    depth = max(len(g) for g in groups)
    blank = Line2D([], [], linestyle='', label=' ')
    handles = [h for g in groups for h in g + [blank] * (depth - len(g))]
    left, right = axes[-1, 0].get_position().x0, axes[-1, -1].get_position().x1
    fig.legend(handles=handles, loc='upper center', ncol=len(groups), frameon=False, fontsize=FONT_SIZE,
               handlelength=1.6, columnspacing=1.4, borderaxespad=0,
               bbox_to_anchor=((left + right) / 2, (MARGINS['bottom'] - 0.85) / fig_h))

    out_path = Path(out_dir) / f"FOX_Meteo_Aggregated_{season_name}.svg"
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Aggregated plot created successfully: {out_path.with_suffix('.png')}")


def main(fox_path=FILE_PATH, out_dir=OUT_DIR):
    df = load_data(Path(fox_path))
    if df is None:
        return
    out_dir = Path(out_dir) if out_dir else Path(fox_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for season_name, days in SEASONS.items():
        if days:
            print(f"\n--- Processing Season: {season_name} ---")
            plot_season(df, season_name, days, out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:3])
