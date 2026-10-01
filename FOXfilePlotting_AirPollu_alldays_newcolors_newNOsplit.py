import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import json
import numpy as np
from scipy.interpolate import make_interp_spline
import cmcrameri.cm as cmc  # Official Crameri color library

# --------------------------
# Config
# --------------------------
# Global Style Settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.constrained_layout.use'] = False # Handled manually for wspace control

# Data Paths
FILE_PATH = Path(r'D:\enviprojects\Berlin_Feinstaub_Friedrichstr\new.fox')
CSV_MEASURED_PATH = Path(r"D:\enviprojects\Berlin_Feinstaub_LineSource\Berlin_Feinstaub_Messdaten.csv")

# Dynamic Time Periods split by Season - Add or remove days here!
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

# Plot Styling
NUM_Y_TICKS = 6
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 14
SHOW_GRID = True
GRID_STYLE = {'color': '#DDDDDD', 'linestyle': '--', 'linewidth': 0.4}
LINEWIDTH = 2.0

# Labels
AXIS_LABELS = {
    'y_sw': "Shortwave Radiation\n[W/m²]",
    'y_temp': "Air Temperature\n[°C]",
    'y_q': "Specific Humidity\n[g/kg]",
    'y_wind_speed': "Wind Speed\n[m/s]",
    'y_wind_direction': "Wind Direction\n[°]",
    'y_pollutants': "Background Concentration\n[µg/m³]"
}

# --------------------------
# Helper Functions
# --------------------------
def smooth_data(dates, values, num_points=300, clip_zero=True):
    """Applies cubic spline interpolation to smooth time series data."""
    values = pd.to_numeric(values, errors='coerce')
    
    temp_df = pd.DataFrame({'dates': dates, 'values': values}).dropna()
    
    if len(temp_df) < 4:
        return dates, values
        
    temp_df = temp_df.sort_values('dates').drop_duplicates(subset='dates')
    
    if len(temp_df) < 4:
        return dates, values
        
    dates_num = mdates.date2num(temp_df['dates'])
    values_clean = temp_df['values'].values
    
    x_smooth = np.linspace(dates_num.min(), dates_num.max(), num_points)
    spline = make_interp_spline(dates_num, values_clean, k=3)
    y_smooth = spline(x_smooth)
    
    if clip_zero:
        y_smooth = np.maximum(y_smooth, 0)
        
    return mdates.num2date(x_smooth), y_smooth

def load_data(file_path):
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading file: {file_path.name}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        timestep_list = data.get('timestepList', [])
        if not timestep_list: return None

        processed_data = []
        for item in timestep_list:
            record = {'Date': item.get('date'), 'Time': item.get('time'),
                      'directrad': item.get('swDir', 0), 'diffuserad': item.get('swDif', 0),
                      'lw': item.get('lwRad', 0)}
            
            t_prof = item.get('tProfile', [])
            if t_prof: record['at'] = t_prof[0].get('value')
                
            q_prof = item.get('qProfile', [])
            if q_prof: record['q'] = q_prof[0].get('value')
                
            w_prof = item.get('windProfile', [])
            if w_prof:
                record['ws'] = w_prof[0].get('wSpdValue')
                record['wd'] = w_prof[0].get('wDirValue')
                
            bg_poll = item.get('backgrPollutants', {})
            record['NO'] = bg_poll.get('NO', np.nan)
            record['NO2'] = bg_poll.get('NO2', np.nan)
            record['PM10'] = bg_poll.get('PM10', np.nan)
            record['PM25'] = bg_poll.get('PM25', np.nan)
            
            processed_data.append(record)

        df = pd.DataFrame(processed_data)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_measured_data(csv_path):
    if not csv_path or not csv_path.exists(): return None
    try:
        df = pd.read_csv(csv_path, sep=';')
        df['DateTime'] = pd.to_datetime(df['Zeit'], format='%d.%m.%Y %H:%M')
        
        df['DateTime'] = df['DateTime'].dt.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')\
                                       .dt.tz_convert('Etc/GMT-1')\
                                       .dt.tz_localize(None)
        
        df = df.dropna(subset=['DateTime'])
        
        for col in ['PM10', 'PM2_5', 'NO2', 'NO']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading measured data: {e}")
        return None

def filter_data(df, start_datetime, end_datetime):
    if df is None or df.empty: return pd.DataFrame()
    return df[(df['DateTime'] >= start_datetime) & (df['DateTime'] <= end_datetime)]

def format_plot(ax, y_lim=None, num_yticks=None, y_label=None, x_lim=None, yticks=None, show_ylabel=True):
    if y_lim: ax.set_ylim(y_lim)
    
    if x_lim: 
        ax.set_xlim(x_lim)
        ticks = pd.date_range(start=x_lim[0], end=x_lim[1], freq='6h') # 6-hour intervals for cleaner look
        ax.set_xticks(ticks)
        
        labels = [t.strftime("%H:%M") for t in ticks]
        if len(labels) >= 2:
            labels[0] = ""
            labels[-1] = ""
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', length=5, direction='in', labelsize=TICK_LABEL_SIZE)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    ax.tick_params(axis='y', which='major', length=0, direction='in', labelsize=TICK_LABEL_SIZE)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    elif num_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(num_yticks))
        
    ax.grid(SHOW_GRID, **GRID_STYLE)
    
    if y_label and show_ylabel:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=10)

# --------------------------
# Plotting Functions
# --------------------------

def plot_temperature_humidity(df, start_dt, end_dt, ax_temp, show_ylabel, show_twin, season_name):
    lines, labels = [], []
    ax_humidity = ax_temp.twinx()
    
    if 'at' in df.columns:
        x_s, y_s = smooth_data(df['DateTime'], df['at'] - 273.15, clip_zero=False)
        l1 = ax_temp.plot(x_s, y_s, color=cmc.roma(0.1), linestyle='-', linewidth=LINEWIDTH, label='Air Temperature')
        lines.extend(l1)
        labels.append('Air Temperature')
    if 'q' in df.columns:
        x_s, y_s = smooth_data(df['DateTime'], df['q'])
        l2 = ax_humidity.plot(x_s, y_s, color=cmc.roma(0.9), linestyle='-', linewidth=LINEWIDTH, label='Specific Humidity')
        lines.extend(l2)
        labels.append('Specific Humidity')
        
    # Dynamic Limits based on Season
    if season_name == "Autumn":
        temp_ylim, humidity_ylim = [0, 15], [0, 7.5]
    else: # Summer
        temp_ylim, humidity_ylim = [15, 35], [0, 15]

    temp_ticks = np.linspace(temp_ylim[0], temp_ylim[1], NUM_Y_TICKS)
    humidity_ticks = np.linspace(humidity_ylim[0], humidity_ylim[1], NUM_Y_TICKS)
    
    format_plot(ax_temp, y_lim=temp_ylim, yticks=temp_ticks, y_label=AXIS_LABELS['y_temp'], x_lim=[start_dt, end_dt], show_ylabel=show_ylabel)
    format_plot(ax_humidity, y_lim=humidity_ylim, yticks=humidity_ticks, y_label=AXIS_LABELS['y_q'], show_ylabel=show_twin)
    
    if not show_twin:
        ax_humidity.set_yticklabels([])
        ax_humidity.tick_params(axis='y', length=0)
        
    return lines, labels

def plot_sw_radiation(df, start_dt, end_dt, axdir, show_ylabel, season_name):
    lines, labels = [], []
    if 'directrad' in df.columns:
        x_s, y_s = smooth_data(df['DateTime'], df['directrad'])
        l1 = axdir.plot(x_s, y_s, color=cmc.lajolla(0.8), linestyle='-', linewidth=LINEWIDTH, label='Direct')
        lines.extend(l1)
        labels.append('Direct Radiation')
    if 'diffuserad' in df.columns:
        x_s, y_s = smooth_data(df['DateTime'], df['diffuserad'])
        l2 = axdir.plot(x_s, y_s, color=cmc.lajolla(0.6), linestyle='-', linewidth=LINEWIDTH, label='Diffuse')
        lines.extend(l2)
        labels.append('Diffuse Radiation')

    # Dynamic Limits based on Season
    if season_name == "Autumn":
        sw_ylim = [0, 500]
        sw_ticks = np.arange(0, 501, 100)
    else:
        sw_ylim = [0, 1000]
        sw_ticks = np.arange(0, 1001, 200)

    format_plot(axdir, y_lim=sw_ylim, yticks=sw_ticks, y_label=AXIS_LABELS['y_sw'], x_lim=[start_dt, end_dt], show_ylabel=show_ylabel)

    return lines, labels

def plot_wind(df, start_dt, end_dt, ax_speed, show_ylabel, show_twin, season_name):
    lines, labels = [], []
    ax_direction = ax_speed.twinx()
    
    if 'ws' in df.columns:
        x_s, y_s = smooth_data(df['DateTime'], df['ws'])
        l1 = ax_speed.plot(x_s, y_s, color=cmc.broc(0.1), linestyle='-', linewidth=LINEWIDTH, label='Wind Speed')
        lines.extend(l1)
        labels.append('Wind Speed')
    if 'wd' in df.columns:
        x_s, y_s = smooth_data(df['DateTime'], df['wd'], clip_zero=False)
        y_s = np.clip(y_s, 0, 360) # Ensure smoothed wind direction stays within physical bounds
        l2 = ax_direction.plot(x_s, y_s, color=cmc.broc(0.9), linestyle='-', linewidth=LINEWIDTH, label='Wind Direction')
        lines.extend(l2)
        labels.append('Wind Direction')

    # Dynamic limits based on Season
    if season_name == "Autumn":
        speed_ylim = [0, 2.0]
    else:
        speed_ylim = [0, 1.2]
        
    direction_ylim = [0, 360]
    speed_ticks = np.linspace(speed_ylim[0], speed_ylim[1], 5)
    direction_ticks = np.arange(0, 361, 90)

    format_plot(ax_speed, y_lim=speed_ylim, yticks=speed_ticks, y_label=AXIS_LABELS['y_wind_speed'], x_lim=[start_dt, end_dt], show_ylabel=show_ylabel)
    format_plot(ax_direction, y_lim=direction_ylim, yticks=direction_ticks, y_label=AXIS_LABELS['y_wind_direction'], show_ylabel=show_twin)
    
    if not show_twin:
        ax_direction.set_yticklabels([])
        ax_direction.tick_params(axis='y', length=0)
        
    return lines, labels

def plot_combined_pollutants(df, start_dt, end_dt, ax, target_list, df_measured, show_ylabel, season_name):
    lines, labels = [], []
    
    colors = {
        'NO': cmc.batlow(0.1),   
        'NO2': cmc.batlow(0.7),  
        'PM10': cmc.hawaii(0.8),  
        'PM25': cmc.hawaii(0.2)   
    } 
    styles = {
        'NO': '-', 
        'NO2': '-', 
        'PM10': '-', 
        'PM25': '-'
    }
    
    csv_map = {'NO': 'NO', 'NO2': 'NO2', 'PM10': 'PM10', 'PM25': 'PM2_5'}
    
    for pol in target_list:
        if pol in df.columns:
            x_s, y_s = smooth_data(df['DateTime'], df[pol])
            l = ax.plot(x_s, y_s, color=colors[pol], linestyle=styles[pol], linewidth=LINEWIDTH, label=f"{pol} (BG)")
            lines.extend(l)
            labels.append(f"{pol}")
            
    if df_measured is not None and not df_measured.empty:
        for pol in target_list:
            csv_col = csv_map.get(pol)
            if csv_col and csv_col in df_measured.columns:
                x_s, y_s = smooth_data(df_measured['DateTime'], df_measured[csv_col])
                l_m = ax.plot(x_s, y_s, color=colors[pol], linestyle=styles[pol], linewidth=LINEWIDTH+1, alpha=0.35, label=f"{pol} (Meas)")
                lines.extend(l_m)
                labels.append(f"{pol} (Meas)")
                
    # Both sets are bound 0 to 40 as requested
    if season_name == "Autumn":
        poll_ylim = [0, 40]
    else:
        poll_ylim = [0, 40] 
        
    ax.set_ylim(poll_ylim) 
    format_plot(ax, y_lim=poll_ylim, y_label=AXIS_LABELS['y_pollutants'], x_lim=[start_dt, end_dt], show_ylabel=show_ylabel)
    
    return lines, labels

def main():
    df = load_data(FILE_PATH)
    df_meas_all = load_measured_data(CSV_MEASURED_PATH)
    
    if df is not None:
        # Loop through each season and generate a separate figure
        for season_name, days_list in SEASONS.items():
            print(f"\n--- Processing Season: {season_name} ---")
            num_days = len(days_list)
            
            if num_days == 0:
                continue

            fig, axes = plt.subplots(4, num_days, figsize=(4.5 * num_days, 14), sharey='row')
            
            # Adjusted bottom spacing to make room for the combined legend
            plt.subplots_adjust(wspace=0.04, hspace=0.13, left=0.05, right=0.95, top=0.95, bottom=0.085)

            all_lines = []
            all_labels = []

            for col, (start, end) in enumerate(days_list):
                s_dt = pd.to_datetime(start, format="%d.%m.%Y %H:%M:%S")
                e_dt = pd.to_datetime(end, format="%d.%m.%Y %H:%M:%S")
                
                df_sub = filter_data(df, s_dt, e_dt)
                df_meas_sub = filter_data(df_meas_all, s_dt, e_dt) if df_meas_all is not None else None
                
                date_title = s_dt.strftime("%d.%m.%Y")
                
                if not df_sub.empty:
                    ax_col = [axes[row, col] if num_days > 1 else axes[row] for row in range(4)]
                    ax_col[0].set_title(date_title, fontsize=16, fontweight='bold', pad=15)
                    
                    show_ylabel = (col == 0) 
                    show_twin = (col == num_days - 1) 
                    
                    # Row 0: Temperature & Humidity
                    l1, lab1 = plot_temperature_humidity(df_sub, s_dt, e_dt, ax_col[0], show_ylabel, show_twin, season_name)            
                    
                    # Row 1: Shortwave Radiation
                    l2, lab2 = plot_sw_radiation(df_sub, s_dt, e_dt, ax_col[1], show_ylabel, season_name)
                    
                    # Row 2: Wind 
                    l3, lab3 = plot_wind(df_sub, s_dt, e_dt, ax_col[2], show_ylabel, show_twin, season_name)
                    
                    # Row 3: Combined Pollutants
                    l4, lab4 = plot_combined_pollutants(df_sub, s_dt, e_dt, ax_col[3], ['NO', 'NO2', 'PM10', 'PM25'], df_meas_sub, show_ylabel, season_name)
                    
                    # Collect legend components strictly from the first column to avoid duplicates
                    if col == 0:
                        all_lines.extend(l1 + l2 + l3 + l4)
                        all_labels.extend(lab1 + lab2 + lab3 + lab4)
                else:
                    print(f"No data found for {date_title}")

            # Plot combined legend underneath the windows
            if all_lines:
                fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
                           ncol=5, frameon=True, prop={'size': LEGEND_FONT_SIZE})

            out_path = FILE_PATH.parent / f"FOX_Meteo_Aggregated_{season_name}.svg"
            plt.savefig(out_path, format='svg', bbox_inches='tight')
            plt.savefig(out_path.with_suffix('.png'), format='png', bbox_inches='tight')
            plt.close()
            print(f"Aggregated plot created successfully: {out_path.stem}")

if __name__ == "__main__":
    main()