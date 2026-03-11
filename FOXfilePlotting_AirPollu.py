import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import json
import numpy as np

# --------------------------
# Config
# --------------------------
# Globale Stileinstellungen
sns.set_theme(style="whitegrid", context="paper", font_scale=1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.constrained_layout.use'] = True

# Datenkonfiguration
FILE_PATH = Path(r'D:\enviprojects\Berlin_Mehringdamm_Base\merge7_clean_2024_Jun_Nov_smthWind_realBG2.FOX')

# Zeiträume (Two 24h Days)
DAY1_START = "15.07.2018 00:00:00"
DAY1_END = "16.07.2018 00:00:00"

DAY2_START = "06.11.2018 00:00:00"
DAY2_END = "07.11.2018 00:00:00"

# Plot Styling
NUM_Y_TICKS = 6
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 15
LEGEND_FONT_SIZE = 15
SHOW_GRID = True
GRID_STYLE = {'color': '#DDDDDD', 'linestyle': '--', 'linewidth': 0.2}

# Liniendicke für alle Linien
LINEWIDTH = 1.5

# Labels
AXIS_LABELS = {
    'y_sw': "Shortwave Radiation [W/m²]",
    'y_temp': "Air Temperature [°C]",
    'y_q': "Specific Humidity [g/kg]",
    'y_wind_speed': "Wind Speed [m/s]",
    'y_wind_direction': "Wind Direction [°]",
    'y_pollutants': "Background Concentration [ug/m³]",
    'title_sw_direct_diffuse': 'Direct and Diffuse Shortwave Radiation',
    'title_temp_humidity': 'Air Temperature and Specific Humidity',
    'title_wind': 'Wind Conditions'
}

# --------------------------
# Helper Functions
# --------------------------
def load_data(file_path):
    """Loads the FOX (JSON) file and extracts raw profiles."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading file: {file_path.name}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        timestep_list = data.get('timestepList', [])
        
        if not timestep_list:
            print("Error: 'timestepList' not found or empty in JSON.")
            return None

        processed_data = []
        for item in timestep_list:
            record = {}
            record['Date'] = item.get('date')
            record['Time'] = item.get('time')
            record['directrad'] = item.get('swDir', 0)
            record['diffuserad'] = item.get('swDif', 0)
            record['lw'] = item.get('lwRad', 0)
            
            t_prof = item.get('tProfile', [])
            if t_prof:
                record['at'] = t_prof[0].get('value') # Kelvin
                
            q_prof = item.get('qProfile', [])
            if q_prof:
                record['q'] = q_prof[0].get('value') # g/kg
                
            w_prof = item.get('windProfile', [])
            if w_prof:
                record['ws'] = w_prof[0].get('wSpdValue')
                record['wd'] = w_prof[0].get('wDirValue')
                
            # Extract background pollutants
            bg_poll = item.get('backgrPollutants', {})
            record['NO'] = bg_poll.get('NO', np.nan)
            record['NO2'] = bg_poll.get('NO2', np.nan)
            record['O3'] = bg_poll.get('O3', np.nan)
            record['PM10'] = bg_poll.get('PM10', np.nan)
            record['PM25'] = bg_poll.get('PM25', np.nan)
            
            processed_data.append(record)

        df = pd.DataFrame(processed_data)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        print(f"Successfully loaded {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def filter_data(df, start_datetime, end_datetime):
    if df is None or df.empty: return pd.DataFrame()
    return df[(df['DateTime'] >= start_datetime) & (df['DateTime'] <= end_datetime)]

def format_plot(ax, y_lim=None, num_yticks=None, y_label=None, x_lim=None, yticks=None):
    if y_lim: ax.set_ylim(y_lim)
    
    # Only format the X-axis for primary axes (where x_lim is provided)
    if x_lim: 
        ax.set_xlim(x_lim)
        
        # Explicitly generate ticks every 3 hours based on the limits
        ticks = pd.date_range(start=x_lim[0], end=x_lim[1], freq='3h')
        ax.set_xticks(ticks)
        
        # Generate the HH:MM labels, but set the first and last to an empty string
        labels = [t.strftime("%H:%M") for t in ticks]
        if len(labels) >= 2:
            labels[0] = ""
            labels[-1] = ""
        ax.set_xticklabels(labels)
        
        # Apply X-axis tick parameters and rotation
        ax.tick_params(axis='x', which='major', length=5, direction='in', labelsize=TICK_LABEL_SIZE)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Y-axis configs (applied to both primary and secondary axes)
    ax.tick_params(axis='y', which='major', length=3, direction='in', labelsize=TICK_LABEL_SIZE)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    elif num_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(num_yticks))
        
    ax.grid(SHOW_GRID, **GRID_STYLE)
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, fontweight='bold')

# --------------------------
# Plotting Functions
# --------------------------

def plot_temperature_humidity(df, start_dt, end_dt, ax_temp):
    ax_humidity = ax_temp.twinx()
    
    if 'at' in df.columns:
        l1 = ax_temp.plot(df['DateTime'], df['at'] - 273.15, 
                     color='black', 
                     linestyle='-', 
                     linewidth=LINEWIDTH, 
                     label='Air Temperature')
    
    if 'q' in df.columns:
        l2 = ax_humidity.plot(df['DateTime'], df['q'], 
                         color='black', 
                         linestyle=':', 
                         linewidth=LINEWIDTH,
                         label='Specific Humidity')
        
    temp_ylim = [0, 35]
    humidity_ylim = [0, 15]
    
    temp_ticks = np.linspace(temp_ylim[0], temp_ylim[1], NUM_Y_TICKS)
    humidity_ticks = np.linspace(humidity_ylim[0], humidity_ylim[1], NUM_Y_TICKS)
    
    format_plot(ax_temp, y_lim=temp_ylim, yticks=temp_ticks, y_label=AXIS_LABELS['y_temp'], x_lim=[start_dt, end_dt])
    format_plot(ax_humidity, y_lim=humidity_ylim, yticks=humidity_ticks, y_label=AXIS_LABELS['y_q'])
    
    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax_temp.legend(lns, labs, loc='upper left', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass


def plot_sw_radiation(df, start_dt, end_dt, axdir):
    if 'directrad' in df.columns:
        l1 = axdir.plot(df['DateTime'], df['directrad'], 
                      color='black', 
                      linestyle='-', 
                      linewidth=LINEWIDTH, 
                      label='Direct')
    
    if 'diffuserad' in df.columns:
        l2 = axdir.plot(df['DateTime'], df['diffuserad'], 
                          color='black', 
                          linestyle=':', 
                          linewidth=LINEWIDTH,
                          label='Diffuse')

    sw_ylim = [0, 1000]
    sw_ticks = np.arange(0, 1001, 200)

    format_plot(axdir, y_lim=sw_ylim, yticks=sw_ticks, y_label=AXIS_LABELS['y_sw'], x_lim=[start_dt, end_dt])

    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        axdir.legend(lns, labs, loc='upper left', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass    


def plot_wind(df, start_dt, end_dt, ax_speed):
    ax_direction = ax_speed.twinx()
    
    if 'ws' in df.columns:
        l1 = ax_speed.plot(df['DateTime'], df['ws'], 
                      color='black', 
                      linestyle='-', 
                      linewidth=LINEWIDTH, 
                      label='Wind Speed')
    
    if 'wd' in df.columns:
        l2 = ax_direction.plot(df['DateTime'], df['wd'], 
                          color='black', 
                          linestyle=':', 
                          linewidth=LINEWIDTH,
                          label='Wind Direction')

    speed_ylim = [0, 4] 
    direction_ylim = [0, 360]
    
    speed_ticks = np.linspace(speed_ylim[0], speed_ylim[1], 5)
    direction_ticks = np.arange(0, 361, 90)

    format_plot(ax_speed, y_lim=speed_ylim, yticks=speed_ticks, y_label=AXIS_LABELS['y_wind_speed'], x_lim=[start_dt, end_dt])
    format_plot(ax_direction, y_lim=direction_ylim, yticks=direction_ticks, y_label=AXIS_LABELS['y_wind_direction'])
    
    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax_speed.legend(lns, labs, loc='upper left', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass

def plot_background_pollutants(df, start_dt, end_dt, ax):
    lines = []
    labels = []
    
    colors = {'NO': 'black', 'NO2': 'black', 'O3': 'black', 'PM10': '#d62728', 'PM25': '#d62728'} 
    linestyles = {'NO': '-', 'NO2': '--', 'O3': ':', 'PM10': '-', 'PM25': ':'}
    
    for pol in ['NO', 'NO2', 'O3', 'PM10', 'PM25']:
        if pol in df.columns:
            l = ax.plot(df['DateTime'], df[pol], 
                        color=colors[pol], 
                        linestyle=linestyles[pol], 
                        linewidth=LINEWIDTH, 
                        label=pol)
            lines.extend(l)
            labels.append(pol)
            
    ax.set_ylim(bottom=0)
    format_plot(ax, y_label=AXIS_LABELS['y_pollutants'], x_lim=[start_dt, end_dt])
    
    try:
        ax.legend(lines, labels, loc='upper left', frameon=False, prop={'size': LEGEND_FONT_SIZE}, ncol=2)
    except:
        pass

def main():
    df = load_data(FILE_PATH)
    
    if df is not None:
        start_dt_1 = pd.to_datetime(DAY1_START, format="%d.%m.%Y %H:%M:%S")
        end_dt_1 = pd.to_datetime(DAY1_END, format="%d.%m.%Y %H:%M:%S")
        df_filtered_1 = filter_data(df, start_dt_1, end_dt_1)
        
        start_dt_2 = pd.to_datetime(DAY2_START, format="%d.%m.%Y %H:%M:%S")
        end_dt_2 = pd.to_datetime(DAY2_END, format="%d.%m.%Y %H:%M:%S")
        df_filtered_2 = filter_data(df, start_dt_2, end_dt_2)
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)

        # Plot Left Column: Day 1 (July)
        if not df_filtered_1.empty:
            axes[0, 0].set_title("2024-07-15", fontsize=14, fontweight='bold')
            plot_temperature_humidity(df_filtered_1, start_dt_1, end_dt_1, axes[0, 0])            
            plot_sw_radiation(df_filtered_1, start_dt_1, end_dt_1, axes[1, 0])
            plot_wind(df_filtered_1, start_dt_1, end_dt_1, axes[2, 0])
            plot_background_pollutants(df_filtered_1, start_dt_1, end_dt_1, axes[3, 0])
        else:
            print(f"No data found in the time range {DAY1_START} to {DAY1_END}")
            
        # Plot Right Column: Day 2 (November)
        if not df_filtered_2.empty:
            axes[0, 1].set_title("2024-11-06", fontsize=14, fontweight='bold')
            plot_temperature_humidity(df_filtered_2, start_dt_2, end_dt_2, axes[0, 1])            
            plot_sw_radiation(df_filtered_2, start_dt_2, end_dt_2, axes[1, 1])
            plot_wind(df_filtered_2, start_dt_2, end_dt_2, axes[2, 1])
            plot_background_pollutants(df_filtered_2, start_dt_2, end_dt_2, axes[3, 1])
        else:
            print(f"No data found in the time range {DAY2_START} to {DAY2_END}")
            
        out_path = FILE_PATH.parent / "combined_meteorology_2days.svg"
        plt.savefig(out_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"Combined plot created successfully at {out_path}.")

if __name__ == "__main__":
    main()