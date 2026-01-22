import pandas as pd
import os

# ==========================================
#  USER SETTINGS - EDIT THIS SECTION
# ==========================================

INPUT_FILE_NAME = 'merge5.csv'
OUTPUT_FILE_NAME = 'merge5__clean_2019_Jun_Nov.csv'

DATETIME_COLUMN = 'Merged_Timestamp'
REMOVE_ORIGINAL_DATETIME_COLUMN = True

# ------------------------------------------
# Date range clipping
# ------------------------------------------

ENABLE_DATE_RANGE_CLIP = True
CLIP_START_DATETIME = '201906010000'
CLIP_END_DATETIME   = '201911302350'


# ------------------------------------------
# Time continuity check
# ------------------------------------------

ENABLE_TIME_CONTINUITY_CHECK = True
EXPECTED_TIME_STEP_MINUTES = 10
STOP_ON_TIME_DISCONTINUITY = True

# ------------------------------------------
# Time zone conversion
# ------------------------------------------

ENABLE_TIMEZONE_SHIFT = True

# Input and target time zones
# Examples: "UTC", "UTC+1", "UTC-2", "UTC+10"
INPUT_TIMEZONE = "UTC"
TARGET_TIMEZONE = "UTC+1"



# ------------------------------------------
# Air temperature unit conversion
# ------------------------------------------

ENABLE_TEMPERATURE_CONVERSION = True

# Options:
# "K_TO_C" → Kelvin to Celsius
# "C_TO_K" → Celsius to Kelvin
TEMPERATURE_CONVERSION_MODE = "C_TO_K"

# List of temperature columns to convert
TEMPERATURE_COLUMNS = [
    'TT_ST_10'
]


# ------------------------------------------
# Shortwave Direct calculation
# ------------------------------------------

ENABLE_SHORTWAVE_DIRECT_CALC = True
SHORTWAVE_GLOBAL_COLUMN = 'GS_10'
SHORTWAVE_DIFFUSE_COLUMN = 'DS_10'
SHORTWAVE_DIRECT_OUTPUT_COLUMN = 'shortwave_dir'


# ------------------------------------------
# Invalid / error value handling
# ------------------------------------------

# === Mode 1: Handle ALL negative values (supersedes error-value replacement) ===
ENABLE_NEGATIVE_VALUE_HANDLING = False

# How to handle negative values:
# 'replace' = replace negatives with a fixed value
# 'drop'    = drop rows containing negatives
NEGATIVE_VALUE_MODE = 'replace'

# Replacement value if mode == 'replace'
NEGATIVE_VALUE_REPLACEMENT = 0

# Columns to check for negative values
# Use None to apply to ALL numeric columns
NEGATIVE_VALUE_COLUMNS = [
    'DS_10',
    'GS_10',
    'LS_10',
    'RF_ST_10',
    'FF_ST_10',
    'DD_ST_10',
    'shortwave_dir',
    'FF_10',
    'DD_10'
]


# === Mode 2: Replace specific error values (used only if above is disabled) ===
ENABLE_ERROR_VALUE_REPLACEMENT = True

ERROR_VALUES_TO_REPLACE = [-999, (-999+273.15)]
ERROR_REPLACEMENT_VALUE = 0

# Columns to apply error-value replacement
# Use None to apply to ALL numeric columns
ERROR_REPLACEMENT_COLUMNS = [
    'DS_10',
    'GS_10',
    'LS_10',
    'RF_ST_10',
    'FF_ST_10',
    'DD_ST_10',
    'shortwave_dir',
    'FF_10',
    'DD_10'
]


# ------------------------------------------
# Radiation unit conversion (J/cm² → W/m²)
# ------------------------------------------

ENABLE_RADIATION_UNIT_CONVERSION = True

# Length of one timestep in seconds (10 minutes = 600)
RADIATION_TIME_STEP_SECONDS = 600

# Radiation columns to convert
RADIATION_COLUMNS_J_CM2 = [
    'DS_10',
    'GS_10',
    'LS_10',
    'shortwave_dir'
]

# ------------------------------------------
# Negative radiation check
# ------------------------------------------

ENABLE_NEGATIVE_RADIATION_CHECK = True
RADIATION_COLUMNS_TO_CHECK = [
    'shortwave_dir',
    'DS_10',
    'LS_10'
]
STOP_ON_NEGATIVE_RADIATION = False


# ------------------------------------------
# Wind data sanitization (FOX compatibility)
# ------------------------------------------

ENABLE_WIND_DIRECTION_SANITIZATION = False
ENABLE_ZERO_WIND_SPEED_FIX = False
ZERO_WIND_REPLACEMENT_VALUE = 0.1
WINDSPEED_COLUMN_NAME = 'FF_10'     # alternativ FF_ST_10
WINDDIRECTION_COLUMN_NAME = 'DD_10'     #alternativ DD_ST_10

# ------------------------------------------
# Optional precipitation column
# ------------------------------------------

ENABLE_PRECIPITATION_COLUMN = True

# Name of the precipitation column to create
PRECIPITATION_COLUMN_NAME = 'Precipitation'

# Constant value to assign (e.g. 0 for no rain)
PRECIPITATION_DEFAULT_VALUE = 0.0


# ------------------------------------------
# FOX column reordering
# ------------------------------------------

ENABLE_FOX_REORDERING = True
KEEP_NON_FOX_COLUMNS = False

FOX_COLUMN_MAPPING = {
    "Shortwave_Direct": "shortwave_dir",
    "Shortwave_Diffuse": "DS_10",
    "Longwave": "LS_10",
    "Air_Temperature": "TT_ST_10",
    "Relative_Humidity": "RF_ST_10",
    "Windspeed": "FF_10",            #alternativ auch FF_ST_10
    "Wind_Direction": "DD_10",      #alternativ auch DD_ST_10
    "Precipitation": "Precipitation"
}


# ==========================================
#  END OF SETTINGS
# ==========================================


def split_datetime_column():
    print("--- Starting FOX CSV Preparation Script ---")

    if not os.path.exists(INPUT_FILE_NAME):
        print(f"Error: File '{INPUT_FILE_NAME}' not found.")
        return

    try:
        df = pd.read_csv(INPUT_FILE_NAME)

        if DATETIME_COLUMN not in df.columns:
            print(f"Error: Datetime column '{DATETIME_COLUMN}' not found.")
            return

        # --------------------------------------------------
        # Parse datetime
        # --------------------------------------------------
        dt = pd.to_datetime(df[DATETIME_COLUMN], format='%Y%m%d%H%M')
        df['_dt'] = dt
        MAX_ALLOWED_ROWS = 20000

        if len(df) > MAX_ALLOWED_ROWS:
            print(
                f"Warning: Dataset contains {len(df)} rows. "
                "FOX may reject very long forcing periods."
            )

        # --------------------------------------------------
        # Date range clipping
        # --------------------------------------------------
        if ENABLE_DATE_RANGE_CLIP:
            print("Clipping dataset to date range...")
            start = pd.to_datetime(CLIP_START_DATETIME, format='%Y%m%d%H%M')
            end   = pd.to_datetime(CLIP_END_DATETIME, format='%Y%m%d%H%M')

            before = len(df)
            df = df[(df['_dt'] >= start) & (df['_dt'] <= end)]
            after = len(df)

            print(f"Rows before: {before}, after clipping: {after}")

        # --------------------------------------------------
        # Time continuity check
        # --------------------------------------------------
        if ENABLE_TIME_CONTINUITY_CHECK:
            print("Checking time continuity...")
            diffs = df['_dt'].diff().dropna()
            expected = pd.Timedelta(minutes=EXPECTED_TIME_STEP_MINUTES)

            bad = diffs[diffs != expected]

            if not bad.empty:
                print("ERROR: Time discontinuity detected!")
                print(bad.value_counts())

                if STOP_ON_TIME_DISCONTINUITY:
                    return
            else:
                print("Time continuity OK.")

        # --------------------------------------------------
        # Insert Date and Time columns
        # --------------------------------------------------
        idx = df.columns.get_loc(DATETIME_COLUMN) + 1
        df.insert(idx, 'Date', df['_dt'].dt.strftime('%d.%m.%Y'))
        df.insert(idx + 1, 'Time', df['_dt'].dt.strftime('%H:%M:%S'))

        if REMOVE_ORIGINAL_DATETIME_COLUMN:
            df.drop(columns=[DATETIME_COLUMN], inplace=True)
        
        # --------------------------------------------------
        # Time zone shift (UTC offset based)
        # --------------------------------------------------
        if ENABLE_TIMEZONE_SHIFT:
            print(f"Shifting time zone from {INPUT_TIMEZONE} to {TARGET_TIMEZONE}...")

            def parse_utc_offset(tz_str):
                if tz_str == "UTC":
                    return 0
                if tz_str.startswith("UTC"):
                    return int(tz_str.replace("UTC", ""))
                raise ValueError(
                    "Invalid timezone format. Use 'UTC', 'UTC+1', 'UTC-2', etc."
                )

            input_offset = parse_utc_offset(INPUT_TIMEZONE)
            target_offset = parse_utc_offset(TARGET_TIMEZONE)

            hour_shift = target_offset - input_offset

            # Combine Date and Time into datetime
            dt = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format='%d.%m.%Y %H:%M:%S'
            )

            # Apply shift
            dt = dt + pd.to_timedelta(hour_shift, unit='h')

            # Split back into Date and Time
            df['Date'] = dt.dt.strftime('%d.%m.%Y')
            df['Time'] = dt.dt.strftime('%H:%M:%S')

        
        # --------------------------------------------------
        # Air temperature unit conversion
        # --------------------------------------------------
        if ENABLE_TEMPERATURE_CONVERSION:
            print(f"Converting air temperature ({TEMPERATURE_CONVERSION_MODE})...")

            for col in TEMPERATURE_COLUMNS:
                if col not in df.columns:
                    print(f"Warning: temperature column '{col}' not found, skipping.")
                    continue

                if TEMPERATURE_CONVERSION_MODE == "K_TO_C":
                    df[col] = df[col] - 273.15

                elif TEMPERATURE_CONVERSION_MODE == "C_TO_K":
                    df[col] = df[col] + 273.15

                else:
                    raise ValueError(
                        "Invalid TEMPERATURE_CONVERSION_MODE. "
                        "Use 'K_TO_C' or 'C_TO_K'."
                    )

        # --------------------------------------------------
        # Shortwave Direct calculation
        # --------------------------------------------------
        if ENABLE_SHORTWAVE_DIRECT_CALC:
            df[SHORTWAVE_DIRECT_OUTPUT_COLUMN] = (
                df[SHORTWAVE_GLOBAL_COLUMN] - df[SHORTWAVE_DIFFUSE_COLUMN]
            ).clip(lower=0)


        # --------------------------------------------------
        # Invalid / error value handling
        # --------------------------------------------------

        # === Mode 1: Negative value handling (highest priority) ===
        if ENABLE_NEGATIVE_VALUE_HANDLING:
            print("Handling negative values...")

            if NEGATIVE_VALUE_COLUMNS is None:
                target_columns = df.select_dtypes(include='number').columns
            else:
                target_columns = [c for c in NEGATIVE_VALUE_COLUMNS if c in df.columns]

            if NEGATIVE_VALUE_MODE == 'replace':
                total_replaced = 0

                for col in target_columns:
                    mask = df[col] < 0
                    count = mask.sum()
                    if count > 0:
                        df.loc[mask, col] = NEGATIVE_VALUE_REPLACEMENT
                        total_replaced += count
                        print(f"  {col}: replaced {count} negative values")

                if total_replaced == 0:
                    print("  No negative values found.")

            elif NEGATIVE_VALUE_MODE == 'drop':
                before = len(df)
                mask = (df[target_columns] < 0).any(axis=1)
                df = df[~mask]
                after = len(df)
                print(f"  Dropped {before - after} rows containing negative values")

            else:
                print(f"Error: Unknown NEGATIVE_VALUE_MODE '{NEGATIVE_VALUE_MODE}'")
                return


        # === Mode 2: Specific error value replacement (only if negatives not handled) ===
        elif ENABLE_ERROR_VALUE_REPLACEMENT:
            print("Replacing specific error values...")

            if ERROR_REPLACEMENT_COLUMNS is None:
                target_columns = df.select_dtypes(include='number').columns
            else:
                target_columns = [
                    c for c in ERROR_REPLACEMENT_COLUMNS if c in df.columns
                ]

            total_replaced = 0

            for col in target_columns:
                for err_val in ERROR_VALUES_TO_REPLACE:
                    mask = df[col] == err_val
                    count = mask.sum()
                    if count > 0:
                        df.loc[mask, col] = ERROR_REPLACEMENT_VALUE
                        total_replaced += count
                        print(
                            f"  {col}: replaced {count} occurrences of {err_val}"
                        )

            if total_replaced == 0:
                print("  No error values found.")

        # --------------------------------------------------
        # Radiation unit conversion (J/cm² → W/m²)
        # --------------------------------------------------
        if ENABLE_RADIATION_UNIT_CONVERSION:
            print("Converting radiation units from J/cm² to W/m²...")

            conversion_factor = 10000 / RADIATION_TIME_STEP_SECONDS

            for col in RADIATION_COLUMNS_J_CM2:
                if col not in df.columns:
                    print(f"Warning: radiation column '{col}' not found, skipping.")
                    continue

                df[col] = df[col] * conversion_factor

        # --------------------------------------------------
        # Negative radiation check
        # --------------------------------------------------
        if ENABLE_NEGATIVE_RADIATION_CHECK:
            print("Checking for negative radiation values...")
            found = False

            for col in RADIATION_COLUMNS_TO_CHECK:
                if col not in df.columns:
                    continue
                count = (df[col] < 0).sum()
                if count > 0:
                    print(f"Warning: {count} negative values in '{col}'")
                    found = True

            if found and STOP_ON_NEGATIVE_RADIATION:
                return

        # --------------------------------------------------
        # Wind direction sanitization (0–360°, FOX-safe)
        # --------------------------------------------------
        if ENABLE_WIND_DIRECTION_SANITIZATION:
            if WINDDIRECTION_COLUMN_NAME in df.columns:
                print("Sanitizing wind direction values...")
                df[WINDDIRECTION_COLUMN_NAME] = df[WINDDIRECTION_COLUMN_NAME] % 360

        # --------------------------------------------------
        # Replace zero wind speed values (FOX workaround)
        # --------------------------------------------------
        if ENABLE_ZERO_WIND_SPEED_FIX:
            if WINDDIRECTION_COLUMN_NAME in df.columns:
                zero_count = (df[WINDSPEED_COLUMN_NAME] == 0).sum()
                if zero_count > 0:
                    print(f"Replacing {zero_count} zero wind speed values with {ZERO_WIND_REPLACEMENT_VALUE}...")
                    df.loc[df[WINDSPEED_COLUMN_NAME] == 0, WINDSPEED_COLUMN_NAME] = ZERO_WIND_REPLACEMENT_VALUE

        # --------------------------------------------------
        # Optional precipitation column creation
        # --------------------------------------------------
        if ENABLE_PRECIPITATION_COLUMN:
            if PRECIPITATION_COLUMN_NAME in df.columns:
                print(f"Precipitation column '{PRECIPITATION_COLUMN_NAME}' already exists, leaving unchanged.")
            else:
                print(
                    f"Adding precipitation column '{PRECIPITATION_COLUMN_NAME}' "
                    f"with constant value {PRECIPITATION_DEFAULT_VALUE}..."
                )
                df[PRECIPITATION_COLUMN_NAME] = PRECIPITATION_DEFAULT_VALUE


        # --------------------------------------------------
        # FOX column reordering
        # --------------------------------------------------
        if ENABLE_FOX_REORDERING:
            fox_order = ['Date', 'Time']
            used = set(fox_order)

            for key in [
                "Shortwave_Direct",
                "Shortwave_Diffuse",
                "Longwave",
                "Air_Temperature",
                "Relative_Humidity",
                "Windspeed",
                "Wind_Direction",
                "Precipitation"
            ]:
                col = FOX_COLUMN_MAPPING.get(key)
                if col and col in df.columns:
                    fox_order.append(col)
                    used.add(col)

            if KEEP_NON_FOX_COLUMNS:
                fox_order.extend([c for c in df.columns if c not in used])

            df = df[fox_order]

        if '_dt' in df.columns:
            df.drop(columns=['_dt'], inplace=True)

        # Forcing Date and Time to pure, formatted strings
        print("Forcing FOX-safe Date and Time formatting...")

        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')


        df.to_csv(
            OUTPUT_FILE_NAME,
            index=False,
            sep=',',
            decimal='.',
            float_format='%.3f'
        )

        print("Success! FOX-ready CSV written.")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    split_datetime_column()
