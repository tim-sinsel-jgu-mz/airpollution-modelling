import pandas as pd
import os
import sys

# ==========================================
#  USER SETTINGS - EDIT THIS SECTION
# ==========================================

# 1. Input File Names (Must be in the same folder as this script)
FILE_1_NAME = 'merge7_T_RF_W.csv' 
FILE_2_NAME = 'general_10_427_solar_hist_bis202412.csv'

# 2. Output File Name
OUTPUT_FILE_NAME = 'merge7_T_RF_W_S.csv'

# 3. Date Time Column Headers (The name of the column containing YYYYMMDDHHMM)
#    Change these if they are different in your files.
TIME_COL_1 = 'MESS_DATUM'
TIME_COL_2 = 'MESS_DATUM'

# 4. Columns to Keep
#    List the specific columns you want from each file.
#    Leave the list empty [] if you want to keep ALL columns from that file. # WARNING: Do NOT leave empty, that doesn't work!
#    NOTE: Do not include the Time Column here; the script handles that automatically.
COLS_TO_KEEP_FILE_1 = ['TT_ST_10', 'RF_ST_10', 'FF_10', 'DD_10']   #'TT_ST_10', 'RF_ST_10', 'DS_10', 'LS_10', 'GS_10', 'FF_10', 'DD_10'
COLS_TO_KEEP_FILE_2 = ['DS_10', 'LS_10', 'GS_10']

# 5. CSV format settings
# This script outputs like this: SEPARATOR = ','; DECIMAL = '.'; ENCODING = 'utf-8'
# Keep in mind if you want to use script outputs for further merges.
FILE_1_SEPARATOR = ','
FILE_1_DECIMAL = '.'
FILE_1_ENCODING = 'utf-8'

FILE_2_SEPARATOR = ';'
FILE_2_DECIMAL = '.'
FILE_2_ENCODING = 'utf-8'

OUTPUT_SEPARATOR = ','
OUTPUT_DECIMAL = '.'
OUTPUT_ENCODING = 'utf-8'
OUTPUT_NA_REP = ''      

# ==========================================
#  END OF SETTINGS
# ==========================================

def merge_csvs():
    print("--- Starting CSV Merge Script ---")

    # 1. Check if files exist
    if not os.path.exists(FILE_1_NAME):
        print(f"Error: File '{FILE_1_NAME}' not found in current directory.")
        return
    if not os.path.exists(FILE_2_NAME):
        print(f"Error: File '{FILE_2_NAME}' not found in current directory.")
        return

    try:
        # 2. Load Data
        print(f"Reading {FILE_1_NAME}...")
        df1 = pd.read_csv(
            FILE_1_NAME,
            sep=FILE_1_SEPARATOR,
            decimal=FILE_1_DECIMAL,
            encoding=FILE_1_ENCODING,
            skipinitialspace=True
        )
        
        print(f"Reading {FILE_2_NAME}...")
        df2 = pd.read_csv(
            FILE_2_NAME,
            sep=FILE_2_SEPARATOR,
            decimal=FILE_2_DECIMAL,
            encoding=FILE_2_ENCODING,
            skipinitialspace=True
        )
        
        # 2.1 Clean column headers
        df1.columns = df1.columns.str.strip()
        df2.columns = df2.columns.str.strip()


        # 3. Validate Columns Exist
        if TIME_COL_1 not in df1.columns:
            print(f"Error: Timestamp column '{TIME_COL_1}' not found in {FILE_1_NAME}.")
            return
        if TIME_COL_2 not in df2.columns:
            print(f"Error: Timestamp column '{TIME_COL_2}' not found in {FILE_2_NAME}.")
            return

        # 4. Parse Dates (Format: YYYYMMDDHHMM)
        #    We convert them to datetime objects for accurate comparison
        print("Parsing timestamps...")
        try:
            df1['timestamp_dt'] = pd.to_datetime(df1[TIME_COL_1], format='%Y%m%d%H%M')
            df2['timestamp_dt'] = pd.to_datetime(df2[TIME_COL_2], format='%Y%m%d%H%M')
        except ValueError as e:
            print(f"Error: Could not parse timestamps. Ensure format is YYYYMMDDHHMM.\nDetails: {e}")
            return

        # 5. Check for Overlap
        #    We check if the ranges intersect at all.
        start1, end1 = df1['timestamp_dt'].min(), df1['timestamp_dt'].max()
        start2, end2 = df2['timestamp_dt'].min(), df2['timestamp_dt'].max()

        latest_start = max(start1, start2)
        earliest_end = min(end1, end2)

        if latest_start > earliest_end:
            print("\n!!! ERROR: NO DATE OVERLAP DETECTED !!!")
            print(f"File 1 Range: {start1} to {end1}")
            print(f"File 2 Range: {start2} to {end2}")
            print("Stopping script.")
            return
        else:
            print(f"Overlap detected between {latest_start} and {earliest_end}.")

        # 6. Filter Columns (if user requested specific columns)
        def filter_columns(df, keep_list, filename):
            if not keep_list:
                return df # Keep all if list is empty
            
            # Always ensure we keep the parsing column and the original time column
            # (We will clean up the parsing column later)
            cols_needed = keep_list + ['timestamp_dt']
            
            # Check if user requested columns that don't exist
            missing = [c for c in keep_list if c not in df.columns]
            if missing:
                print(f"Warning: Columns {missing} requested but not found in {filename}. Skipping them.")
                cols_needed = [c for c in cols_needed if c in df.columns]
                
            return df[cols_needed]

        df1 = filter_columns(df1, COLS_TO_KEEP_FILE_1, FILE_1_NAME)
        df2 = filter_columns(df2, COLS_TO_KEEP_FILE_2, FILE_2_NAME)

        # 7. Perform the Merge
        #    We set the timestamp as the Index to align them perfectly.
        #    We use 'outer' join to keep the union of data (partial overlaps),
        #    but because we checked for overlap earlier, we know they touch.
        df1.set_index('timestamp_dt', inplace=True)
        df2.set_index('timestamp_dt', inplace=True)

        print("Merging data...")
        # suffixes handles columns that have the same name in both files (e.g. Temperature_file1, Temperature_file2)
        merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer', suffixes=('_file1', '_file2'))

        # 8. Sort and Clean up
        merged_df.sort_index(inplace=True)
        
        # Reset index to make the timestamp a normal column again
        merged_df.reset_index(inplace=True)
        
        # Format the timestamp back to the original YYYYMMDDHHMM format if desired,
        # or keep it as standard ISO format. Here we convert back to string as per input style.
        # usually usually 'MESS_DATUM'
        merged_df.rename(columns={'timestamp_dt': 'MESS_DATUM'}, inplace=True)
        merged_df['MESS_DATUM'] = merged_df['MESS_DATUM'].dt.strftime('%Y%m%d%H%M')

        # 9. Save Output
        print(f"Writing output to {OUTPUT_FILE_NAME}...")
        merged_df.to_csv(
            OUTPUT_FILE_NAME,
            index=False,
            sep=OUTPUT_SEPARATOR,
            decimal=OUTPUT_DECIMAL,
            encoding=OUTPUT_ENCODING,
            na_rep=OUTPUT_NA_REP
        )
        print("Success! Merge complete.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    merge_csvs()