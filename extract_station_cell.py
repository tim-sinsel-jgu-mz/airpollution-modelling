"""Extract the station grid cell from ENVI-met NetCDF output into one CSV.

Set the NetCDF location of each scenario in SCENARIOS (a single .nc file or a folder
with the .nc files of one run) and run the script. Every output time of every
scenario becomes one row of OUT_CSV."""
import glob
import os

import pandas as pd
import xarray as xr

# Scenario name (evaluated day) -> NetCDF file or folder
SCENARIOS = {
        '20240626': r"Y:\BerlinLeipzigerStr_new\20240626_15\NetCDF",
        '20240708': r"Y:\BerlinLeipzigerStr_new\20240708_15\NetCDF",
        '20240715': r"Y:\BerlinLeipzigerStr_new\20240715_15\NetCDF",
        '20241106': r"Y:\BerlinLeipzigerStr_new\20241106_15\NetCDF",
        '20241111': r"Y:\BerlinLeipzigerStr_new\20241111_15\NetCDF",
        '20241122': r"Y:\BerlinLeipzigerStr_new\20241122_15\NetCDF",
}

CELL = {'GridsI': 162, 'GridsJ': 125, 'GridsK': 3}  # station cell, 1.75 m above ground
OUT_CSV = r"D:\Plotting\output\station_cell_all_scenarios.csv"

# NetCDF variable -> CSV column; variables missing from a file are left empty
VARIABLES = {'PM25Conc': 'PM2.5', 'PM10Conc': 'PM10', 'NOConc': 'NO', 'NO2Conc': 'NO2',
             'OzoneConc': 'O3', 'NOxConc': 'NOx'}


def nc_files(source):
    if source.lower().endswith('.nc'):
        return [source] if os.path.isfile(source) else []
    return sorted(glob.glob(os.path.join(source, '*.nc')))


def extract(files):
    chunks = []
    for f in files:
        print(f"    {os.path.basename(f)}")
        with xr.open_dataset(f) as ds:
            present = [v for v in VARIABLES if v in ds]
            pt = ds[present].isel(**CELL).load()
            chunk = pd.DataFrame({VARIABLES[v]: pt[v].values for v in present},
                                 index=pd.to_datetime(pt['Time'].values))
        chunks.append(chunk.reindex(columns=list(VARIABLES.values())))
    df = pd.concat(chunks).sort_index()
    return df[~df.index.duplicated(keep='last')]


if __name__ == '__main__':
    tables = []
    for name, source in SCENARIOS.items():
        files = nc_files(source)
        if not files:
            print(f"{name}: no NetCDF found at {source}, skipped")
            continue
        print(f"{name}: {len(files)} file(s)")
        df = extract(files)
        print(f"    {len(df)} outputs, {df.index.min()} - {df.index.max()}")
        df.index.name = 'Time'
        tables.append(df.reset_index().assign(Scenario=name))

    if tables:
        out = pd.concat(tables)[['Scenario', 'Time'] + list(VARIABLES.values())]
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        out.to_csv(OUT_CSV, index=False, float_format='%.4f')
        print(f"\nWrote {len(out)} rows to {OUT_CSV}")
