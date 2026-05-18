#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-20
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import numpy as np
import pandas as pd
import xarray as xr

from obspy import UTCDateTime

# region find project root
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# endregion

def cal_sta(cached_file, climate_forcing_file):
    
    climate_forcing = pd.read_csv(climate_forcing_file, header=0)
    print(climate_forcing.columns)
    data_str = climate_forcing.iloc[:, 1].values
    precipitation = climate_forcing.iloc[:, 2].values
    temperature = climate_forcing.iloc[:, 3].values
    sun_radiation = climate_forcing.iloc[:, 4].values

    julday_list = []
    for s in data_str:
        julday_list.append(UTCDateTime(s).julday)
    julday_list = np.array(julday_list)

    days = np.arange(1, 367)
    precp_sta = []
    temp_sta = []
    radiation_sta = []
    for j in days:
        idx = np.where(julday_list == j)[0]

        precp_sta.append([
            np.max(precipitation[idx]),
            np.mean(precipitation[idx]),
            np.min(precipitation[idx]),
            np.std(precipitation[idx], ddof=1),
            np.quantile(precipitation[idx], 0.05),
            np.quantile(precipitation[idx], 0.95),
        ])

        temp_sta.append([
            np.max(temperature[idx]),
            np.mean(temperature[idx]),
            np.min(temperature[idx]),
            np.std(temperature[idx], ddof=1),
            np.quantile(temperature[idx], 0.05),
            np.quantile(temperature[idx], 0.95),
        ])

        radiation_sta.append([
            np.max(sun_radiation[idx]),
            np.mean(sun_radiation[idx]),
            np.min(sun_radiation[idx]),
            np.std(sun_radiation[idx], ddof=1),
            np.quantile(sun_radiation[idx], 0.05),
            np.quantile(sun_radiation[idx], 0.95),
        ])

    precp_sta = np.array(precp_sta, dtype=float)
    temp_sta = np.array(temp_sta, dtype=float)
    radiation_sta = np.array(radiation_sta, dtype=float)
    
    metadata = ["max", "mean", "min", "std", "Q5", "Q95"]
    note = (
        f"Note:\n"
        f"This data cached at: {UTCDateTime.now().isoformat()}.\n"
        f"Statistics are based on <daily> resolution data from 1931 to 2025 MVE."
    )

    ds = xr.Dataset(
        data_vars={
            "precipitation": (["day", "stats"], precp_sta, {"units": "Daily Total mm"}),
            "temperature": (["day", "stats"], temp_sta, {"units": "Daily Mean degC"}),
            "radiation": (["day", "stats"], radiation_sta, {"units": "W/m2"}),
        },
        coords={
            "day": days,
            "stats": metadata
        },
        attrs={"note": note}
    )
    
    ds.to_netcdf(cached_file)

    return metadata, precp_sta, temp_sta, radiation_sta

def sta_loader(cached_file=None):
    
    if cached_file is None:
        cached_file=f"{project_root}/data/climate_statistics/cached_sta.nc"
    else:
        cached_file = cached_file
    
    cached_file = Path(cached_file)
    if cached_file.exists():
        with xr.open_dataset(cached_file) as ds:
            metadata = ds.coords["stats"].values
            precp_sta = ds.precipitation.values
            temp_sta = ds.temperature.values
            radiation_sta = ds.radiation.values

        print(f"Load cached data from: {cached_file}")
    else:
        climate_forcing_file = f"{project_root}/data/SedCas_input/climate_1931_2025_d.txt"
        metadata, precp_sta, temp_sta, radiation_sta = cal_sta(cached_file, climate_forcing_file)
        print(f"Re-calculate from txt file: {climate_forcing_file}.")

    return metadata, precp_sta, temp_sta, radiation_sta

def main():
    
    metadata, precp_sta, temp_sta, radiation_sta = sta_loader()

if __name__ == "__main__":
    main()