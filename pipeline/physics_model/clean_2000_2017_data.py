#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-29
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd


import pandas as pd
import numpy as np

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.download_MeteoSwiss.fetch_data import map_meta_data, replace_nan

# the data should be always start with 00:00:00 and end at 23:00:00
time1 = "2004-02-01T00:00:00"
time2 = "2017-10-31T23:00:00"

time_resolution_list = ["10 minutes", "Hourly"]

for time_resolution in time_resolution_list:

    meta_data = map_meta_data(time_resolution)

    if time_resolution == "10 minutes":
        t = "t"
    elif time_resolution == "Hourly":
        t = "h"
    else:
        print("Unknown time resolution {}".format(time_resolution))

    df1 = pd.read_csv(f"{current_dir}/ogd-smn_mve_{t}_historical_2000-2009.csv", header=0, delimiter=";")
    df2 = pd.read_csv(f"{current_dir}/ogd-smn_mve_{t}_historical_2010-2019.csv", header=0, delimiter=";")

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.reset_index(drop=True)

    # select only the columns in meta_data values
    selected_cols = [col for col in meta_data.values() if col in df.columns]
    df = df[selected_cols]

    # Rename columns to friendly names (keys of meta_data)
    rename_dict = {v: k for k, v in meta_data.items() if v in df.columns}
    df.rename(columns=rename_dict, inplace=True)

    # convert time format
    df["timestamp [UTC+0]"] = pd.to_datetime(df["timestamp [UTC+0]"], format='%d.%m.%Y %H:%M')
    df["timestamp [UTC+0]"] = df["timestamp [UTC+0]"].dt.strftime('%Y-%m-%dT%H:%M:%S')  # '%Y-%m-%dT%H:%M:%S'
    df = replace_nan(df, default_value=0, time_index=1)


    date = np.array(df.iloc[:, 1])
    idx1 = np.where(date == time1)[0][0]
    idx2 = np.where(date == time2)[0][0] + 1

    df = df.iloc[idx1:idx2, :]

    data_type = f"{time1[:4]}_{time2[:4]}_{t}"
    df.to_csv(f"{project_root}/data/SedCas_input/climate_{data_type}.txt", index=False, mode='w')