#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-29
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd


import pandas as pd
import numpy as np

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.download_MeteoSwiss.fetch_data import fetch_data4SedCas, replace_nan

def request_1h_data(time_resolution="Hourly", time1="2023-01-01T00:00:00", time2="2026-03-01T00:00:00"):

    df0 = fetch_data4SedCas(station="mve", time_resolution=time_resolution,  time_period="2020-2029")
    df0 = replace_nan(df0, default_value = 0)

    df1 = fetch_data4SedCas(station="mve", time_resolution=time_resolution,  time_period="Current year")
    df1 = replace_nan(df1, default_value = 0)

    df = pd.concat([df0, df1], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp [UTC+0]'], utc=True)
    df = df.sort_values(by='timestamp')
    df = df.reset_index(drop=True)
    df.drop(columns=['timestamp'], inplace=True)

    date = np.array(df.iloc[:, 1])

    idx1 = np.where(date == time1)[0][0]
    idx2 = np.where(date == time2)[0][0]

    df = df.iloc[idx1:idx2, :]

    if time_resolution == "Hourly":
        marker = "h"
    elif time_resolution == "10 minutes":
        marker = "t"
    else:
        raise ValueError("time_resolution must be 'Hourly' or '10'")

    df.to_csv(f"{current_dir}/climate_{time1[:4]}_{time2[:4]}_{marker}.txt", index=False, mode='w')

if __name__ == "__main__":
    # the data should be always start with 00:00:00 and end at 23:00:00
    time1 = "2023-01-01T00:00:00" # 10 minutes data start at 2004-02-01T00:00:00
    time2 = "2026-03-01T00:00:00"
    time_resolution = "10 minutes"

    request_1h_data(time_resolution="10 minutes")