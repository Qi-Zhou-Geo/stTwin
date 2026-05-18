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


temp_df = []
time_resolution = "Hourly"
for time_period in ["1980-1989", "1990-1999", "2000-2009", "2010-2019", "2020-2029"]:
    df = fetch_data4SedCas(station="mve", time_resolution=time_resolution, time_period=time_period)
    df = replace_nan(df, default_value=0)
    temp_df.append(df)

df = pd.concat(temp_df, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp [UTC+0]'], utc=True)
df = df.sort_values(by='timestamp')
df = df.reset_index(drop=True)
df.drop(columns=['timestamp'], inplace=True)
df.to_csv(f"{current_dir}/climate_1981_2025_h.txt", index=False, mode='w')






temp_df = []
time_resolution = "Daily"
time_period = "historical"
df = fetch_data4SedCas(station="mve", time_resolution=time_resolution, time_period=time_period)
df = replace_nan(df, default_value=0)
temp_df.append(df)

df = pd.concat(temp_df, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp [UTC+0]'], utc=True)
df = df.sort_values(by='timestamp')
df = df.reset_index(drop=True)
df.drop(columns=['timestamp'], inplace=True)
df.to_csv(f"{current_dir}/climate_1931_2025_d.txt", index=False, mode='w')



temp_df = []
time_resolution = "Monthly"
time_period = "historical"
df = fetch_data4SedCas(station="mve", time_resolution=time_resolution, time_period=time_period)
df = replace_nan(df, default_value=0)
temp_df.append(df)

df = pd.concat(temp_df, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp [UTC+0]'], utc=True)
df = df.sort_values(by='timestamp')
df = df.reset_index(drop=True)
df.drop(columns=['timestamp'], inplace=True)
df.to_csv(f"{current_dir}/climate_1931_2025_m.txt", index=False, mode='w')
