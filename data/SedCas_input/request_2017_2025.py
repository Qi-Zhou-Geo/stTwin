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
from functions.download_MeteoSwiss.fetch_data import fetch_data4SedCas, replace_nan


df1 = fetch_data4SedCas(station="mve", time_resolution="Hourly",  time_period="2010-2019")
df1 = replace_nan(df1, default_value = 0)

df2 = fetch_data4SedCas(station="mve", time_resolution="Hourly",  time_period="2020-2029")
df2 = replace_nan(df2, default_value = 0)

df3 = fetch_data4SedCas(station="mve", time_resolution="Hourly",  time_period="Current year")
df3 = replace_nan(df3, default_value = 0)

df4 = fetch_data4SedCas(station="mve", time_resolution="Hourly",  time_period="Today")
df4 = replace_nan(df4, default_value = 0)


df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp [UTC+0]'], utc=True)
df = df.sort_values(by='timestamp')
df = df.reset_index(drop=True)
df.drop(columns=['timestamp'], inplace=True)
aaa
# the data should be always start with 00:00:00 and end at 23:00:00
time1 = "2017-01-01T00:00:00"
time2 = "2025-10-01T00:00:00"

date = np.array(df.iloc[:, 1])

idx1 = np.where(date == time1)[0][0]
idx2 = np.where(date == time2)[0][0]

df = df.iloc[idx1:idx2, :]

df.to_csv(f"{current_dir}/climate_2017_2025.txt", index=False, mode='w')