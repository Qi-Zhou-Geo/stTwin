#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import pandas as pd
import numpy as np

import yaml

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.download_MeteoSwiss.fetch_data import fetch_data4SedCas



# selected_period = ["2017-05-18T00:00:00", "2017-07-01T00:00:00"]
# selected_period = ["2018-06-01T00:00:00", "2018-08-15T00:00:00"]
# selected_period = ["2019-05-25T00:00:00", "2019-08-25T00:00:00"]
# selected_period = ["2020-05-29T00:00:00", "2020-09-25T00:00:00"]
# selected_period = ["2022-06-01T00:00:00", "2022-09-15T00:00:00"]

# selected_period = ["2023-06-06T00:00:00", "2023-11-02T00:00:00"]
selected_period = ["2024-06-05T00:00:00", "2024-11-01T00:00:00"]


df3 = pd.read_csv(f"/Users/qizhou/Desktop/2024_R9BF5_EHZ_all_B.txt", header=0)

# df3 = pd.read_csv(f"{project_root}/data/seismic_temp/seis_energy/{selected_period[0][:4]}_ILL12_EHZ_all_B.txt", header=0)
df3 = df3.rename(columns={"time_window_start": "timestamp [UTC+0]"})
id1 = df3.index[df3['timestamp [UTC+0]'] == selected_period[0]][0]
id2 = df3.index[df3['timestamp [UTC+0]'] == selected_period[1]][0] # + 1 # +1 with 00:00:00
df3 = df3.iloc[id1:id2]

# convert to array
df_arr = np.array(df3)

day_of_year = []
time_of_day = []
for t in df_arr[:, 0]:
    dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
    julday = dt.timetuple().tm_yday
    day_of_year.append(julday)
    time_of_day.append(t.split("T")[1])

day_of_year = np.array(day_of_year).reshape(-1, 1)
time_of_day = np.array(time_of_day).reshape(-1, 1)
temp_t = np.hstack((day_of_year, time_of_day))
df_arr_t = np.hstack((temp_t, df_arr))


# <editor-fold desc="find the event julday">
default_data_path = f"{project_root}/config/data_path.yaml"
with open(default_data_path, "r") as f:
    config = yaml.safe_load(f)
    sac_path = config[f"glic_sac_dir"]
    event_catalog_version = config[f"event_catalog_version"]

file_path = f"{project_root}/data/event_catalog/{event_catalog_version}"
df = pd.read_csv(f"{file_path}", header=0)

event_julday = []
for idx in range(len(df)):
    row_idx = df.loc[idx]  # select row_idx

    catchment = row_idx["Catchment"]
    data_start = row_idx["Manually-Start-time(UTC+0)"]
    data_end = row_idx["Manually-End-time(UTC+0)"]
    sta_s = row_idx["Start-time(UTC+0)-by-STA/LTA"]
    sta_e = row_idx["End-time(UTC+0)-by-STA/LTA"]

    if catchment == "Illgraben" and data_start[:4] == selected_period[0][:4]:
        for idy in [data_start, data_end, sta_s, sta_e]:
            dt = datetime.fromisoformat(idy).replace(tzinfo=timezone.utc)
            julday = dt.timetuple().tm_yday
            event_julday.append(julday)
    else:
        pass

event_julday = np.unique(event_julday)
# </editor-fold>



# 3D arr, shape by [how many days, how many segement in one day, how many features in one segement]
day_of_year = day_of_year.squeeze()
mask = ~np.isin(day_of_year.astype(float), event_julday)
temp_arr_filtered = df_arr_t[mask]

num_non_event_day = len(np.unique(day_of_year)) - len(event_julday)
num_features = df_arr_t.shape[1]
# num_steps_per_day = 24 * 3600 /
arr3d = temp_arr_filtered.reshape(num_non_event_day, 1440, df_arr_t.shape[1])
arr3d = arr3d[:, :, 6:]
mean_arr3d = arr3d.mean(axis=0)
print(mean_arr3d.shape)

plt.rcParams.update({'font.size': 7,
                     'font.family': "Arial",
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

plt.title(f"{selected_period[0][:4]}")
plt.plot(mean_arr3d[:, 11], label="ES_0, 1-5 Hz")
plt.plot(mean_arr3d[:, 12], label="ES_1, 5-15 Hz")
plt.plot(mean_arr3d[:, 13], label="ES_2, 15-25 Hz")
plt.plot(mean_arr3d[:, 14], label="ES_3, 25-35 Hz")
plt.plot(mean_arr3d[:, 15], label="ES_4, 35-45 Hz")
plt.axvline(x=720)
plt.legend()
plt.show()

