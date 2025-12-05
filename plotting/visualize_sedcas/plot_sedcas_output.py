#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import pandas as pd
import numpy as np

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


plt.rcParams.update( {'font.size':7,
                      'axes.formatter.limits': (-2, 3),
                      'axes.formatter.use_mathtext': True} )


#selected_period = ["2017-05-01 00:00:00", "2017-07-01 00:00:00"]
selected_period = ["2018-06-01 00:00:00", "2018-08-15 00:00:00"]


df1 = pd.read_csv(f"{project_root}/data/SedCas_output/Hydro_2017-2025.txt")
df1_arr = np.array(df1)
df1_label = df1.columns

df2 = pd.read_csv(f"{project_root}/data/SedCas_output/Sediment_2017-2025.txt")
df2_arr = np.array(df2)
df2_label = df2.columns


df_arr_selected = []
for idx, df in enumerate([df1_arr, df2_arr, df2_arr]):
    date = np.array(df1.iloc[:, 0])

    id1 = np.where(date == selected_period[0])[0][0]
    id2 = np.where(date == selected_period[1])[0][0]

    df_arr_selected.append(df[id1:id2, :])

df1_arr, df2_arr = df_arr_selected[0], df_arr_selected[1]


# model output
df3 = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2017_2025.txt")
df3_arr = np.array(df3)
df3_label = df3.columns
date = df3_arr[:, 1]
id1 = np.where(date == selected_period[0].replace(" ", "T"))[0][0]
id2 = np.where(date == selected_period[1].replace(" ", "T"))[0][0]
df3_arr = df3_arr[id1:id2, :]



# plot it
sps_data = 1/3600 # unit is Hz, data per second
x_interval = 15 * 24 # unit is hour

fig = plt.figure(figsize=(5.5, 9))
gs = gridspec.GridSpec(5, 1)

for i, (idx, df, label) in enumerate(zip([1, 2, 3, 2],
                                         [df1_arr, df1_arr, df1_arr, df3_arr],
                                         ["Q", "Qs", "Qss", "Rainfall"])):
    ax = plt.subplot(gs[i], label=label)
    plt.plot(df[:, idx].astype(float))
    ax.set_ylabel(f"{label}")
    ax.set_yscale("log")
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))

#plt.legend()

ax = plt.subplot(gs[4], label=label)
idx = 5
x = np.arange(df2_arr[:, idx].shape[0])
plt.plot(x, df2_arr[:, idx], label=df2_label[idx])
ax.set_ylabel("Sediments ouput")
# idy, idz = 2, 8 # Q5 Q95, they seems similar
# plt.fill_between(x, df2_arr[:, idy].astype(float),
#                  df2_arr[:, idz].astype(float),
#                  color='blue', alpha=0.2, label=f"{df2_label[idy]} to {df2_label[idz]}")
plt.legend()

t1 = datetime.strptime(selected_period[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
t2 = datetime.strptime(selected_period[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

duration = int((t2-t1) / 3600)
xLocation = np.arange(0, sps_data * 3600 * (duration + x_interval), sps_data * 3600 * x_interval)
xTicks = []
for idx, i in enumerate(xLocation):
    if idx == 0:
        t_temp = datetime.fromtimestamp(t1 + i * 1 / sps_data, tz=timezone.utc).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S')
    else:
        t_temp = datetime.fromtimestamp(t1 + i * 1 / sps_data, tz=timezone.utc).strftime('%Y-%m-%d')
    xTicks.append(t_temp)

ax.set_xticks(xLocation, xTicks)
ax.set_xlabel("Time [Hourly interval]")



plt.tight_layout()
#plt.savefig(f"{current_dir}/discharge_{selected_period[0]}_{selected_period[1]}.png", dpi=600)
plt.show()

