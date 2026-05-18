#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# endregion

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

climate_forcing_file = f"{project_root}/data/SedCas_input/climate_1981_2025_h.txt"
climate_forcing = pd.read_csv(climate_forcing_file, header=0)
print(climate_forcing.columns)
data_str = climate_forcing.iloc[:, 1].values
precipitation = climate_forcing.iloc[:, 2].values
temperature = climate_forcing.iloc[:, 3].values
radiation = climate_forcing.iloc[:, 4].values

julday_list = []
hourly_list = []
for s in data_str:
    julday_list.append(UTCDateTime(s).julday)
    hourly_list.append(UTCDateTime(s).hour)
julday_list = np.array(julday_list)
hourly_list = np.array(hourly_list)



precp_sta = []
temp_sta = []
radiation_sta = []
for j in range(1, 367):
    for k in range(0, 24):
        idx = np.where( (julday_list == j) | (hourly_list == k) )[0]

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
            np.max(radiation[idx]),
            np.mean(radiation[idx]),
            np.min(radiation[idx]),
            np.std(radiation[idx], ddof=1),
            np.quantile(radiation[idx], 0.05),
            np.quantile(radiation[idx], 0.95),
        ])

precp_sta = np.array(precp_sta)
temp_sta = np.array(temp_sta)
radiation_sta = np.array(radiation_sta)



fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(3, 1)
y_label = [f"Hourly Total Precipitation\n[mm]", f"Hourly Mean Temperature\n[degree]", f"Sun Radiation\n[W per squared m]"]
y_min = [1, -30, 0]
y_max = [100, 30, 400]
for idx, data in enumerate([precp_sta, temp_sta, radiation_sta]):

    ax = plt.subplot(gs[idx])

    max_data = data[:, 0]
    mean_data = data[:, 1]
    min_data = data[:, 2]
    std_data = data[:, 3]
    q5_data = data[:, 4]
    q95_data = data[:, 5]

    x = range(1, len(max_data) + 1)
    y = mean_data
    y1 = mean_data - std_data
    y2 = mean_data + std_data
    y3 = min_data
    y4 = max_data
    # y3 = q5_data
    # y4 = q95_data

    if idx == 1:
        pass
    else:
        # this for radation
        y1 = np.clip(y1, a_min=0, a_max=np.max(y1))
        y2 = np.clip(y2, a_min=0, a_max=np.max(y2))

    ax.plot(x, y, color="black", label="Mean", zorder=4)

    if idx == 0:
        ax.plot(x, y2, color="C0", label="Mean + Std.", zorder=3)
        ax.plot(x, y4, color="C2", label="Max", zorder=2)
        ax.set_yscale("log")
    else:
        ax.fill_between(x, y1, y2, color="C0", label="Mean +- Std.", alpha=0.5, zorder=3)
        ax.fill_between(x, y3, y4, color="C2", label="Min to Max", alpha=0.5, zorder=2)

    ax.set_ylabel(f"{y_label[idx]}", fontweight='bold')
    ax.legend(loc="upper right", fontsize='6')
    # ax.set_ylim(y_min[idx], y_max[idx])
    # ax.set_xlim(1, 366)
    # ax.set_xticks([1, 50, 100, 150, 200, 250, 300, 350, 366],
    #               [1, 50, 100, 150, 200, 250, 300, 350, 366])
    # ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

ax.set_xlabel("Day of Year (1981–2025, MeteoSwiss)", fontweight='bold')
ax.legend(loc="upper left", fontsize='6')
plt.tight_layout()
plt.savefig(f"./stastic_hourly.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)


# details
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(3, 1)
y_label = [f"Hourly Total Precipitation\n[mm]", f"Hourly Mean Temperature\n[degree]", f"Sun Radiation\n[W per squared m]"]
y_min = [1, -30, 0]
y_max = [100, 30, 400]
for idx, data in enumerate([precipitation, temperature, radiation]):

    ax = plt.subplot(gs[idx])

    y_temp = []
    for year in range(1981, 2025 + 1):

        if year % 4 == 0:
            end = 367
        else:
            end = 366

        for julday in range(121, 274): #range(1, end):
            t1 = UTCDateTime(year=year, julday=julday, hour=0).strftime("%Y-%m-%dT%H:%M:%S")
            t2 = UTCDateTime(year=year, julday=julday, hour=23).strftime("%Y-%m-%dT%H:%M:%S")

            id1 = np.where(data_str == t1)[0][0]
            id2 = np.where(data_str == t2)[0][0] + 1

            y = data[id1:id2]
            y_temp.append(y)

        print(f"Done year: {year}")

    # now it has all daily data
    y_temp = np.vstack(y_temp)
    print(f"y_temp.shape: {y_temp.shape}")

    # do stastic
    max_data = np.max(y_temp, axis=0)
    mean_data = np.mean(y_temp, axis=0)
    min_data = np.min(y_temp, axis=0)
    std_data = np.std(y_temp, ddof=1, axis=0)

    y = mean_data
    y1 = mean_data - std_data
    y2 = mean_data + std_data
    y3 = min_data
    y4 = max_data

    x = range(0, 24)
    ax.plot(x, y, color="black", label="Mean", zorder=3)

    if idx == 0:
        ax.plot(x, y2, color="C0", label="Mean + Std.", zorder=3)
        ax.plot(x, y4, color="C2", label="Max", zorder=2)
        ax.set_yscale("log")
    else:
        ax.fill_between(x, y1, y2, color="C0", label="Mean +- Std.", alpha=0.5, zorder=3)
        ax.fill_between(x, y3, y4, color="C2", label="Min to Max", alpha=0.5, zorder=2)

    ax.set_ylabel(f"{y_label[idx]}", fontweight='bold')
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23],
                  [0, 4, 8, 12, 16, 20, 23])
    ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    ax.legend(loc="upper left", fontsize='6')

ax.set_xlabel("Hour of Day (1981-2025, MeteoSwiss)", fontweight='bold')
plt.tight_layout()
plt.savefig(f"./stastic_hourly_raw.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)
