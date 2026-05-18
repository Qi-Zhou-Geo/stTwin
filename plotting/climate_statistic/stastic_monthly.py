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

climate_forcing_file = f"{project_root}/data/SedCas_input/climate_1926_2026_m.txt"
climate_forcing = pd.read_csv(climate_forcing_file, header=0)
print(climate_forcing.columns)
data_str = climate_forcing.iloc[:, 1].values
precipitation = climate_forcing.iloc[:, 2].values
temperature = climate_forcing.iloc[:, 3].values
radiation = climate_forcing.iloc[:, 4].values

month_list = []
for s in data_str:
    month_list.append(UTCDateTime(s).month)
month_list = np.array(month_list)



precp_sta = []
temp_sta = []
radiation_sta = []
for j in range(1, 13):
    idx = np.where(month_list == j)[0]

    precp_sta.append([
        np.max(precipitation[idx]),
        np.mean(precipitation[idx]),
        np.min(precipitation[idx]),
        np.std(precipitation[idx], ddof=1)
    ])

    temp_sta.append([
        np.max(temperature[idx]),
        np.mean(temperature[idx]),
        np.min(temperature[idx]),
        np.std(temperature[idx], ddof=1)
    ])

    radiation_sta.append([
        np.max(radiation[idx]),
        np.mean(radiation[idx]),
        np.min(radiation[idx]),
        np.std(radiation[idx], ddof=1)
    ])


precp_sta = np.array(precp_sta)
temp_sta = np.array(temp_sta)
radiation_sta = np.array(radiation_sta)

fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(3, 1)
y_label = [f"Precipitation\n[mm per time_step]", f"Temperature\n[degree]", f"Sun Radiation\n[W per squared m]"]
for idx, data in enumerate([precp_sta, temp_sta, radiation_sta]):

    ax = plt.subplot(gs[idx])

    max_data = data[:, 0]
    mean_data = data[:, 1]
    min_data = data[:, 2]
    std_data = data[:, 3]

    x = range(0, len(max_data))
    y = mean_data
    y1 = mean_data - std_data
    y2 = mean_data + std_data
    y3 = min_data
    y4 = max_data

    if idx == 2:
        # this radation
        y1 = np.clip(y1, a_min=0, a_max=np.max(y1))
        y2 = np.clip(y2, a_min=0, a_max=np.max(y2))
        y3 = np.clip(y3, a_min=0, a_max=np.max(y3))
        y4 = np.clip(y4, a_min=0, a_max=np.max(y4))

    ax.plot(x, y, color="black", label="mean", zorder=3)
    ax.fill_between(x, y1, y2, color="C0", label="mean +- std.", alpha=0.2, zorder=2)
    ax.fill_between(x, y3, y4, color="C1", label="min to max", alpha=0.2, zorder=1)

    ax.set_ylabel(f"{y_label[idx]}", fontweight='bold')
    ax.set_xlabel("Month of Year [monthly resolution]", fontweight='bold')

ax.legend(loc="upper left", fontsize='6')
plt.tight_layout()
plt.savefig(f"./stastic_monthly.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)
