#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

#region ### add the sys.path to search for custom modules ###
from pathlib import Path

from scipy.constants import minute

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# endregion

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})

data_resolution = "t"
climate_forcing_file1 = f"{project_root}/data/SedCas_input/climate_2004_2023_{data_resolution}.txt"
climate_forcing1 = pd.read_csv(climate_forcing_file1, header=0)
climate_forcing_file2 = f"{project_root}/data/SedCas_input/climate_2023_2026_{data_resolution}.txt"
climate_forcing2 = pd.read_csv(climate_forcing_file2, header=0)

climate_forcing = pd.concat([climate_forcing1, climate_forcing2])
date_str = climate_forcing['timestamp [UTC+0]'].values
temperature = climate_forcing['temperature [degree]'].values

# find the max-min temperature in each year
min_day = []
max_day = []
for year in range(2005, 2025):
    t1 = UTCDateTime(year=year, month=1, day=1, hour=0, minute=0).strftime("%Y-%m-%dT%H:%M:%S")
    t2 = UTCDateTime(year=year, month=12, day=31, hour=23, minute=50).strftime("%Y-%m-%dT%H:%M:%S")

    id1 = np.where(date_str == t1)[0][0]
    id2 = np.where(date_str == t2)[0][0] + 1

    date_str_slice = date_str[id1:id2]
    temperature_slice = temperature[id1:id2]

    id3 = np.argmin(temperature_slice)
    id4 = np.argmax(temperature_slice)

    time_min = date_str_slice[id3]
    time_max = date_str_slice[id4]

    min_day.append(UTCDateTime(time_min).julday)
    max_day.append(UTCDateTime(time_max).julday)
min_day = np.array(min_day)
max_day = np.array(max_day)

# find the max-min temperature in each day
min_minute = []
max_minute = []
for year in range(2005, 2025):
    for julday in range(1, 366):
        try:
            t1 = UTCDateTime(year=year, julday=julday, hour=0, minute=0).strftime("%Y-%m-%dT%H:%M:%S")
            t2 = UTCDateTime(year=year, julday=julday, hour=23, minute=50).strftime("%Y-%m-%dT%H:%M:%S")

            id1 = np.where(date_str == t1)[0][0]
            id2 = np.where(date_str == t2)[0][0]

            date_str_slice = date_str[id1:id2]
            temperature_slice = temperature[id1:id2]

            id3 = np.argmin(temperature_slice)
            id4 = np.argmax(temperature_slice)

            time_min = date_str_slice[id3]
            time_max = date_str_slice[id4]

            # from 1 to 1440
            min_minute.append(UTCDateTime(time_min).hour * 60 + UTCDateTime(time_min).minute + 1)
            max_minute.append(UTCDateTime(time_max).hour * 60 + UTCDateTime(time_max).minute + 1)

        except Exception as e:
            print(year, julday)
min_minute = np.array(min_minute)
max_minute = np.array(max_minute)


def get_kde_peak_filled(ax):
    """
    Get the x-value of the peak of the last plotted filled KDE in ax.
    Works when sns.kdeplot(fill=True) creates a PolyCollection.
    """
    # Get all collections (the filled areas)
    collections = ax.collections
    if not collections:
        raise ValueError("No collections found in axis")

    # The last collection corresponds to the last plotted KDE
    poly = collections[-1]
    # Extract the paths (vertices of the filled polygon)
    paths = poly.get_paths()
    if not paths:
        raise ValueError("No paths found in PolyCollection")

    # Use the first path (main KDE curve)
    vertices = paths[0].vertices
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Find the x corresponding to max y (peak)
    peak_idx = y.argmax()

    idx_x, idx_y = x[peak_idx], y[peak_idx]

    return idx_x, idx_y


fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(gs[0])
sns.kdeplot(min_day, color="C0", fill=True, clip=(-366, 366), zorder=3, ax=ax1)
idx_x, idx_y = get_kde_peak_filled(ax1)
ax1.axvline(x=idx_x, color="C0", zorder=2, label=f"Cold Day: {idx_x:.0f}")

sns.kdeplot(max_day, color="black", fill=True, clip=(-366, 366), zorder=4, ax=ax1)
idx_x, idx_y = get_kde_peak_filled(ax1)
ax1.axvline(x=idx_x, color="black", zorder=2, label=f"Warm Day: {idx_x:.0f}")

ax1.set_xlim(-366, 366)
ax1.set_xticks([-366, -200, -100, 0, 100, 200, 366], [-366, -200, -100, 0, 100, 200, 366])
ax1.set_ylabel(f"Kernel Density Estimation", fontweight='bold')
ax1.set_xlabel("Day of Year", fontweight='bold')
ax1.legend(loc="upper left", fontsize=6)


ax2 = plt.subplot(gs[1])
sns.kdeplot(min_minute, color="C0", fill=True, clip=(-1440, 1440), zorder=4, ax=ax2)
idx_x, idx_y = get_kde_peak_filled(ax2)
ax2.axvline(x=idx_x, color="C0", zorder=2, label=f"Cold Minute: {idx_x:.0f}")


sns.kdeplot(max_minute, color="black", fill=True, clip=(-1440, 1440), zorder=4, ax=ax2)
idx_x, idx_y = get_kde_peak_filled(ax2)
ax2.axvline(x=idx_x, color="black", zorder=2, label=f"Warm Minute: {idx_x:.0f}")

ax2.set_xlim(-1440, 1440)
ax2.set_xticks([-1440, -1000, -500, 0, 500, 1000, 1440], [-1440, -1000, -500, 0, 500, 1000, 1440])
ax2.set_ylabel(f"Kernel Density Estimation", fontweight='bold')
ax2.set_xlabel("Minute of Day", fontweight='bold')
ax2.legend(loc="upper left", fontsize=6)


plt.tight_layout()
plt.savefig(f"{current_dir}/temperature_idx.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)

