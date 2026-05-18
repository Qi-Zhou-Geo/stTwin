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

# region ### add the sys.path to search for custom modules ###
from pathlib import Path

from scipy.stats import alpha

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# endregion
from func.generator.load_statistics import sta_loader

plt.rcParams.update({"font.size": 7, 
                     "axes.formatter.limits": (-4, 6), 
                     "axes.formatter.use_mathtext": True})

metadata, precp_sta, temp_sta, radiation_sta = sta_loader()
print(metadata)


# Air temperature at 5 cm above grass
# Global radiation; daily mean
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(3, 1)
y_label = [
    f"Daily Total Precipitation\n[mm]",
    f"Daily Mean Temperature\n[degree]",
    f"Sun Radiation\n[W per squared m]",
]
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
        ax.fill_between(
            x, y1, y2, color="C0", label="Mean +- Std.", alpha=0.5, zorder=3
        )
        ax.fill_between(x, y3, y4, color="C2", label="Min to Max", alpha=0.5, zorder=2)

    ax.set_ylabel(f"{y_label[idx]}", fontweight="bold")
    ax.legend(loc="upper right", fontsize="6")
    ax.set_ylim(y_min[idx], y_max[idx])
    ax.set_xlim(1, 366)
    ax.set_xticks(
        [1, 50, 100, 150, 200, 250, 300, 350, 366],
        [1, 50, 100, 150, 200, 250, 300, 350, 366],
    )
    ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

ax.set_xlabel("Day of Year (1931–2025, MeteoSwiss)", fontweight="bold")
plt.tight_layout()
# plt.savefig(f"./stastic_daily.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)
