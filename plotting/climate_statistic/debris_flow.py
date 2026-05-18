#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

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


event_catalog_file = f"{project_root}/data/event_catalog/debris_flow_volume_2004_2022.txt"
event_catalog = pd.read_csv(event_catalog_file, skiprows=6, header=0)
volume_df = event_catalog["Volume[m3]"].copy()

mean_volume = np.nanmean(volume_df)
mean_volume_log = np.nanmean(np.log(volume_df))

std_volume = np.nanstd(volume_df, ddof=1)
std_volume_log = np.nanstd(np.log(volume_df), ddof=1)

num_none_nan_evnet = len(event_catalog) - (np.isnan(event_catalog['Volume[m3]']) == True).sum()

print(f"num_event={num_none_nan_evnet}, \n"
      f"mean_volume={mean_volume:.1f}, \n"
      f"std_volume={std_volume:.1f}, \n"
      f'mean_volume in log={mean_volume_log}, \n'
      f'std_volume in log={std_volume_log}, \n')
# <std_volume in log = 1.1> means typical multiplicative variation
# of your debris-flow volumes is about factor of exp(1.1) ≈ 3.


event_season = []
event_month = []

for idx in range(event_catalog.shape[0]):
    t = UTCDateTime(event_catalog.iloc[idx, 0])
    year = t.year
    month = t.month
    julday= t.julday

    spring_start = UTCDateTime(year, 3, 1).julday # 3, 4, 5
    summer_start = UTCDateTime(year, 6, 1).julday
    autumn_start = UTCDateTime(year, 9, 1).julday
    winter_start = UTCDateTime(year, 12, 1).julday # 12, 1, 2

    if spring_start <= julday < summer_start:
        season = "Spring"
    elif summer_start <= julday < autumn_start:
        season = "Summer"
    elif autumn_start <= julday < winter_start:
        season = "Autumn"
    else:
        season = "Winter"
        print(f"Unknown month {month}")

    event_season.append(season)
    event_month.append(month)

event_catalog = np.array(event_catalog)
event_catalog = np.hstack( (event_catalog, np.array(event_season).reshape(-1, 1)) )
event_catalog = np.hstack( (event_catalog, np.array(event_month).reshape(-1, 1)) )

# prepare the data for plotting
seasonally_plot = {}
for season in ["Spring,[Mar-Apr-May]", "Summer,[Jun-Jul-Aug]", "Autumn,[Sep-Oct-Nov]", "Winter,[Dec-Jan-Feb]"]:

    idx = np.where(event_catalog[:, 3] == season.split(",")[0] )[0]
    volume = event_catalog[idx, 2]

    if len(idx) == 0:
        # no obs
        mean_volume = 0
        std_volume = 0
    elif len(idx) == 1:
        # with only one obs
        mean_volume = np.nanmean(volume)
        std_volume = 0
    else:
        # with multiple obs
        mean_volume = np.nanmean(volume)
        # ddof=0 -> assume the data is the entire population.
        # ddof=1 -> assume the data is a sample from a larger population
        std_volume = np.nanstd(volume, ddof=1)

    season = season.replace(",", "\n")
    seasonally_plot[season] = (len(idx), mean_volume, std_volume)


# prepare the data for plotting
monthly_plot = {}
for month in range(1, 13):

    idx = np.where(event_catalog[:, 4] == month)[0]
    volume = event_catalog[idx, 2]

    if len(idx) == 0:
        # no obs
        mean_volume = 0
        std_volume = 0
    elif len(idx) == 1:
        # with only one obs
        mean_volume = np.nanmean(volume)
        std_volume = 0
    else:
        # with multiple obs
        mean_volume = np.nanmean(volume)
        # ddof=0 -> assume the data is the entire population.
        # ddof=1 -> assume the data is a sample from a larger population
        std_volume = np.nanstd(volume, ddof=1)

    monthly_plot[month] = (len(idx), mean_volume, std_volume)



# plot it
fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 2)

ax = plt.subplot(gs[0])
ax.set_title("(a)", fontweight='bold', fontsize=7, loc='left')
for key, value in monthly_plot.items():
    num_event, mean_volume, std_volume = value

    # prevent lower error from going below zero
    yerr_lower = np.clip(std_volume, a_min=None, a_max=mean_volume)
    yerr_upper = std_volume  # upper error as usual

    # use asymmetric error bars
    ax.errorbar(key, mean_volume, yerr=[[yerr_lower], [yerr_upper]],
                label="Mean ± Std.",
                fmt='o', color='black', ecolor='gray', capsize=3, alpha=0.7)

ax.set_yscale('log')
ax.set_ylim(1e3, 1e5)

ax.set_xticks(range(1, 13), range(1, 13))
ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc='lower left')

ax.set_ylabel("Debris Flow Volume [m³]", fontweight='bold')
ax.set_xlabel("Month", fontweight='bold')



ax = plt.subplot(gs[1])
ax.set_title("(b)", fontweight='bold', fontsize=7, loc='left')
for key, value in seasonally_plot.items():
    num_event, mean_volume, std_volume = value

    # prevent lower error from going below zero
    yerr_lower = np.clip(std_volume, a_min=None, a_max=mean_volume)
    yerr_upper = std_volume  # upper error as usual

    # use asymmetric error bars
    ax.errorbar(key, mean_volume, yerr=[[yerr_lower], [yerr_upper]],
                label="Mean ± Standard Deviation",
                fmt='o', color='black', ecolor='gray', capsize=3, alpha=0.7)

ax.set_yscale('log')
ax.set_ylim(1e3, 1e5)

ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc='lower left')

ax.set_ylabel("", fontweight='bold')
ax.set_xlabel("Season", fontweight='bold')

plt.tight_layout()
plt.savefig(f"{current_dir}/debris_flow.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)

