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

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# </editor-fold>

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})


event_catalog_file = f"{project_root}/data/SedCas_input/climate_2004_2023_h.txt"
event_catalog = pd.read_csv(event_catalog_file, header=0)

event_season = []
event_month = []
event_day = []
event_hour = []
event_minute = []

for idx in range(event_catalog.shape[0]):
    t = UTCDateTime(event_catalog.iloc[idx, 1])

    year = t.year
    month = t.month
    julday = t.julday
    hour = t.hour
    minute = t.minute

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

    event_season.append(season)
    event_month.append(month)
    event_day.append(julday)
    event_hour.append(hour)
    event_minute.append(minute)

event_catalog = np.array(event_catalog)
event_catalog = np.hstack( (event_catalog, np.array(event_season).reshape(-1, 1)) )
event_catalog = np.hstack( (event_catalog, np.array(event_month).reshape(-1, 1)) )
event_catalog = np.hstack( (event_catalog, np.array(event_day).reshape(-1, 1)) )
event_catalog = np.hstack( (event_catalog, np.array(event_hour).reshape(-1, 1)) )
event_catalog = np.hstack( (event_catalog, np.array(event_minute).reshape(-1, 1)) )

mapping_element = {"Precipitation [mm per time_step]":2, "Temperature [degree]":3, "Sun Radiation [W per squared m]":4}
for item, item_value in mapping_element.items():

    item = item.replace("time_step", "1 hour")
    # prepare the data for plotting
    seasonally_plot = {}
    for season in ["Spring,[Mar-Apr-May]", "Summer,[Jun-Jul-Aug]", "Autumn,[Sep-Oct-Nov]", "Winter,[Dec-Jan-Feb]"]:

        idx = np.where(event_catalog[:, 5] == season.split(",")[0] )[0]
        value = event_catalog[idx, item_value]

        if len(idx) == 0:
            # no obs
            mean_value = 0
            std_value = 0
        elif len(idx) == 1:
            # with only one obs
            mean_value = np.nanmean(value)
            std_value = 0
        else:
            # with multiple obs
            mean_value = np.nanmean(value)
            # ddof=0 -> assume the data is the entire population.
            # ddof=1 -> assume the data is a sample from a larger population
            std_value = np.nanstd(value, ddof=1)

        season = season.replace(",", "\n")
        seasonally_plot[season] = (mean_value, std_value)


    # prepare the data for plotting
    monthly_plot = {}
    for month in range(1, 13):

        idx = np.where(event_catalog[:, 6] == month)[0]
        value = event_catalog[idx, item_value]

        if len(idx) == 0:
            # no obs
            mean_value = 0
            std_value = 0
        elif len(idx) == 1:
            # with only one obs
            mean_value = np.nanmean(value)
            std_value = 0
        else:
            # with multiple obs
            mean_value = np.nanmean(value)
            # ddof=0 -> assume the data is the entire population.
            # ddof=1 -> assume the data is a sample from a larger population
            std_value = np.nanstd(value, ddof=1)

        monthly_plot[month] = (mean_value, std_value)


    # prepare the data for plotting
    daily_plot = {}
    for day in range(1, 366):

        idx = np.where(event_catalog[:, 7] == day)[0]
        value = event_catalog[idx, item_value]

        if len(idx) == 0:
            # no obs
            mean_value = 0
            std_value = 0
        elif len(idx) == 1:
            # with only one obs
            mean_value = np.nanmean(value)
            std_value = 0
        else:
            # with multiple obs
            mean_value = np.nanmean(value)
            # ddof=0 -> assume the data is the entire population.
            # ddof=1 -> assume the data is a sample from a larger population
            std_value = np.nanstd(value, ddof=1)

        daily_plot[day] = (mean_value, std_value)


    # prepare the data for plotting
    hourly_plot = {}
    for hour in range(0, 24):

        idx = np.where(event_catalog[:, 8] == hour)[0]
        value = event_catalog[idx, item_value]

        if len(idx) == 0:
            # no obs
            mean_value = 0
            std_value = 0
        elif len(idx) == 1:
            # with only one obs
            mean_value = np.nanmean(value)
            std_value = 0
        else:
            # with multiple obs
            mean_value = np.nanmean(value)
            # ddof=0 -> assume the data is the entire population.
            # ddof=1 -> assume the data is a sample from a larger population
            std_value = np.nanstd(value, ddof=1)

        hourly_plot[hour] = (mean_value, std_value)


    # plot it
    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(2, 2)

    ax = plt.subplot(gs[0])
    ax.set_title("(a)", fontweight='bold', fontsize=7, loc='left')
    for key, value in hourly_plot.items():
        mean_value, std_value = value


        if item == "Temperature [degree]":
            yerr_lower = std_value  # lower error as usual
        else:
            # prevent lower error from going below zero
            yerr_lower = np.clip(std_value, a_min=None, a_max=mean_value)
        yerr_upper = std_value  # upper error as usual

        # use asymmetric error bars
        ax.errorbar(key, mean_value, yerr=[[yerr_lower], [yerr_upper]],
                    label="Mean ± Std.",
                    fmt='o', color='black', ecolor='gray', capsize=3, alpha=0.7)

    ax.set_xticks([0, 6, 12, 18, 23], [0, 6, 12, 18, 23])
    ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc='lower left')

    ax.set_ylabel(f"{item}", fontweight='bold')
    ax.set_xlabel("Hour", fontweight='bold')



    ax = plt.subplot(gs[1])
    ax.set_title("(b)", fontweight='bold', fontsize=7, loc='left')
    for key, value in daily_plot.items():
        mean_value, std_value = value

        if item == "Temperature [degree]":
            yerr_lower = std_value  # lower error as usual
        else:
            # prevent lower error from going below zero
            yerr_lower = np.clip(std_value, a_min=None, a_max=mean_value)
        yerr_upper = std_value  # upper error as usual

        # use asymmetric error bars
        ax.errorbar(key, mean_value, yerr=[[yerr_lower], [yerr_upper]],
                    label="Mean ± Standard Deviation",
                    fmt='o', color='black', ecolor='gray', capsize=3, alpha=0.7)

    ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)

    # ax.set_xticks(range(1, 366), range(0, 23))
    ax.set_ylabel("", fontweight='bold')
    ax.set_xlabel("Julday", fontweight='bold')



    ax = plt.subplot(gs[2])
    ax.set_title("(c)", fontweight='bold', fontsize=7, loc='left')
    for key, value in monthly_plot.items():
        mean_value, std_value = value

        if item == "Temperature [degree]":
            yerr_lower = std_value  # lower error as usual
        else:
            # prevent lower error from going below zero
            yerr_lower = np.clip(std_value, a_min=None, a_max=mean_value)
        yerr_upper = std_value  # upper error as usual

        # use asymmetric error bars
        ax.errorbar(key, mean_value, yerr=[[yerr_lower], [yerr_upper]],
                    label="Mean ± Standard Deviation",
                    fmt='o', color='black', ecolor='gray', capsize=3, alpha=0.7)

    ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)

    ax.set_xticks(range(1, 13), range(1, 13))

    ax.set_ylabel(f"{item}", fontweight='bold')
    ax.set_xlabel("Month", fontweight='bold')


    ax = plt.subplot(gs[3])
    ax.set_title("(d)", fontweight='bold', fontsize=7, loc='left')
    for key, value in seasonally_plot.items():
        mean_value, std_value = value

        if item == "Temperature [degree]":
            yerr_lower = std_value  # lower error as usual
        else:
            # prevent lower error from going below zero
            yerr_lower = np.clip(std_value, a_min=None, a_max=mean_value)
        yerr_upper = std_value  # upper error as usual

        # use asymmetric error bars
        ax.errorbar(key, mean_value, yerr=[[yerr_lower], [yerr_upper]],
                    label="Mean ± Standard Deviation",
                    fmt='o', color='black', ecolor='gray', capsize=3, alpha=0.7)

    ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    ax.set_ylabel(f"", fontweight='bold')
    ax.set_xlabel("Season", fontweight='bold')




    plt.tight_layout()
    plt.savefig(f"{current_dir}/climate_forcing_{item}.png", dpi=600)  # , transparent=True
    plt.show()
    plt.close(fig=fig)

