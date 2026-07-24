#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import numpy as np
import pandas as pd

import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator


from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# region <add Arial font>
import platform, getpass
# Specify the directory containing the Arial font
if platform.system() == "Linux" and getpass.getuser() == "qizhou":

    from matplotlib import font_manager
    font_dirs = ['/storage/vast-gfz-hpc-01/home/qizhou/2python/font']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
# endregion

# import custom func.
from func.toolkit.physical_unit_converter import unit_converter


plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})

precp_path = Path(project_root) / "data/SedCas_input/climate_2004_2025_t.txt"
df = pd.read_csv(precp_path, header=0)
date_str = np.array(df.iloc[:, 1])
precp = np.array(df.iloc[:, 2])

color_map = {2009: "blue", 2010: "C4", 2011: "C5"}

fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 2)
ax_line = plt.subplot(gs[0])
ax_bar = plt.subplot(gs[1])

for idx, year in enumerate(range(2005, 2026, 1)):
    
    id1 = np.where(date_str == f"{year}-01-01T00:00:00")[0][0]
    id2 = np.where(date_str == f"{year}-12-31T23:50:00")[0][0] + 1
    
    # select one year
    year_dates = pd.to_datetime(date_str[id1:id2])
    year_precp = precp[id1:id2]

    # remove Feb 29
    mask = ~((year_dates.month == 2) & (year_dates.day == 29))

    year_dates = year_dates[mask]
    year_precp = year_precp[mask]

    # cumulative precipitation
    select_precp = np.cumsum(year_precp)

    # day of year after removing Feb 29
    x = year_dates.dayofyear

    # shift days after Feb 29 back by one day
    x = np.where(x > 59, x - 1, x)

    if year in [2009, 2010, 2011]:
        label = f"{year} ({select_precp[-1]:.1f} mm)"
    else:
        label = None
    
    print(year, select_precp[-1])
    color = color_map.get(year, "black")
    ax_line.plot(x, select_precp, color=color, alpha=0.5, zorder=4, label=label)
    ax_bar.bar(year, select_precp[-1], color=color, alpha=0.5, zorder=3)


ax = plt.subplot(gs[0])
ax.set_title("(a)", fontweight='bold', fontsize=7, loc='left')
ax.set_ylim(0, 1800)
ax.set_xlim(1, 365)
ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

month_ticks = [1, 32, 60, 91, 121, 152, 
               182, 213, 244, 274, 305, 335, 
               365]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
               "Jan"]

ax.set_xticks(month_ticks)
ax.set_xticklabels(month_names)

ax.set_xlabel("Month", fontweight='bold')
ax.set_ylabel("Cumulative Precipitation [mm]", fontweight='bold')
ax.axhline(y=513.8, color="C1", linestyle="--", label="Historical Minimum (513.8 mm)", zorder=2)
ax.axhline(y=919.7, color="C2", linestyle="--", label="Historical Mean (919.7 mm)", zorder=2)
ax.axhline(y=1608.6, color="C3", linestyle="--", label="Historical Maximum (1608.6 mm)", zorder=2)

ax.legend(loc="upper left", fontsize=6)


ax = plt.subplot(gs[1])
ax.set_title("(b)", fontweight='bold', fontsize=7, loc='left')
ax.set_ylim(0, 1800)
ax.set_xlim(2004, 2026)
ax.set_xlabel("Year", fontweight='bold')
ax.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.axhline(y=513.8, color="C1", linestyle="--", label="Historical Minimum (513.8 mm)", zorder=2)
ax.axhline(y=919.7, color="C2", linestyle="--", label="Historical Mean (919.7 mm)", zorder=2)
ax.axhline(y=1608.6, color="C3", linestyle="--", label="Historical Maximum (1608.6 mm)", zorder=2)





ax_line = plt.subplot(gs[2])
ax_bar = plt.subplot(gs[3])

for idx, year in enumerate(range(2005, 2026, 1)):
    
    id1 = np.where(date_str == f"{year}-04-15T00:00:00")[0][0]
    id2 = np.where(date_str == f"{year}-10-15T23:50:00")[0][0] + 1
    
    # select one year
    year_dates = pd.to_datetime(date_str[id1:id2])
    year_precp = precp[id1:id2]

    # remove Feb 29
    mask = ~((year_dates.month == 2) & (year_dates.day == 29))

    year_dates = year_dates[mask]
    year_precp = year_precp[mask]

    # cumulative precipitation
    select_precp = np.cumsum(year_precp)

    # day of year after removing Feb 29
    x = year_dates.dayofyear

    # shift days after Feb 29 back by one day
    x = np.where(x > 59, x - 1, x)

    if year in [2009, 2010, 2011]:
        label = year
    else:
        label = None
    
    print(year, select_precp[-1])
    color = color_map.get(year, "black")
    ax_line.plot(x, select_precp, color=color, alpha=0.5, zorder=4, label=label)
    ax_bar.bar(year, select_precp[-1], color=color, alpha=0.5, zorder=3)


ax = plt.subplot(gs[2])
ax.set_title("(c)", fontweight='bold', fontsize=7, loc='left')
ax.set_ylim(0, 800)
ax.set_xlim(1, 365)
ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

month_ticks = [1, 32, 60, 91, 121, 152, 
               182, 213, 244, 274, 305, 335, 
               365]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
               "Jan"]

ax.set_xticks(month_ticks)
ax.set_xticklabels(month_names)

ax.set_xlabel("Month", fontweight='bold')
ax.set_ylabel("Cumulative Precipitation [mm]", fontweight='bold')
ax.axhline(y=513.8, color="C1", linestyle="--", label="Historical Minimum (513.8 mm)", zorder=2)
ax.axhline(y=919.7, color="C2", linestyle="--", label="Historical Mean (919.7 mm)", zorder=2)
ax.axhline(y=1608.6, color="C3", linestyle="--", label="Historical Maximum (1608.6 mm)", zorder=2)


ax = plt.subplot(gs[3])
ax.set_title("(d)", fontweight='bold', fontsize=7, loc='left')
ax.set_ylim(0, 800)
ax.set_xlim(2004, 2026)
ax.set_xlabel("Year", fontweight='bold')
ax.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.axhline(y=513.8, color="C1", linestyle="--", label="Historical Minimum (513.8 mm)", zorder=2)
ax.axhline(y=919.7, color="C2", linestyle="--", label="Historical Mean (919.7 mm)", zorder=2)
ax.axhline(y=1608.6, color="C3", linestyle="--", label="Historical Maximum (1608.6 mm)", zorder=2)



png_path = Path(current_dir) / f"cum_precp.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)