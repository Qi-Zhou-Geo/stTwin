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

monitor_MAP = Path(project_root) / f'pipeline/run_2004_2025_posterior/v0dot4/theta_001/sed_container.nc'
ds = xr.load_dataset(monitor_MAP)
time_str = ds.coords["time_str"].values


ticks_location = [0]
ticks_label = ["2004-02-01"]
for year in [2010, 2015, 2020]:
    label = f"{year}-01-01T00:00:00"
    idx = np.where(time_str == label)[0][0]
    
    ticks_location.append(idx)
    label = f"{year}-01-01"
    ticks_label.append(label)
ticks_location.append(len(time_str))
ticks_label.append("2026-01-01")


minor_ticks = []
for year in range(2005, 2025 + 1, 1):
    label = f"{year}-01-01T00:00:00"
    idx = np.where(time_str == label)[0][0]
    minor_ticks.append(idx)
    
    

fig = plt.figure(figsize=(6.5, 6))
gs = gridspec.GridSpec(2, 1)

color = "black"
label = "Synthetic Landslide"
zorder = 2
alpha = 0.3
        
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

x = np.arange(len(time_str))
for i in range(0, 100):
    ls = ds["ls"][:, i].values
    ls = unit_converter(input=ls, catchment_area=4.83, method="area-aggregated")
    cum_ls = np.cumsum(ls)
    
    
    ax0.plot(x, ls, lw=1.5, alpha=0.05, color=color, zorder=zorder)
    ax1.plot(x, cum_ls, lw=1.5, alpha=alpha, color=color, zorder=zorder)

ax0.set_title(label="(a)", loc="left", fontweight='bold', fontsize=7)
ax0.set_xticks(ticks_location, labels=ticks_label)
ax0.set_xticks(minor_ticks, minor=True)
ax0.set_xlim(x[0], x[-1]+1)
ax0.set_yscale("log")
ax0.set_ylim(1e0, 2e6)
ax0.set_ylabel(r"Landslide Volume $v$ [$\mathrm{m}^3$]", fontweight='bold')
ax0.set_xlabel("", fontweight='bold')
ax0.grid(axis='both', which='major', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)


ax1.set_title(label="(b)", loc="left", fontweight='bold', fontsize=7)
ax1.set_xticks(ticks_location, labels=ticks_label)
ax1.set_xticks(minor_ticks, minor=True)
ax1.set_yscale("log")
ax1.set_ylim(1e0, 1e8)
ax1.set_xlim(x[0], x[-1]+1)
ax1.set_ylabel(r"Cumulative Landslide Volume $v$ [$\mathrm{m}^3$]", fontweight='bold')
ax1.set_xlabel("UTC+0 Time [Resolution = 10 minutes]", fontweight='bold')
ax1.grid(axis='both', which='major', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
# handles, labels = ax1.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax1.legend(by_label.values(), by_label.keys(), loc="lower right")


# complementary cumulative distribution function
png_path = Path(current_dir) / f"landslide_t.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
