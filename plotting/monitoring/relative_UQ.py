#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-01T11:38:02
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
                     'axes.formatter.limits': (-3, 4),
                     'axes.formatter.use_mathtext': True})


def plot_region(ax):
    ax.grid(axis="x", color="grey", which="both", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax.grid(axis="y", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax.axvspan(np.where(time_str == "2008-09-01T00:00:00")[0][0], 
               np.where(time_str == "2012-06-08T17:00:00")[0][0], 
               color="C3", alpha=0.2, zorder=1)#, label="2008-09-01 to 2012-06-08 17:00")
    ax.axvspan(np.where(time_str == "2023-01-01T00:00:00")[0][0], 
               np.where(time_str == "2025-12-31T23:50:00")[0][0], 
                color="C1", alpha=0.2, zorder=1)#, label="2023-01-01 to 2025-12-31 23:50")


monitor_MAP = Path(project_root) / f'pipeline/run_2004_2025_posterior/v0dot4/theta_001/sed_container.nc'
ds = xr.load_dataset(monitor_MAP)
time_str = ds.coords["time_str"].values
x = np.arange(len(time_str))


# region < ticks >
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
# endregion


# region < label >
subplot_idx = ["(a)", "(b)", "(c)"]
y_label = ["Hillslope Storage", 
           "Channel Storage", 
           "Sediment Yield"]
y_zoom = np.array([1e5, 1e6, 1e4])
y_min = np.array([5.764e5, 0,   0]) / y_zoom
y_max = np.array([5.772e5, 6e6, 6e4]) / y_zoom
y_scale = ["linear", "linear", "linear"]
# endregion



cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in [0, 0.5, 0.99]]  # evenly spaced: start, middle, end
uq_list = ["fix_MAP__use_ls_1-100", "fix_ls__use_MCMC_1-21", "use_ls_1-100__use_MCMC_1-21"]
uq_list = ["fix_MAP__use_ls_1-100", "fix_ls__use_MCMC_1-21", "use_ls_1-100__use_MCMC_1-21"]


fig = plt.figure(figsize=(6.5, 6))
gs = gridspec.GridSpec(3, 1)

for idx, key in enumerate(["hillslope_storage", "channel_storage", "sed_transport_real"]):


    ax = plt.subplot(gs[idx])
    ax.set_title(label=f"{subplot_idx[idx]}", loc="left", fontsize=7, fontweight='bold')
    
    for idy, cache_name in enumerate(uq_list):
        
        if cache_name == "fix_ls__use_MCMC_1-21":
            cache_path = Path(current_dir) / f"cache/{cache_name}_uq_in_{key}_0.npz"
        else:
            cache_path = Path(current_dir) / f"cache/{cache_name}_uq_in_{key}.npz"
        
        cache_data = np.load(cache_path, allow_pickle=True)
        q05, q50, q95 = cache_data["q05"], cache_data["q50"], cache_data["q95"]

        y = (q95 - q05) / (q95 + q05) # normalize, return 0-1
        ax.plot(x, y, color=colors[idy], zorder=5, alpha=0.75, label=cache_name)

    ax.set_xlim(x[0], x[-1]+1)
    
    if idx == 0:
        ax.set_ylim(0, 0.1)
    else:
        ax.set_ylim(0, 1)

    ax.set_xticks(ticks_location, labels=ticks_label)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_ylabel(y_label[idx], fontweight='bold')
    ax.set_xlabel("", fontweight='bold')
    plot_region(ax)




ax = plt.subplot(gs[0])
ax.legend(loc="upper left", fontsize=6, ncol=4)

ax = plt.subplot(gs[2])
ax.set_xlabel("UTC+0 Time", fontweight='bold') #  [Resolution = 10 minutes]


plt.tight_layout()
png_path = Path(current_dir) / f"compare_UQ.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
