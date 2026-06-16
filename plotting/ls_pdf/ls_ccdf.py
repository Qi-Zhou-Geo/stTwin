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
ls_volume = ds["ls"].values
ls_volume = unit_converter(input=ls_volume, catchment_area=4.83, method="area-aggregated")




fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

color = "black"
label = "Synthetic Landslide"
zorder = 2
alpha = 0.3

for scenario_idx in range(0, 100):
    
    ls_idx = ls_volume[:, scenario_idx]
    sorted_v = np.sort(ls_idx)
    n = len(sorted_v)
    # P(V > sorted_v[i]) = (n - rank) / n, where rank is 1-indexed
    ccdf = (n - np.arange(1, n + 1)) / n  # equivalent to 1 - rank/n
    
    ax.plot(sorted_v, ccdf, alpha=alpha, color=color, zorder=zorder)

ax.axvline(x=2, color="C0", ls="--", lw=1, zorder=1, label=r"Min volume = $2 \times 10^0$")
ax.axvline(x=1000000, color="C1", ls="--", lw=1, zorder=1, label=r"Max volume = $1 \times 10^6$")
ax.legend(loc="upper center", fontsize=6)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="lower left")



ax.set_xlim(1e0, 2e6)
ax.set_ylim(1e-8, 1e-3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"Probability $P \,\, (\, V > v \,)$", fontweight='bold')
ax.set_xlabel(r"Landslide Volume $v$ [$\mathrm{m}^3$]", fontweight='bold')
ax.grid(axis='both', which='major', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)


# complementary cumulative distribution function
png_path = Path(current_dir) / f"landslide_ccdf.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
