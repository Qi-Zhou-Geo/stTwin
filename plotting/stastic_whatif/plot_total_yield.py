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

#region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

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

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})


model_version = "v0dot4"
npy_folder = f"/home/qizhou/3paper/stTwin/plotting/stastic_whatif/{model_version}"


npy_real = np.load(f"{npy_folder}/real_sed_transport_real_Q50.npz", allow_pickle=True)
total_sed_yield_real = npy_real["total_sed_yield"]




scenario_bound = Path(project_root) / "pipeline" / "what_if" / "scenario_bound.txt"
df = pd.read_csv(scenario_bound, header=0)

ratio_list = []
for idx in range(len(df)):
    
    temp = df.iloc[idx, :].values
    i, cp, r, m, d = temp
    
    whatif_type = f"CP={cp}_R={r}_M={m}_D={d}"
    npy_whatif = np.load(f"{npy_folder}/what-if_sed_transport_real_Q50_{whatif_type}.npz", allow_pickle=True)
    total_sed_yield_whatif = npy_whatif["total_sed_yield"]
    
    ratio = total_sed_yield_whatif / total_sed_yield_real
    ratio_list.append(ratio)

df_ratio = df.copy()
df_ratio["ratio"] = np.array(ratio_list)
df_ratio.to_csv(f"./df_ratio.txt", index=False)
print(df_ratio.shape)



df_raw = pd.read_csv(f"./df_ratio.txt", header=0)
extreme = {"min": [15, 0.1, 2], "max":[180, 0.5, 10]}
status = "max"

# "cycle_period", "storm2drought_ratio", "storm_onset_month"
idx_map = {
    0: ("cycle_period", "storm2drought_ratio", ("storm_onset_month", extreme[status][2])),
    1: ("cycle_period", "storm_onset_month", ("storm2drought_ratio", extreme[status][1])),
    2: ("storm2drought_ratio", "storm_onset_month", ("cycle_period", extreme[status][0])),
}

variable_map = {"cycle_period": "Cycle Period [1/f]", 
                "storm2drought_ratio": "Storm to Drought Ratio [Rs2d]", 
                "storm_onset_month": "Storm Onset Month [t0]"}


for idx in range(3):

    fig = plt.figure(figsize=(6, 6))

    gs = gridspec.GridSpec(3, 2, 
                           width_ratios=[10, 1.5], height_ratios=[1.5, 10, 0.2], 
                           wspace=0.1, hspace=0.15)

    ax_bar_x = plt.subplot(gs[0, 0])
    ax_heat = plt.subplot(gs[1, 0])
    ax_cbar = plt.subplot(gs[2, 0])
    ax_bar_y = plt.subplot(gs[1, 1])

    x, y, (collapse_dim, collapse_var) = idx_map[idx]
    v = "ratio"
    
    df_sub = df_raw[df_raw[collapse_dim] == collapse_var]
    x_vals = df_sub[x].dropna()
    y_vals = df_sub[y].dropna()
    
    df_heat = df_sub.pivot(index=y, columns=x, values=v)
    heatmap = sns.heatmap(df_heat, cmap="inferno", cbar=False, ax=ax_heat, vmin=0.2, vmax=0.3)
    ax_heat.invert_yaxis()
    ax_heat.set_xlabel(variable_map[x], fontweight='bold')
    ax_heat.set_ylabel(variable_map[y], fontweight='bold')
    

    cbar = fig.colorbar(heatmap.collections[0], cax=ax_cbar, orientation="horizontal")
    cbar_label = f"Total Sediment Yield Ratio (What-If / MVE Observed Climate Forcing)\n{variable_map[collapse_dim]}={collapse_var}" # Total Sediment Yield Ratio (What-If / MVE Observed Climate Forcing) 
    cbar.set_label(cbar_label, fontsize=6, labelpad=0)  # labelpad pushes text away from bar
    pos = ax_cbar.get_position()
    ax_cbar.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height])


    bar_x = df_heat.columns.values
    bar_y = np.mean(df_heat.values, axis=0)
    ax_bar_x.bar(bar_x, bar_y, color="C0", alpha=0.7, width=0.9*(bar_x[1]-bar_x[0]))
    # ax_bar_x.set_xticks(bar_x, bar_x)
    ax_bar_x.set_xticks(bar_x)
    ax_bar_x.set_xticklabels([])
    ax_bar_x.grid(axis="y", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax_bar_x.set_xlim(bar_x[0] - 0.9*(bar_x[1]-bar_x[0]) / 2, bar_x[-1] + 0.9*(bar_x[1]-bar_x[0]) / 2)
    ax_bar_x.set_ylim(0, 0.3)
    ax_bar_x.set_ylabel("Mean Ratio", fontweight='bold')
    
    
    bar_x = df_heat.index.values
    bar_y = np.mean(df_heat.values, axis=1)
    ax_bar_y.barh(bar_x, bar_y, color="C0", alpha=0.7, height=0.9*(bar_x[1]-bar_x[0]), zorder=2)
    # ax_bar_y.set_yticks(bar_x, bar_x)
    ax_bar_y.set_yticks(bar_x)
    ax_bar_y.set_yticklabels([])
    ax_bar_y.grid(axis="x", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax_bar_y.set_ylim(bar_x[0] - 0.9*(bar_x[1]-bar_x[0]) / 2, bar_x[-1] + 0.9*(bar_x[1]-bar_x[0]) / 2)
    ax_bar_y.set_xlim(0, 0.3)
    ax_bar_y.set_xlabel("Mean Ratio", fontweight='bold')



    plt.savefig(f"{current_dir}/{status}_results_{idx}.png", dpi=600)
    plt.close(fig)

