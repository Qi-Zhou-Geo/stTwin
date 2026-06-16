#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-15T10:13:39
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
from func.SedCas.mass_balance_checker import mass_balance_checker

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})


# region < load scenario from MAP theta >
statistic_ratio = Path(project_root) / "pipeline/run_whatif/scenario_bound.txt"
df = pd.read_csv(statistic_ratio, header=0)

cycle_period = df["cycle_period"].values
storm2drought_ratio = df["storm2drought_ratio"].values
storm_onset_month = df["storm_onset_month"].values
# endregion


# region < select cycle_period = 30 for all Rs2d and t0 >
model_version = "v0dot4"
value_min, value_max, value_num = 30, 180, 6
cycle_period_l = np.linspace(value_min, value_max, value_num)
cycle_period_l = [30]
cp = cycle_period_l[0]
idx = np.where(cycle_period==cp)[0]
# endregion


benchmark_data_dir = Path(current_dir) / f"cache/monitor_MAP_real.npz"
data = np.load(benchmark_data_dir, allow_pickle=True)
benchmark_data = data["stat_values"]
stat_col_name = data["stat_col_name"]


df_sub = df.iloc[idx, :].copy()
df_sub = df_sub.reset_index(drop=True)
temp = []
stat_col_name = None
for scenario_idx in range(len(df_sub)):

        base_values = df_sub.iloc[scenario_idx, :].values
        i, cp, Rs2d, t0, d = base_values

        whatif_type = f"CP={cp}_R={Rs2d}_M={t0}_D={d}"
        data_dir = Path(current_dir) / f"cache/whatif_MAP_{whatif_type}.npz"
        data = np.load(data_dir, allow_pickle=True)
        stat_values = data["stat_values"]
        stat_col_name = data["stat_col_name"]
        temp.append(stat_values)
        
df_temp = pd.DataFrame(data=temp, columns=stat_col_name)
df_all = pd.concat([df_sub, df_temp], axis=1)


df_sub = df.iloc[idx, :].copy()
df_sub = df_sub.reset_index(drop=True)
temp = []
for scenario_idx in range(len(df_sub)):

        base_values = df_sub.iloc[scenario_idx, :].values
        i, cp, Rs2d, t0, d = base_values

        whatif_type = f"CP={cp}_R={Rs2d}_M={t0}_D={d}"
        data_dir = Path(current_dir) / f"cache/monitor_precp_{whatif_type}.npz"
        data = np.load(data_dir, allow_pickle=True)
        ratio = data["ratio"]
        precp_real = data["precp_real"]
        precp_whatif = data["precp_whatif"]
        
        if Rs2d in [0.05, 0.1]:
                print(f"{Rs2d}, {t0}, {precp_real}, {precp_whatif}")
        
        temp.append(ratio)
        
df_temp = pd.DataFrame(data=temp, columns=["precp_ratio"])
df_all = pd.concat([df_all, df_temp], axis=1)








fig = plt.figure(figsize=(3, 6))
gs = gridspec.GridSpec(2, 1)

ax_heat = plt.subplot(gs[0])
ax_heat.set_title(label=f"(a) Cumulative Precipitation", loc="left", fontsize=7, fontweight='bold')

x = "storm_onset_month"
y = "storm2drought_ratio"
v = "precp_ratio"
df_heat = df_all.pivot(index=y, columns=x, values=v)

print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")
heatmap = sns.heatmap(df_heat, cmap="Blues", cbar=True, ax=ax_heat, square=False, 
                        vmin=0.5, vmax=4, 
                        annot=True, fmt=".2f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "horizontal", # "vertical", #
                                "shrink": 1})
cbar = heatmap.collections[0].colorbar
cbar.set_label("Ratio of What-if to Observed", fontsize=6) # type: ignore
ax_heat.invert_yaxis()
ax_heat.set_ylabel("Storm to Drought Ratio", fontweight='bold')
# ax_heat.set_xlabel("Storm Onset Month", fontweight='bold')
ax_heat.set_xlabel("", fontweight='bold')


ax_heat = plt.subplot(gs[1])
ax_heat.set_title(label=f"(b) Channel Storage Loss", loc="left", fontsize=7, fontweight='bold')

x = "storm_onset_month"
y = "storm2drought_ratio"
v = "ch_loss_50"
df_heat = df_all.pivot(index=y, columns=x, values=v) / benchmark_data[np.where(stat_col_name==v)[0][0]]

print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")
heatmap = sns.heatmap(df_heat, cmap="Oranges", cbar=True, ax=ax_heat, square=False, 
                        vmin=1.5, vmax=3, 
                        annot=True, fmt=".2f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "horizontal", # "vertical", #
                                "shrink": 1})
cbar = heatmap.collections[0].colorbar
cbar.set_label("Ratio of What-if to Observed", fontsize=6) # type: ignore

ax_heat.invert_yaxis()
ax_heat.set_ylabel("Storm to Drought", fontweight='bold')
ax_heat.set_xlabel("Storm Onset Month", fontweight='bold')




png_path = Path(current_dir) / "plots" / f"cycle_period_{cp}_precp_yield.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)







fig = plt.figure(figsize=(3, 7))
gs = gridspec.GridSpec(3, 1)

ax_heat = plt.subplot(gs[0])
ax_heat.set_title(label=f"(a) Q05", loc="left", fontsize=7, fontweight='bold')

x = "storm_onset_month"
y = "storm2drought_ratio"
v = "sed_out_05"
df_heat = df_all.pivot(index=y, columns=x, values=v) / benchmark_data[np.where(stat_col_name==v)[0][0]]

print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")
heatmap = sns.heatmap(df_heat, cmap="coolwarm", cbar=True, ax=ax_heat, square=False, 
                        vmin=0.99, vmax=1.09, 
                        annot=True, fmt=".3f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "vertical", # "horizontal", #
                                "shrink": 1})
cbar = heatmap.collections[0].colorbar
cbar.set_label("Sediment Yield Ratio\nWhat-if / MVE Observed (2023–2025)", fontsize=5) # type: ignore
ax_heat.invert_yaxis()
ax_heat.set_ylabel("Storm to Drought Ratio", fontweight='bold')
ax_heat.set_xlabel("", fontweight='bold')



ax_heat = plt.subplot(gs[1])
ax_heat.set_title(label=f"(a) Q50", loc="left", fontsize=7, fontweight='bold')

x = "storm_onset_month"
y = "storm2drought_ratio"
v = "sed_out_50"
df_heat = df_all.pivot(index=y, columns=x, values=v) / benchmark_data[np.where(stat_col_name==v)[0][0]]

print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")
heatmap = sns.heatmap(df_heat, cmap="coolwarm", cbar=True, ax=ax_heat, square=False, 
                        vmin=0.99, vmax=1.09, 
                        annot=True, fmt=".3f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "vertical", # "horizontal", #
                                "shrink": 1})
cbar = heatmap.collections[0].colorbar
cbar.set_label("Sediment Yield Ratio\nWhat-if / MVE Observed (2023–2025)", fontsize=5) # type: ignore
ax_heat.invert_yaxis()
ax_heat.set_ylabel("Storm to Drought Ratio", fontweight='bold')
ax_heat.set_xlabel("", fontweight='bold')



ax_heat = plt.subplot(gs[2])
ax_heat.set_title(label=f"(c) Q95", loc="left", fontsize=7, fontweight='bold')

x = "storm_onset_month"
y = "storm2drought_ratio"
v = "sed_out_95"
df_heat = df_all.pivot(index=y, columns=x, values=v) / benchmark_data[np.where(stat_col_name==v)[0][0]]

print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")
heatmap = sns.heatmap(df_heat, cmap="coolwarm", cbar=True, ax=ax_heat, square=False, 
                      vmin=0.99, vmax=1.09, 
                        annot=True, fmt=".3f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "vertical", # "horizontal", #
                                "shrink": 1})
cbar = heatmap.collections[0].colorbar
cbar.set_label("Sediment Yield Ratio\nWhat-if / MVE Observed (2023–2025)", fontsize=5) # type: ignore
ax_heat.invert_yaxis()
ax_heat.set_ylabel("Storm to Drought Ratio", fontweight='bold')
ax_heat.set_xlabel("Storm Onset Month", fontweight='bold')



png_path = Path(current_dir) / "plots" / f"cycle_period_{cp}_precp_yield_sed.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
