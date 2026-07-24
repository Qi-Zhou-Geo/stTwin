#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-22T15:35:14
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
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

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


# all scenarios, MAP + what-if
scenario_name = "run_whatif_Rs2d=0.05"
statistic_ratio = Path(project_root) / f"pipeline/{scenario_name}/scenario_bound.txt"
df0 = pd.read_csv(statistic_ratio, header=0)

# benchmark results, MAP + MVE obs.
benchmark_data_dir = Path(current_dir) / f"cache/monitor_MAP_real.npz"
data = np.load(benchmark_data_dir, allow_pickle=True)
benchmark_data = data["stat_values"]
stat_col_name = data["stat_col_name"]

# add sediment 
temp = []
stat_col_name = None
for scenario_idx in range(len(df0)):

        base_values = df0.iloc[scenario_idx, :].values
        i, cp, Rs2d, t0, d = base_values

        whatif_type = f"CP={int(cp)}_R={Rs2d:.3f}_M={int(t0)}_D={int(d)}"
        data_dir = Path(current_dir) / f"cache/whatif_MAP_{whatif_type}.npz"
        data = np.load(data_dir, allow_pickle=True)
        stat_values = data["stat_values"]
        stat_col_name = data["stat_col_name"]
        temp.append(stat_values)
        
df_temp = pd.DataFrame(data=temp, columns=stat_col_name)
df_all = pd.concat([df0, df_temp], axis=1)


# add precp
temp = []
for scenario_idx in range(len(df0)):

        base_values = df0.iloc[scenario_idx, :].values
        i, cp, Rs2d, t0, d = base_values

        whatif_type = f"CP={int(cp)}_R={Rs2d:.3f}_M={int(t0)}_D={int(d)}"
        data_dir = Path(current_dir) / f"cache/monitor_precp_{whatif_type}.npz"
        data = np.load(data_dir, allow_pickle=True)
        ratio = data["ratio"]
        precp_real = data["precp_real"]
        precp_whatif = data["precp_whatif"]
        
        temp.append(ratio)
        
df_temp = pd.DataFrame(data=temp, columns=["precp_ratio"])
df_all = pd.concat([df_all, df_temp], axis=1)


# add SDR
# sediments yield to ls input ratio for different scenarios
what_if_sed_out2in_ratio = df_all['sed_out_50'] / df_all['ls_input_sum_50']
df_all['what_if_sed_out2in_ratio'] = np.array(what_if_sed_out2in_ratio)

# real SDR
# obs by mve
idx1 = np.where(stat_col_name == 'sed_out_50')[0][0]
idx2 = np.where(stat_col_name == 'ls_input_sum_50')[0][0]
obs_sed_out2in_ratio = benchmark_data[idx1] / benchmark_data[idx2]




# region < 2 * 2 >
vmin = 0
vmax = 2
norm_color = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 2)



ax_heat = plt.subplot(gs[0])
ax_heat.set_title(label=f"(a) Precipitation", loc="left", fontsize=7, fontweight='bold')

# region
x = "storm_onset_month"
y = "cycle_period"
v = "precp_ratio"
df_heat = df_all.pivot(index=y, columns=x, values=v)

print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")
heatmap = sns.heatmap(df_heat, cmap="coolwarm", cbar=True, ax=ax_heat, square=False, 
                        vmin=vmin, vmax=vmax, norm=norm_color,
                        annot=True, fmt=".2f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "horizontal", # "vertical", #
                                "shrink": 1, "aspect": 30,
                                "extend": "both"})
cbar = heatmap.collections[0].colorbar                                                                                          
cbar.set_label("Ratio of " + r"$\frac{\Sigma P_\text{s2d}(t)}{\Sigma P_\text{mve}(t)}$", fontsize=6, labelpad=-8, fontweight='bold') # type: ignore
cbar.set_ticks([vmin, vmax]) # type: ignore
cbar.set_ticklabels([f"Low ({vmin:.1f})", f"High ({vmax:.1f})"], fontsize=6) # type: ignore

ax_heat.invert_yaxis()
ax_heat.set_ylabel("Cycle Length " + r"$1/f$" + " [day]", fontweight='bold')
ax_heat.set_xlabel("First Storm Onset Month", fontweight='bold')
# endregion





ax_heat = plt.subplot(gs[1])
ax_heat.set_title(label=f"(b) Sediment Source", loc="left", fontsize=7, fontweight='bold')

# region
vmin=0.05
vmax=0.20
x = "storm_onset_month"
y = "cycle_period"
v = 'ls_input_sum_50'

idx = np.where(stat_col_name == v)[0][0]
obs = benchmark_data[idx]

df_heat = df_all.pivot(index=y, columns=x, values=v)  / obs
print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")

heatmap = sns.heatmap(df_heat, cmap="Greens", cbar=True, ax=ax_heat, square=False, 
                        vmin=vmin, vmax=vmax, 
                        annot=True, fmt=".2f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "horizontal", #"vertical", # 
                                "shrink": 1, "aspect": 30, "extend": 'max'})
cbar = heatmap.collections[0].colorbar
cbar.set_label("Ratio of " + r"$\frac{\Sigma S_\text{s2d}(t)}{\Sigma S_\text{mve}(t)}$", fontsize=6, labelpad=-8, fontweight='bold') # type: ignore
cbar.set_ticks([vmin, vmax]) # type: ignore
cbar.set_ticklabels([f"Low ({vmin:.2f})", f"High ({vmax:.2f})"], fontsize=6) # type: ignore

ax_heat.invert_yaxis()
ax_heat.set_ylabel("", fontweight='bold')
ax_heat.set_xlabel("First Storm Onset Month", fontweight='bold')
# endregion




ax_heat = plt.subplot(gs[2])
ax_heat.set_title(label=f"(c) Sediment Yield", loc="left", fontsize=7, fontweight='bold')

# region
vmin=0.10
vmax=0.35
x = "storm_onset_month"
y = "cycle_period"
v = 'sed_out_50'

idx = np.where(stat_col_name == v)[0][0]
obs = benchmark_data[idx]

df_heat = df_all.pivot(index=y, columns=x, values=v)  / obs
print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")

heatmap = sns.heatmap(df_heat, cmap="Reds", cbar=True, ax=ax_heat, square=False, 
                        vmin=vmin, vmax=vmax, 
                        annot=True, fmt=".2f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "horizontal", #"vertical", # 
                                "shrink": 1, "aspect": 30, "extend": "max"})
cbar = heatmap.collections[0].colorbar
cbar.set_label("Ratio of " + r"$\frac{\Sigma Y_\text{s2d}(t)}{\Sigma Y_\text{mve}(t)}$", fontsize=6, labelpad=-8, fontweight='bold') # type: ignore
cbar.set_ticks([vmin, vmax]) # type: ignore
cbar.set_ticklabels([f"Low ({vmin:.2f})", f"High ({vmax:.2f})"], fontsize=6) # type: ignore

ax_heat.invert_yaxis()
ax_heat.set_ylabel("Cycle Length " + r"$1/f$" + " [day]", fontweight='bold')
ax_heat.set_xlabel("First Storm Onset Month", fontweight='bold')
# endregion





ax_heat = plt.subplot(gs[3])
ax_heat.set_title(label=f"(d) Sediment Delivery Efficiency", loc="left", fontsize=7, fontweight='bold')

# region
vmin=0.5
vmax=4.0
vcenter = obs_sed_out2in_ratio
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

x = "storm_onset_month"
y = "cycle_period"
v = 'what_if_sed_out2in_ratio'

df_heat = df_all.pivot(index=y, columns=x, values=v) #- obs_sed_out2in_ratio
print(f"{v}, {np.max(df_heat):.3f}, {np.min(df_heat):.3f}")

heatmap = sns.heatmap(df_heat, cmap="coolwarm", cbar=True, ax=ax_heat, square=False, 
                        vmin=vmin, vmax=vmax,
                        norm=norm,
                        annot=True, fmt=".2f", annot_kws={"size": 5},
                        cbar_kws={"orientation": "horizontal", #"vertical", # 
                                "shrink": 1, "aspect": 30, "extend": "both"})
cbar = heatmap.collections[0].colorbar
cbar.set_label(rf"Ratio of " + r"$\frac{\Sigma Y_\text{s2d}(t)}{\Sigma S_\text{s2d}(t)}$", fontsize=6, labelpad=-8, fontweight='bold') # type: ignore
cbar.set_ticks([vmin, vmax]) # type: ignore
cbar.set_ticklabels([f"Low ({vmin:.1f})", f"High ({vmax:.1f})"], fontsize=6) # type: ignore

ax_heat.invert_yaxis()
ax_heat.set_ylabel("", fontweight='bold')
ax_heat.set_xlabel("First Storm Onset Month", fontweight='bold')
# endregion



png_path = Path(current_dir) / f"plots/Rs2d_0.05_precp_yield.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
# endregion
