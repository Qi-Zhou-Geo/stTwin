#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-24T18:06:28
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

from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm


from obspy import UTCDateTime

from scipy.stats import spearmanr

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


# region <copied from s02_plot_precp_vs_yield.py>


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

# endregion




cycle_period = np.array(df_all["cycle_period"])
cycle_period_marker = {}
marker_scaler = 20
for idx, cp in enumerate(np.unique(cycle_period)):
        cycle_period_marker[int(cp)] = (idx + 1) * marker_scaler


colors = plt.cm.coolwarm(np.linspace(0, 1, 9)) # type: ignore
colors = [mcolors.to_hex(c) for c in colors]
storm_onset_month = np.unique(np.array(df_all["storm_onset_month"])).tolist()
storm_onset_month_color = dict(zip(storm_onset_month, colors))



fig = plt.figure(figsize=(6, 6.5))
gs = gridspec.GridSpec(3, 2, height_ratios=[10, 10, 1])


ax = plt.subplot(gs[0])

v = "sed_out_50"
idx = np.where(stat_col_name == v)[0][0]
obs = benchmark_data[idx]
y = df_all[v] / obs
x = df_all['precp_ratio']

rho, p_value = spearmanr(x, y)
ax.set_title(label=f"(a) " + r"$\rho$" + f"={rho:.3f}, p-value={p_value:.3f}", loc="left", fontsize=7, fontweight='bold')

for idx in np.arange(len(df_all)):
        cp = df_all["cycle_period"][idx]
        t0 = df_all["storm_onset_month"][idx]
        ax.scatter(x=x[idx], y=y[idx], 
                   edgecolors="black", facecolor=storm_onset_month_color.get(t0), 
                   alpha=0.5, zorder=15 - cp, 
                   s=cycle_period_marker.get(cp))

ax.set_xlabel("Precipitation Ratio "  + r"$\frac{\Sigma P_\text{s2d}(t)}{\Sigma P_\text{mve}(t)}$", fontweight="bold")
ax.set_ylabel("Sediment Yield Ratio " + r"$\frac{\Sigma Y_\text{s2d}(t)}{\Sigma Y_\text{mve}(t)}$", fontweight="bold")
ax.set_ylim(0.10, 0.35)
ax.set_xlim(0, 2)
ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)




ax = plt.subplot(gs[1])

v = "sed_out_50"
idx = np.where(stat_col_name == v)[0][0]
obs = benchmark_data[idx]
y = df_all[v] / obs

v = 'ls_input_sum_50'
idx = np.where(stat_col_name == v)[0][0]
obs = benchmark_data[idx]
x = df_all[v] / obs

rho, p_value = spearmanr(x, y)
ax.set_title(label=f"(b) " + r"$\rho$" + f"={rho:.3f}, p-value={p_value:.3f}", loc="left", fontsize=7, fontweight='bold')


for idx in np.arange(len(df_all)):
        cp = df_all["cycle_period"][idx]
        t0 = df_all["storm_onset_month"][idx]
        ax.scatter(x=x[idx], y=y[idx], 
                   edgecolors="black", facecolor=storm_onset_month_color.get(t0), 
                   alpha=0.5, zorder=15 - cp, 
                   s=cycle_period_marker.get(cp))

ax.set_xlabel(r"Sediment Source Ratio " + r"$\frac{\Sigma S_\text{s2d}(t)}{\Sigma S_\text{mve}(t)}$", fontweight="bold")
ax.set_ylabel("Sediment Yield Ratio " + r"$\frac{\Sigma Y_\text{s2d}(t)}{\Sigma Y_\text{mve}(t)}$", fontweight="bold")
ax.set_ylim(0.10, 0.35)
ax.set_xlim(0.05, 0.20)
ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)





ax = plt.subplot(gs[2])

v = 'what_if_sed_out2in_ratio'
y = df_all[v]
x = df_all['precp_ratio']

rho, p_value = spearmanr(x, y)
ax.set_title(label=f"(c) " + r"$\rho$" + f"={rho:.3f}, p-value={p_value:.3f}", loc="left", fontsize=7, fontweight='bold')

for idx in np.arange(len(df_all)):
        cp = df_all["cycle_period"][idx]
        t0 = df_all["storm_onset_month"][idx]
        ax.scatter(x=x[idx], y=y[idx], 
                   edgecolors="black", facecolor=storm_onset_month_color.get(t0), 
                   alpha=0.5, zorder=15 - cp, 
                   s=cycle_period_marker.get(cp))

sdr = 1.065
ax.axhline(y=sdr, color="black", ls="--", lw=0.7, zorder=2, label=f"SDE"+ r"$_{\mathrm{mve}}$" + f"={sdr}")
ax.legend(loc="lower right", fontsize=6)
ax.set_xlabel("Precipitation Ratio "  + r"$\frac{\Sigma P_\text{s2d}(t)}{\Sigma P_\text{mve}(t)}$", fontweight="bold")
ax.set_ylabel("Sediment Delivery Efficiency " + r"$\frac{\Sigma Y_\text{s2d}(t)}{\Sigma S_\text{s2d}(t)}$", fontweight="bold")
ax.set_ylim(0.5, 4)
ax.set_xlim(0, 2)
ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)




ax = plt.subplot(gs[3])

v = 'what_if_sed_out2in_ratio'
y = df_all[v]

v = 'ls_input_sum_50'
idx = np.where(stat_col_name == v)[0][0]
obs = benchmark_data[idx]
x = df_all[v] / obs

rho, p_value = spearmanr(x, y)
ax.set_title(label=f"(d) " + r"$\rho$" + f"={rho:.3f}, p-value={p_value:.3f}", loc="left", fontsize=7, fontweight='bold')


for idx in np.arange(len(df_all)):
        cp = df_all["cycle_period"][idx]
        t0 = df_all["storm_onset_month"][idx]
        ax.scatter(x=x[idx], y=y[idx], 
                   edgecolors="black", facecolor=storm_onset_month_color.get(t0), 
                   alpha=0.5, zorder=15 - cp, 
                   s=cycle_period_marker.get(cp))
sdr = 1.065
ax.axhline(y=sdr, color="black", ls="--", lw=0.7, zorder=2, label=f"Observed SDR={sdr}")
ax.set_xlabel(r"Sediment Source Ratio " + r"$\frac{\Sigma S_\text{s2d}(t)}{\Sigma S_\text{mve}(t)}$", fontweight="bold")
ax.set_ylabel("Sediment Delivery Efficiency " + r"$\frac{\Sigma Y_\text{s2d}(t)}{\Sigma S_\text{s2d}(t)}$", fontweight="bold")
ax.set_xlim(0.05, 0.20)
ax.set_ylim(0.5, 4)
ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)


# legend
ax_marker = plt.subplot(gs[2, 0])
ax_marker.axis("off")

size_handles = []
for cp in sorted(np.unique(cycle_period)):
    h = ax_marker.scatter([], [],
        s=cycle_period_marker[int(cp)],
        edgecolor="black", facecolor="none", alpha=1, label=f"{cp:g}",
    )
    size_handles.append(h)

ax_marker.legend(
    handles=size_handles,
    loc="center",
    ncol=7,
    fontsize=6,
    title_fontsize=7,
    frameon=False,
    columnspacing=0.8,
    handletextpad=0.4,
    labelspacing=0.4,
    scatterpoints=1,
)

ax_marker.text(x=0.5, y=1.5, s="Cycle Length",
    transform=ax_marker.transAxes,
    ha="center",
    va="center",
    fontsize=6,
    fontweight="bold",
)


# colorbar
ax_cbar = plt.subplot(gs[2, 1])

months = sorted(np.unique(df_all["storm_onset_month"]))
cmap = ListedColormap([storm_onset_month_color[m] for m in months])

bounds = np.arange(len(months) + 1)
norm = BoundaryNorm(bounds, cmap.N)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal", ticks=np.arange(len(months)) + 0.5)

cbar.ax.set_xticklabels([f"{m:g}" for m in months])
cbar.ax.tick_params(labelsize=6)
ax_cbar.text(x=0.5, y=1.5,
    s="First Storm Onset Month",
    transform=ax_cbar.transAxes,
    ha="center",
    va="center",
    fontsize=6,
    fontweight="bold",
)




png_path = Path(current_dir) / f"plots/SDE_correlation.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
