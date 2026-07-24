#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-21T17:12:33
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
from matplotlib.colors import Normalize

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl


from obspy import UTCDateTime

from scipy.stats import spearmanr
from scipy.stats import linregress

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


scenario_name = "run_whatif_Rs2d=0.05"
statistic_ratio = Path(project_root) / f"pipeline/{scenario_name}/scenario_bound.txt"
df = pd.read_csv(statistic_ratio, header=0)
cycle_period = df["cycle_period"].values
storm2drought_ratio = df["storm2drought_ratio"].values
storm_onset_month = df["storm_onset_month"].values


# region < load all run_whatif_Rs2d=0.05>
scenario_name = "run_whatif_Rs2d=0.05"
model_version = "v0dot4"
select_t1, select_t2 = "2023-01-01T00:00:00", "2026-01-01T00:00:00"

cum_channel_storage = []
loss_channel_storage = []
cum_sediment_yield = []

for scenario_idx in range(len(df)):

    base_values = df.iloc[scenario_idx, :].values
    i, cp, Rs2d, t0, d = base_values

    whatif_type = f"CP={int(cp)}_R={Rs2d:.3f}_M={int(t0)}_D={int(d)}"
    data_dir = Path(project_root) / f"pipeline/{scenario_name}/{model_version}/{whatif_type}"

    ds_path = Path(data_dir) / f"theta_001/sed_container.nc" # 001 is MAP
    ds = xr.open_dataset(ds_path)
    mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
    ds = ds.isel(time=mask).load()
    
    for idx, col in enumerate(['channel_storage', 'sed_transport_real']):
        data = ds[col] # shape as (time, num_ls_stochastic)
        
        # axis=1 >> collapse along the 100-iteration dimension
        q05 = np.quantile(a=data, q=0.05, axis=1)
        q50 = np.quantile(a=data, q=0.50, axis=1)
        q95 = np.quantile(a=data, q=0.95, axis=1)

        if idx == 0:
            cum_channel_storage.append([np.sum(q05), np.sum(q50), np.sum(q95)])
            loss_channel_storage.append([
                q05[0] - q05[-1],
                q50[0] - q50[-1],
                q95[0] - q95[-1],
                ])
        else:
            cum_sediment_yield.append([np.sum(q05), np.sum(q50), np.sum(q95)])

    print(f"Done: {whatif_type}")


key = "cum_channel_storage"
cum_channel_storage = np.vstack(cum_channel_storage) # stack as row
df0 = pd.DataFrame(data=cum_channel_storage, columns=[f'{key}_q05', f'{key}_q50', f'{key}_q95'])


key = "cum_sed_transport_real"
cum_sediment_yield  = np.vstack(cum_sediment_yield ) # stack as row
df1 = pd.DataFrame(data=cum_sediment_yield, columns=[f'{key}_q05', f'{key}_q50', f'{key}_q95'])


key = "loss_channel_storage"
cum_channel_storage = np.vstack(loss_channel_storage) # stack as row
df2 = pd.DataFrame(data=cum_channel_storage, columns=[f'{key}_q05', f'{key}_q50', f'{key}_q95'])


# connect as column
df2 = pd.concat([df, df0, df1, df2], axis=1) 
df2.to_csv(f"./cache/df_s05_{scenario_name}.txt", index=False)
# endregion


# region < load all run_whatif_Rs2d=0.05_fix_ls>
scenario_name = "run_whatif_Rs2d=0.05_fix_ls"
model_version = "v0dot4"
select_t1, select_t2 = "2023-01-01T00:00:00", "2026-01-01T00:00:00"

cum_channel_storage = []
loss_channel_storage = []
cum_sediment_yield = []

for scenario_idx in range(len(df)):

    base_values = df.iloc[scenario_idx, :].values
    i, cp, Rs2d, t0, d = base_values

    whatif_type = f"CP={int(cp)}_R={Rs2d:.3f}_M={int(t0)}_D={int(d)}"
    data_dir = Path(project_root) / f"pipeline/{scenario_name}/{model_version}/{whatif_type}"

    ds_path = Path(data_dir) / f"theta_001/sed_container.nc" # 001 is MAP
    ds = xr.open_dataset(ds_path)
    mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
    ds = ds.isel(time=mask).load()
    
    for idx, col in enumerate(['channel_storage', 'sed_transport_real']):
        data = ds[col] # shape as (time, num_ls_stochastic)
        
        # axis=1 >> collapse along the 100-iteration dimension
        q05 = np.quantile(a=data, q=0.05, axis=1)
        q50 = np.quantile(a=data, q=0.50, axis=1)
        q95 = np.quantile(a=data, q=0.95, axis=1)

        if idx == 0:
            cum_channel_storage.append([np.sum(q05), np.sum(q50), np.sum(q95)])
            loss_channel_storage.append([
                q05[0] - q05[-1],
                q50[0] - q50[-1],
                q95[0] - q95[-1],
                ])
        else:
            cum_sediment_yield.append([np.sum(q05), np.sum(q50), np.sum(q95)])

    print(f"Done: {whatif_type}")


key = "cum_channel_storage"
cum_channel_storage = np.vstack(cum_channel_storage) # stack as row
df0 = pd.DataFrame(data=cum_channel_storage, columns=[f'{key}_q05', f'{key}_q50', f'{key}_q95'])


key = "cum_sed_transport_real"
cum_sediment_yield  = np.vstack(cum_sediment_yield ) # stack as row
df1 = pd.DataFrame(data=cum_sediment_yield, columns=[f'{key}_q05', f'{key}_q50', f'{key}_q95'])


key = "loss_channel_storage"
cum_channel_storage = np.vstack(loss_channel_storage) # stack as row
df2 = pd.DataFrame(data=cum_channel_storage, columns=[f'{key}_q05', f'{key}_q50', f'{key}_q95'])

# connect as column
df2 = pd.concat([df, df0, df1, df2], axis=1) 
df2.to_csv(f"./cache/df_s05_{scenario_name}.txt", index=False)
# endregion


# region < load obs >
model_version = "v0dot4"
select_t1, select_t2 = "2023-01-01T00:00:00", "2026-01-01T00:00:00"
ds_path = Path(project_root) / f"pipeline/run_2004_2025_posterior/{model_version}/theta_001/sed_container.nc" # 001 is MAP
ds = xr.load_dataset(ds_path)
mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
ds = ds.isel(time=mask)

col = 'channel_storage'
data = ds[col] # shape as (time, num_ls_stochastic)
# axis=1 >> collapse along the 100-iteration dimension
q05 = np.quantile(a=data, q=0.05, axis=1)
q50 = np.quantile(a=data, q=0.50, axis=1)
q95 = np.quantile(a=data, q=0.95, axis=1)
cum_channel_storage_bench = (np.sum(q05), np.sum(q50), np.sum(q95))


col = 'sed_transport_real'
data = ds[col] # shape as (time, num_ls_stochastic)
# axis=1 >> collapse along the 100-iteration dimension
q05 = np.quantile(a=data, q=0.05, axis=1)
q50 = np.quantile(a=data, q=0.50, axis=1)
q95 = np.quantile(a=data, q=0.95, axis=1)
cum_sed_transport_real_bench = (np.sum(q05), np.sum(q50), np.sum(q95))
# endregion



fig = plt.figure(figsize=(6, 3.5))
gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1])



for subplot_idx, scenario_name in enumerate(["run_whatif_Rs2d=0.05", "run_whatif_Rs2d=0.05_fix_ls"]):
    ax = plt.subplot(gs[subplot_idx])
    
    df_s05_path = Path(current_dir) / f"cache/df_s05_{scenario_name}.txt"
    df_s05 = pd.read_csv(df_s05_path, header=0)

    # convert to ratio
    key = "channel_storage"
    df_s05[f"cum_{key}_q05"] = df_s05[f"cum_{key}_q05"] / cum_channel_storage_bench[0]
    df_s05[f"cum_{key}_q50"] = df_s05[f"cum_{key}_q50"] / cum_channel_storage_bench[1]
    df_s05[f"cum_{key}_q95"] = df_s05[f"cum_{key}_q95"] / cum_channel_storage_bench[2]

    key = "loss_channel_storage"
    df_s05[f"cum_{key}_q05"] = df_s05[f"{key}_q05"] / cum_channel_storage_bench[0]
    df_s05[f"cum_{key}_q50"] = df_s05[f"{key}_q50"] / cum_channel_storage_bench[1]
    df_s05[f"cum_{key}_q95"] = df_s05[f"{key}_q95"] / cum_channel_storage_bench[2]


    key = "sed_transport_real"
    df_s05[f"cum_{key}_q05"] = df_s05[f"cum_{key}_q05"] / cum_sed_transport_real_bench[0]
    df_s05[f"cum_{key}_q50"] = df_s05[f"cum_{key}_q50"] / cum_sed_transport_real_bench[1]
    df_s05[f"cum_{key}_q95"] = df_s05[f"cum_{key}_q95"] / cum_sed_transport_real_bench[2]


    cp_unique = np.unique(np.array(cycle_period)).tolist()
    cp_marker = dict(zip(cp_unique, ["^", "s", "o", "v", "*", "P", "+"]))
    cp_marker = dict(zip(cp_unique, np.arange(1, len(cp_marker)+1, 1)))


    t0_unique = np.unique(np.array(storm_onset_month)).tolist()
    cmap = plt.get_cmap('coolwarm')
    colors = [f"C{i}" for i in range(len(t0_unique))]
    colors = [cmap(i) for i in np.linspace(0, 0.99, len(t0_unique))]
    t0_colors = dict(zip(t0_unique, colors))


    selected_idx = np.where(df_s05["cycle_period"] == 30.0)[0]
    df_s05 = df_s05.iloc[selected_idx, :]
    df_s05 = df_s05.reset_index(drop=True)

    x_list = []
    y_list = []
    for scenario_idx in range(len(df_s05)):

        base_values = df_s05.iloc[scenario_idx, :5].values
        i, cp, Rs2d, t0, d = base_values
        
        key = "sed_transport_real"
        x_q05 = df_s05[f"cum_{key}_q05"][scenario_idx]
        x_q50 = df_s05[f"cum_{key}_q50"][scenario_idx]
        x_q95 = df_s05[f"cum_{key}_q95"][scenario_idx]
        xerr = [[x_q50 - x_q05], [x_q95 - x_q50]]
        
        key = "channel_storage"
        y_q05 = df_s05[f"cum_{key}_q05"][scenario_idx]
        y_q50 = df_s05[f"cum_{key}_q50"][scenario_idx]
        y_q95 = df_s05[f"cum_{key}_q95"][scenario_idx]
        yerr = [[y_q50 - y_q05], [y_q95 - y_q50]]
        
        ax.scatter(x=x_q50, y=y_q50, 
                   color=t0_colors.get(int(t0)),
                   s=cp_marker.get(cp) * 20,  # type: ignore
                   alpha=0.75, zorder=5)

        x_list.append(x_q50)
        y_list.append(y_q50)

    x = np.array(x_list)
    y = np.array(y_list)

    result = linregress(x, y)
    slope = result.slope
    intercept = result.intercept
    r2 = result.rvalue**2

    print(f"Slope     = {slope:.4f}")
    print(f"Intercept = {intercept:.4f}")
    print(f"R²        = {r2:.4f}")
    
    rho, p_value = spearmanr(x, y)
    print(f"spearmanr: rho={rho}, p_value={p_value}")

ax = plt.subplot(gs[0])
ax.set_title("(a)", fontweight='bold', fontsize=7, loc='left')
ax.set_xlabel("Cumulative Sediment Yied\n[Ratio of What-if to Observed]", fontweight="bold")
ax.set_ylabel("Cumulative Channel Storage\n[Ratio of What-if to Observed]", fontweight="bold")
# ax.set_xlim(0.16, 0.22)
# ax.set_ylim(0, 0.35)
ax.grid(axis='both', which="major", color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

ax = plt.subplot(gs[1])
ax.set_title("(b)", fontweight='bold', fontsize=7, loc='left')
ax.set_xlabel("Cumulative Sediment Yied\n[Ratio of What-if to Observed]", fontweight="bold")
ax.set_ylabel("", fontweight="bold")
# ax.set_xlim(0.8, 1.2)
# ax.set_ylim(0, 1.6)
ax.grid(axis='both', which="major", color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)



# ax = plt.subplot(gs[1, 0])
# ax.axis('off')

# size_handles = [
#     plt.scatter(
#         [],
#         [],
#         s=cp_marker[r] * 20, # type: ignore
#         edgecolor="black",
#         facecolor="none",
#         alpha=1,
#         label=f"{r:g}",
#     )
#     for r in sorted(cp_unique)
# ]

# legend = ax.legend(
#     handles=size_handles,
#     loc="upper center",
#     ncol=6,
#     fontsize=6,
#     frameon=False,
#     columnspacing=0.6,   # decrease horizontal spacing
#     handletextpad=0.3,   # decrease marker-text spacing
#     borderpad=0.2,
#     labelspacing=0.2,
#     scatterpoints=1,
# )

# # Put the "title" below the legend
# ax.text(
#     0.5, -0.5,
#     "Storm to Drought Ratio",
#     ha="center",
#     va="bottom",
#     fontsize=6,
#     transform=ax.transAxes,
# )


# ax = plt.subplot(gs[1, 1])
# ax.axis('off')
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])


# cax = inset_axes(
#     ax,
#     width="80%",
#     height="40%",
#     loc="upper center",
# )

# cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
# cbar.set_ticks(np.arange(1, 10))
# cbar.set_label("Storm Onset Month", fontsize=6)
# cbar.ax.tick_params(labelsize=6)


png_path = Path(current_dir) / f"plots/storm_onset_month.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
# plt.subplots_adjust(hspace=0.5)
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
