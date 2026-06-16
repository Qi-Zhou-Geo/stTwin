#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-14T10:49:48
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
                color="C3", alpha=0.1, zorder=1)
    ax.axvspan(np.where(time_str == "2022-01-01T00:00:00")[0][0], 
               np.where(time_str == "2025-12-31T23:50:00")[0][0], 
                color="C1", alpha=0.1, zorder=1)



def uq_in_stochastic_or_mcmc1(source, key, stochastic_id, num_draw):
    
    value = []
    
    for mcmc_draw in range(2, num_draw + 1, 1):
        monitor_MAP = Path(project_root) / f'pipeline/run_2004_2025_posterior/v0dot4/theta_{mcmc_draw:03d}/sed_container.nc'
        ds_temp = xr.load_dataset(monitor_MAP)
        
        if source == "MCMC":
            value_temp = ds_temp[key][:, stochastic_id].values # shape: (time,)
        elif source == "Stochastic_MCMC":
            value_temp = ds_temp[key].values # shape as (time, num_stochastic)
        else:
            raise ValueError(f"Please check your UQ source.")
        
        value_temp = unit_converter(input=value_temp, catchment_area=4.83, method="area-aggregated")
        value.append(value_temp)
        
        del ds_temp # release memory
    
    # for MCMC
    # value is (time, num_draws)
    
    # for Stochastic_MCMC
    # value is (time, num_draws × num_stochastic)

    value = np.column_stack(value)
    print(f"source={source}, key={key}, stochastic_id={stochastic_id}, num_draw={num_draw}, value.shape={value.shape}")
    
    q05 = np.quantile(a=value, q=0.05, axis=1)
    q50 = np.quantile(a=value, q=0.50, axis=1)
    q95 = np.quantile(a=value, q=0.95, axis=1)

    y_mean = np.mean(value, axis=1)
    y_std = np.std(value, axis=1, ddof=1)

    return q05, q50, q95, y_mean, y_std

def uq_in_stochastic_or_mcmc(source, key, stochastic_id, num_draw):
    
    value = []
    
    for mcmc_draw in range(2, num_draw + 1, 1):
        monitor_MAP = Path(project_root) / f'pipeline/run_2004_2025_posterior/v0dot4/theta_{mcmc_draw:03d}/sed_container.nc'
        ds_temp = xr.load_dataset(monitor_MAP)
        
        if source == "MCMC":
            value_temp = ds_temp[key][:, stochastic_id].values # shape: (time,)
        elif source == "Stochastic_MCMC":
            value_temp = ds_temp[key].values # shape as (time, num_stochastic)
            value_temp = np.quantile(a=value_temp, q=0.50, axis=1)  # shape as (time,)
        else:
            raise ValueError(f"Please check your UQ source.")
        
        value_temp = unit_converter(input=value_temp, catchment_area=4.83, method="area-aggregated")
        
        value.append(value_temp)
        
        del ds_temp # release memory
    
    # for MCMC
    # value is (time, num_draws)
    
    # for Stochastic_MCMC
    # value is (time, num_draws × num_stochastic)

    value = np.column_stack(value)
    print(f"source={source}, key={key}, stochastic_id={stochastic_id}, num_draw={num_draw}, value.shape={value.shape}")
    
    q05 = np.quantile(a=value, q=0.05, axis=1)
    q50 = np.quantile(a=value, q=0.50, axis=1)
    q95 = np.quantile(a=value, q=0.95, axis=1)

    y_mean = np.mean(value, axis=1)
    y_std = np.std(value, axis=1, ddof=1)

    return q05, q50, q95, y_mean, y_std


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
y_label = ["Hillslope Storage\n" + r"[$10^5 \times \mathrm{m}^3$]", 
           "Channel Storage\n" + r"[$10^6 \times \mathrm{m}^3$]", 
           "Sediment Yield\n" + r"[$10^4 \times \mathrm{m}^3$]"]
y_zoom = np.array([1e5, 1e6, 1e4])
y_min = np.array([5.765e5, 0,   0]) / y_zoom
y_max = np.array([5.770e5, 5e6, 5e4]) / y_zoom
y_scale = ["linear", "linear", "linear"]
# endregion



fig = plt.figure(figsize=(6.5, 6))
gs = gridspec.GridSpec(3, 1)


# Fix MCMC = MAP, UQ along the ls stochastic dimension
zorder = 10
monitor_MAP = Path(project_root) / f'pipeline/run_2004_2025_posterior/v0dot4/theta_001/sed_container.nc'
ds = xr.load_dataset(monitor_MAP)
for idx, key in enumerate(["hillslope_storage", "channel_storage", "sed_transport_real"]):

    # region
    ax = plt.subplot(gs[idx])
    ax.set_title(label=f"{subplot_idx[idx]}", loc="left", fontsize=7, fontweight='bold')
    
    
    cache_path = Path(current_dir) / f"cache/stochastic_uq_in_{key}.npz"
    try:
        cache_data = np.load(cache_path, allow_pickle=True)
        q05, q50, q95 = cache_data["q05"], cache_data["q50"], cache_data["q95"]
        y_mean, y_std = cache_data["y_mean"], cache_data["y_std"]
        
        print(f"{UTCDateTime.now().isoformat()} Load the stochastic UQ data.\n{cache_path}")
    except:
        value = ds[key].values # shape as (time, num_stochastic)
        value = unit_converter(input=value, catchment_area=4.83, method="area-aggregated")

        q05 = np.quantile(a=value, q=0.05, axis=1) # collapse the num_stochastic dimension
        q50 = np.quantile(a=value, q=0.50, axis=1) # collapse the num_stochastic dimension
        q95 = np.quantile(a=value, q=0.95, axis=1) # collapse the num_stochastic dimension

        y_mean = np.mean(value, axis=1) # collapse the num_stochastic dimension
        y_std = np.std(value, axis=1, ddof=1) # collapse the num_stochastic dimension
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, q05=q05, q50=q50, q95=q95, y_mean=y_mean, y_std=y_std)
        print(f"{UTCDateTime.now().isoformat()} Cache the stochastic UQ data.\n{cache_path}")


    # y = y_mean
    # y1 = y_mean - y_std
    # y2 = y_mean + y_std
    
    y = q50 / y_zoom[idx]
    y1 = q05 / y_zoom[idx]
    y2 = q95 / y_zoom[idx]

    ax.plot(x, y, color="C2", zorder=zorder, alpha=0.75, label="Q50 (draw=MAP)")
    ax.fill_between(x, y1=y1, y2=y2, color="C2", zorder=zorder - 1, alpha=0.25, label="Q5 to Q95 (draw=MAP)")

    ax.set_xlim(x[0], x[-1]+1)
    ax.set_yscale(y_scale[idx])
    ax.set_ylim(y_min[idx], y_max[idx])

    ax.set_xticks(ticks_location, labels=ticks_label)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_ylabel(y_label[idx], fontweight='bold')
    ax.set_xlabel("", fontweight='bold')
    plot_region(ax)
    # endregion



# # UQ along the stochastic (Q50) + MCMC 20 draws
zorder = 5
for idx, key in enumerate(["hillslope_storage", "channel_storage", "sed_transport_real"]):

    # region
    ax = plt.subplot(gs[idx])
    ax.set_title(label=f"{subplot_idx[idx]}", loc="left", fontsize=7, fontweight='bold')

    cache_path = Path(current_dir) / f"cache/stochastic_+_mcmc_uq_in_{key}.npz"
    try:
        cache_data = np.load(cache_path, allow_pickle=True)
        q05, q50, q95 = cache_data["q05"], cache_data["q50"], cache_data["q95"]
        y_mean, y_std = cache_data["y_mean"], cache_data["y_std"]
        
        print(f"{UTCDateTime.now().isoformat()} Load the MCMC UQ data.\n{cache_path}")
    except:
        q05, q50, q95, y_mean, y_std = uq_in_stochastic_or_mcmc(source="Stochastic_MCMC", key=key, stochastic_id=66, num_draw=21)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, q05=q05, q50=q50, q95=q95, y_mean=y_mean, y_std=y_std)

        print(f"{UTCDateTime.now().isoformat()} Cache the MCMC UQ data.\n{cache_path}")

    y = q50 / y_zoom[idx]
    y1 = q05 / y_zoom[idx]
    y2 = q95 / y_zoom[idx]

    ax.plot(x, y, color="black", zorder=zorder, alpha=0.75, label="Q50 (draw=20)")
    ax.fill_between(x, y1=y1, y2=y2, color="black", zorder=zorder - 1, alpha=0.25, label="Q5 to Q95 (draw=20)")

    ax.set_xlim(x[0], x[-1]+1)
    ax.set_yscale(y_scale[idx])
    ax.set_ylim(y_min[idx], y_max[idx])

    ax.set_xticks(ticks_location, labels=ticks_label)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_ylabel(y_label[idx], fontweight='bold')
    ax.set_xlabel("", fontweight='bold')
    plot_region(ax)
    # endregion


ax = plt.subplot(gs[1])
ax.legend(loc="upper right", fontsize=6, ncol=1)

ax = plt.subplot(gs[2])
ax.set_xlabel("UTC+0 Time [Resolution = 10 minutes]", fontweight='bold')


plt.tight_layout()
png_path = Path(current_dir) / f"2004-2025_monitoring.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
