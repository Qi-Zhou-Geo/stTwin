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


def custom_ticks():
    
    # precp
    df_path = Path(project_root) / "data/SedCas_input/climate_2023_2026_t.txt"
    df = pd.read_csv(df_path, header=0)
    
    time_str = df.iloc[:, 1].values
    id1 = np.where(time_str=="2023-01-01T00:00:00")[0][0]
    id2 = np.where(time_str=="2025-12-31T23:50:00")[0][0]
    df = df.iloc[id1:id2, :]
    
    time_str = df.iloc[:, 1]
    
    # major ticks
    major_ticks_location = []
    major_ticks_label = ["2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"]
    for label in major_ticks_label:
        idx = np.where(time_str == f"{label}T00:00:00")[0][0]
        major_ticks_location.append(idx)

    major_ticks_location.append(len(df))
    major_ticks_label.append("2026-01-01")

    # minor ticks
    minor_ticks_location = []
    for year in range(2023, 2025 + 1, 1):
        for month in range(1, 12 + 1, 1):
            label = f"{year}-{month:02d}-01T00:00:00"
            idx = np.where(time_str == label)[0][0]
            minor_ticks_location.append(idx)

    return major_ticks_location, major_ticks_label, minor_ticks_location

def load_whatif_precp(storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0):
    
    # precp
    df_path = Path(project_root) / "data/SedCas_input/climate_2023_2026_t.txt"
    df = pd.read_csv(df_path, header=0)
    
    time_str = df.iloc[:, 1].values
    id1 = np.where(time_str=="2023-01-01T00:00:00")[0][0]
    id2 = np.where(time_str=="2025-12-31T23:50:00")[0][0]
    
    mve_precp = df.iloc[id1:id2, 2]
    x = np.arange(len(mve_precp))


    whatif_type = f"CP={cycle_period}_R={storm2drought_ratio}_M={storm_onset_month}_D={storm_onset_day}"
    whatif_precp_path = Path(project_root) / f"data/SedCas_whatif_input/climate_2023_2026_t_whatif_{whatif_type}.txt"
    whatif_precp = pd.read_csv(whatif_precp_path, header=0)
    
    time_str = df.iloc[:, 1].values
    id1 = np.where(time_str=="2023-01-01T00:00:00")[0][0]
    id2 = np.where(time_str=="2025-12-31T23:50:00")[0][0]
    
    whatif_precp = whatif_precp.iloc[id1:id2, 2]
    
    
    print(storm2drought_ratio, storm_onset_month,
          f"len(mve_precp) = {len(mve_precp)}, len(whatif_precp) = {len(whatif_precp)}\n"
          f"np.sum(mve_precp) = {np.sum(mve_precp):.1f}, np.sum(whatif_precp) = {np.sum(whatif_precp):.1f}\n")

    return x, mve_precp, whatif_precp

def load_whatif_sed(key, storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0):
    select_t1, select_t2 = "2023-01-01T00:00:00", "2025-12-31T23:50:00"
    
    if key == "channel_storage":
        y_zoom = 1e6
    elif key == "sed_transport_real":
        y_zoom = 1e4
    else:
        raise ValueError(f"check your key={key}")
        
    
    # mve
    sed_mve_path = Path(project_root) / f"pipeline/run_2004_2025_posterior/v0dot4/theta_001/sed_container.nc"
    sed_mve = xr.load_dataset(sed_mve_path)

    mask = (sed_mve.time_str >= select_t1) & (sed_mve.time_str <= select_t2)
    sed_mve = sed_mve.isel(time=mask)
            
    time_str = sed_mve.time_str.values
    x = np.arange(len(time_str))

    data = sed_mve[key].values
    data = unit_converter(input=data, catchment_area=4.83, method="area-aggregated")
    data = data / y_zoom
    q05 = np.quantile(a=data, q=0.05, axis=1)
    q50 = np.quantile(a=data, q=0.50, axis=1)
    q95 = np.quantile(a=data, q=0.95, axis=1)
    sed_mve = (q05, q50, q95)

    print(storm2drought_ratio, storm_onset_month, f"sed_mve: np.sum(q50) = {np.sum(q50)}")
    
    
    # wahtif
    whatif_type = f"CP={cycle_period}_R={storm2drought_ratio}_M={storm_onset_month}_D={storm_onset_day}"
    sed_whatif_path = Path(project_root) / f"pipeline/run_whatif/v0dot4/{whatif_type}/theta_001/sed_container.nc"
    sed_whatif = xr.load_dataset(sed_whatif_path)

    mask = (sed_whatif.time_str >= select_t1) & (sed_whatif.time_str <= select_t2)
    sed_whatif = sed_whatif.isel(time=mask)
            
    time_str = sed_whatif.time_str.values
    x = np.arange(len(time_str))

    data = sed_whatif[key].values
    data = unit_converter(input=data, catchment_area=4.83, method="area-aggregated")
    data = data / y_zoom
    q05 = np.quantile(a=data, q=0.05, axis=1)
    q50 = np.quantile(a=data, q=0.50, axis=1)
    q95 = np.quantile(a=data, q=0.95, axis=1)
    sed_whatif = (q05, q50, q95)
    
    print(storm2drought_ratio, storm_onset_month, f"sed_whatif: np.sum(q50) = {np.sum(sed_whatif)}")
    
    return x, sed_mve, sed_whatif


def plot_precp1():
    
    storm_onset_month = 3.0
    storm2drought_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_case = len(storm2drought_ratio_list)
    major_ticks_location, major_ticks_label, minor_ticks_location = custom_ticks()
    
    fig = plt.figure(figsize=(6, 7))
    gs = gridspec.GridSpec(num_case, 1)
    
    for idx, storm2drought_ratio in enumerate(storm2drought_ratio_list):
        ax = plt.subplot(gs[idx])
        x, mve_precp, whatif_precp = load_whatif_precp(storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0)
        ax.plot(x, whatif_precp, color="C0", alpha=0.75, label=r"$R_{s2d}$" + f"={storm2drought_ratio}", zorder=3)
        
        if idx == 0:
            ax.plot(x, mve_precp, color="black", alpha=0.5, label="MVE Obs.", zorder=2)
        else:
            ax.plot(x, mve_precp, color="black", alpha=0.5, zorder=2)

        ax.legend(loc="upper right", fontsize=6, ncol=2)
        ax.set_ylim(1e-1, 1e2)
        ax.set_yscale("log")
        ax.set_xlim(x[0], x[-1])
        
        if idx == num_case - 1:
            ax.set_xticks(major_ticks_location, major_ticks_label)
        else:
            ax.set_xticks(major_ticks_location, [])
        ax.set_xticks(minor_ticks_location, minor=True)
        ax.set_xlabel("", fontweight='bold')
        ax.set_ylabel("Precipitation\n[mm]", fontweight='bold')
        ax.grid(axis="x", color="grey", which="both", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
        ax.grid(axis="y", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax = plt.subplot(gs[num_case - 1])
    ax.set_xlabel("UTC+0 Time [Resolution = 10 minutes]", fontweight='bold')
    
    png_path = Path(current_dir) / "plots" / f"precp_storm_onset_month={storm_onset_month}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=600)
    plt.show()
    plt.close(fig)

# plot_precp1()

def plot_precp2():
    
    storm_onset_month_list = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    storm2drought_ratio = 0.2
    num_case = len(storm_onset_month_list)
    major_ticks_location, major_ticks_label, minor_ticks_location = custom_ticks()
    
    fig = plt.figure(figsize=(6, 7))
    gs = gridspec.GridSpec(num_case, 1)
    
    for idx, storm_onset_month in enumerate(storm_onset_month_list):
        ax = plt.subplot(gs[idx])
        x, mve_precp, whatif_precp = load_whatif_precp(storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0)
        ax.plot(x, whatif_precp, color="C0", alpha=0.75, label=r"$R_{s2d}$" + f"={storm2drought_ratio}", zorder=3)
        
        if idx == 0:
            ax.plot(x, mve_precp, color="black", alpha=0.5, label="MVE Obs.", zorder=2)
        else:
            ax.plot(x, mve_precp, color="black", alpha=0.5, zorder=2)
            
        ax.legend(loc="upper right", fontsize=6, ncol=2)
        ax.set_ylim(1e-1, 1e2)
        ax.set_yscale("log")
        ax.set_xlim(x[0], x[-1])
        
        if idx == num_case - 1:
            ax.set_xticks(major_ticks_location, major_ticks_label)
        else:
            ax.set_xticks(major_ticks_location, [])
        ax.set_xticks(minor_ticks_location, minor=True)
        ax.set_xlabel("", fontweight='bold')
        ax.set_ylabel("Precipitation\n[mm]", fontweight='bold')
        ax.grid(axis="x", color="grey", which="both", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
        ax.grid(axis="y", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax = plt.subplot(gs[num_case - 1])
    ax.set_xlabel("UTC+0 Time [Resolution = 10 minutes]", fontweight='bold')
    
    png_path = Path(current_dir) / "plots" / f"precp_storm2drought_ratio={storm2drought_ratio}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=600)
    plt.show()
    plt.close(fig)

plot_precp2()



def plot_sed():
    
    storm_onset_month = 3.0
    storm2drought_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_case = len(storm2drought_ratio_list)
    colors = plt.cm.viridis(np.linspace(0, 1, num_case)) # type: ignore
    major_ticks_location, major_ticks_label, minor_ticks_location = custom_ticks()
    
    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 1)
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    for idx, storm2drought_ratio in enumerate(storm2drought_ratio_list):
        
        key = "channel_storage"
        x, sed_mve, sed_whatif = load_whatif_sed(key, storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0)
        
        if idx == 0:
            q05, q50, q95 = sed_mve
            ax0.plot(x, q50, color="black", alpha=0.75, label="MVE Obs.", zorder=3)
            ax0.fill_between(x, q05, q95, color="black", alpha=0.1, zorder=2)
            
        q05, q50, q95 = sed_whatif
        ax0.plot(x, q50, color=colors[idx], alpha=0.75, label=r"$R_{s2d}$" + f"={storm2drought_ratio}", zorder=8)
        ax0.fill_between(x, q05, q95, color=colors[idx], alpha=0.1, zorder=7)
        ax0.set_xlim(x[0], x[-1])



        key = "sed_transport_real"
        x, sed_mve, sed_whatif = load_whatif_sed(key, storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0)
        if idx == 0:
            q05, q50, q95 = sed_mve
            ax1.plot(x, q50, color="black", alpha=0.75, label="MVE Obs.", zorder=3)
            ax1.fill_between(x, q05, q95, color="black", alpha=0.1, zorder=2)
            
        q05, q50, q95 = sed_whatif
        ax1.plot(x, q50, color=colors[idx], alpha=0.75, label=r"$R_{s2d}$" + f"={storm2drought_ratio}", zorder=8)
        ax1.fill_between(x, q05, q95, color=colors[idx], alpha=0.1, zorder=7)
        ax1.set_xlim(x[0], x[-1])
        
    ax0.set_ylim(0, 2.5)
    ax0.legend(loc="upper right", fontsize=6, ncol=1)
    ax0.set_xticks(major_ticks_location, [])
    ax0.set_xticks(minor_ticks_location, minor=True)
    ax0.set_xlabel("", fontweight='bold')
    ax0.set_ylabel("Channel Storage\n" + r"[$10^6 \times \mathrm{m}^3$]", fontweight='bold')
    ax0.grid(axis="x", color="grey", which="both", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax0.grid(axis="y", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax1.set_ylim(0, 3)
    ax1.set_xticks(major_ticks_location, major_ticks_label)
    ax1.set_xticks(minor_ticks_location, minor=True)
    ax1.set_xlabel("", fontweight='bold')
    ax1.set_ylabel("Sediment Yield\n" + r"[$10^4 \times \mathrm{m}^3$]", fontweight='bold')
    ax1.grid(axis="x", color="grey", which="both", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax1.grid(axis="y", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax1.set_xlabel("UTC+0 Time [Resolution = 10 minutes]", fontweight='bold')
    
    png_path = Path(current_dir) / "plots" / f"sed_storm_onset_month={storm_onset_month}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=600)
    plt.show()
    plt.close(fig)


def plot_sed1():
    
    storm_onset_month = 3.0
    storm2drought_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    num_case = len(storm2drought_ratio_list)
    colors = plt.cm.viridis(np.linspace(0, 1, num_case)) # type: ignore
    major_ticks_location, major_ticks_label, minor_ticks_location = custom_ticks()
    
    fig = plt.figure(figsize=(6, 7))
    gs = gridspec.GridSpec(num_case, 1)
    
    for idx, storm2drought_ratio in enumerate(storm2drought_ratio_list):
        ax0 = plt.subplot(gs[idx])
        
        key = "channel_storage"
        x, sed_mve, sed_whatif = load_whatif_sed(key, storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0)

        q05, q50, q95 = sed_mve
        if idx == 0:
            ax0.plot(x, q50, color="black", alpha=0.75, label="MVE Obs.", zorder=3)
        else:
            ax0.plot(x, q50, color="black", alpha=0.75, zorder=3)
        ax0.fill_between(x, q05, q95, color="black", alpha=0.1, zorder=2)
        
        q05, q50, q95 = sed_whatif
        ax0.plot(x, q50, color=colors[idx], alpha=0.75, label=r"$R_{s2d}$" + f"={storm2drought_ratio}", zorder=8)
        ax0.fill_between(x, q05, q95, color=colors[idx], alpha=0.1, zorder=7)
        ax0.set_xlim(x[0], x[-1])

        ax0.set_ylim(0, 2.5)
        ax0.legend(loc="upper right", fontsize=6, ncol=1)
        ax0.set_xticks(major_ticks_location, [])
        ax0.set_xticks(minor_ticks_location, minor=True)
        ax0.set_xlabel("", fontweight='bold')
        ax0.set_ylabel("Channel Storage\n" + r"[$10^6 \times \mathrm{m}^3$]", fontweight='bold')
        ax0.grid(axis="x", color="grey", which="both", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
        ax0.grid(axis="y", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax0 = plt.subplot(gs[num_case - 1])
    ax0.set_xlabel("UTC+0 Time [Resolution = 10 minutes]", fontweight='bold')
    
    png_path = Path(current_dir) / "plots" / f"sed_storm_onset_month={storm_onset_month}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=600)
    plt.show()
    plt.close(fig)

plot_sed1()


def plot_sed2():
    
    storm_onset_month_list = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    storm2drought_ratio = 0.2
    num_case = len(storm_onset_month_list)
    colors = plt.cm.viridis(np.linspace(0, 1, num_case)) # type: ignore
    major_ticks_location, major_ticks_label, minor_ticks_location = custom_ticks()
    
    fig = plt.figure(figsize=(6, 7))
    gs = gridspec.GridSpec(num_case, 1)
    
    for idx, storm_onset_month in enumerate(storm_onset_month_list):
        ax0 = plt.subplot(gs[idx])
        
        
        key = "channel_storage"
        x, sed_mve, sed_whatif = load_whatif_sed(key, storm2drought_ratio, storm_onset_month, cycle_period=30.0, storm_onset_day=1.0)
        
        q05, q50, q95 = sed_mve
        if idx == 0:
            ax0.plot(x, q50, color="black", alpha=0.75, label="MVE Obs.", zorder=3)
        else:
            ax0.plot(x, q50, color="black", alpha=0.75, zorder=3)
        ax0.fill_between(x, q05, q95, color="black", alpha=0.1, zorder=2)
            
        q05, q50, q95 = sed_whatif
        ax0.plot(x, q50, color=colors[idx], alpha=0.75, label=f"Storm Onset Month={int(storm_onset_month)}", zorder=8)
        ax0.fill_between(x, q05, q95, color=colors[idx], alpha=0.1, zorder=7)
        ax0.set_xlim(x[0], x[-1])
        
        ax0.set_ylim(0, 2.5)
        ax0.legend(loc="upper right", fontsize=6, ncol=1)
        ax0.set_xticks(major_ticks_location, [])
        ax0.set_xticks(minor_ticks_location, minor=True)
        ax0.set_xlabel("", fontweight='bold')
        ax0.set_ylabel("Channel Storage\n" + r"[$10^6 \times \mathrm{m}^3$]", fontweight='bold')
        ax0.grid(axis="x", color="grey", which="both", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
        ax0.grid(axis="y", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax0 = plt.subplot(gs[num_case - 1])
    ax0.set_xlabel("UTC+0 Time [Resolution = 10 minutes]", fontweight='bold')
    
    png_path = Path(current_dir) / "plots" / f"sed_storm2drought_ratio={storm2drought_ratio}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=600)
    plt.show()
    plt.close(fig)

plot_sed2()
