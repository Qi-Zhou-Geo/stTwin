#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import numpy as np
import pandas as pd

import xarray as xr

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

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})


def load_data(key_type, key, model_version="bayesian_inference0dot4", num_draw=100):
    if key_type in ["hydro"]:
        nc_file = "hydro_output.nc"
    elif key_type in ["sed"]:
        nc_file = "sed_output.nc"

    time_str = None
    temp_l = []

    for theta_idx in range(1, num_draw + 1):

        theta_idx = str(theta_idx).zfill(3)

        output = xr.load_dataset(f"{project_root}/pipeline/real_pred/"
                                 f"{model_version}/theta_{theta_idx}/{nc_file}")

        if time_str is None:
            time_str = output.coords["time_str"].values

        output = output[key].values
        temp_l.append(output.reshape(-1, 1))

    # shape as (time step 1 -> N, seniors 1 -> N)
    arr = np.hstack(temp_l)

    return time_str, arr


def plot_results(output_dir, select_t1=None, select_t2=None):

    key_type, key = "hydro", "Qs"
    time_str, hydro_arr = load_data(key_type, key)

    key_type, key = "sed", "sed_transport_real_Q50"
    time_str, sed_arr = load_data(key_type, key)

    if select_t1 is None or select_t2 is None:
        pass
    else:
        id1, id2 = np.where(time_str == select_t1)[0][0], np.where(time_str == select_t2)[0][0] + 1
        time_str = time_str[id1:id2]
        hydro_arr = hydro_arr[id1:id2]
        sed_arr = sed_arr[id1:id2]

    df = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2023_2026_t.txt", header=0)
    time_date = df['timestamp [UTC+0]'].values
    id1, id2 = np.where(time_date == select_t1)[0][0], np.where(time_date == select_t2)[0][0] + 1
    precipitation = df['precipitation [mm per time_step]'].values[id1:id2]

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(3, 1)

    x = np.arange(len(time_str))

    key = "precipitation"
    ax = fig.add_subplot(gs[0])
    ax.plot(x, precipitation, color='black')
    ax.set_ylabel(key, fontweight='bold')
    ax.set_xlabel(f"UTC+0 Time [from {time_str[0]}]", fontweight='bold')
    ax.xaxis.set_major_locator(MultipleLocator(6)) # every 2 hours
    ax.xaxis.set_minor_locator(MultipleLocator(1)) # every 10 minutes
    ax.set_xlim(x.min(), x.max())

    key = "Qs"
    ax = fig.add_subplot(gs[1])
    y_mean = np.mean(hydro_arr, axis=1) # by row
    y_std = np.std(hydro_arr, axis=1) # by row
    y_lower = y_mean - y_std
    y_lower = np.clip(y_lower, a_min=0, a_max=np.max(y_lower))
    y_upper = y_mean + y_std

    ax.plot(x, y_mean, label=key, color="black", zorder=3)
    ax.fill_between(x, y_lower, y_upper, color="black", alpha=0.2, zorder=2)
    ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    ax.set_ylabel(key, fontweight='bold')
    ax.set_xlabel(f"UTC+0 Time [from {time_str[0]}]", fontweight='bold')
    ax.xaxis.set_major_locator(MultipleLocator(6)) # every 2 hours
    ax.xaxis.set_minor_locator(MultipleLocator(1)) # every 10 minutes
    ax.set_xlim(x.min(), x.max())

    key ="sed_transport_real_Q50"
    ax = fig.add_subplot(gs[2])
    y_mean = np.mean(sed_arr, axis=1) # by row
    y_std = np.std(sed_arr, axis=1) # by row
    y_lower = y_mean - y_std
    y_lower = np.clip(y_lower, a_min=0, a_max=np.max(y_lower))
    y_upper = y_mean + y_std
    ax.plot(x, y_mean, label=key, color="black", zorder=3)
    ax.fill_between(x, y_lower, y_upper, color="black", alpha=0.2, zorder=2)
    ax.grid(axis='y', color='grey', which="both", linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    ax.set_ylabel(key, fontweight='bold')
    ax.set_xlabel(f"UTC+0 Time [from {time_str[0]}]", fontweight='bold')
    ax.xaxis.set_major_locator(MultipleLocator(6)) # every 2 hours
    ax.xaxis.set_minor_locator(MultipleLocator(1)) # every 10 minutes
    ax.set_xlim(x.min(), x.max())

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{select_t1}_{select_t2}.png", dpi=600)  # , transparent=True
    plt.close(fig=fig)

def main():
    model_version = "bayesian_inference0dot4"
    select_t1, select_t2 = "2023-04-28T22:00:00", "2023-04-29T01:00:00"
    # select_t1, select_t2 = "2023-06-21T07:00:00", "2023-06-21T11:00:00"
    output_dir = f"{project_root}/pipeline/real_pred/{model_version}"
    plot_results(output_dir, select_t1, select_t2)

if __name__ == "__main__":
    main()