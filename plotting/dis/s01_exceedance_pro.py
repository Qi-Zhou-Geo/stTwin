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
import matplotlib.colors as mcolors

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


plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})


def exceedance_curve_all(x):
    
    x = np.asarray(x)
    x = np.sort(x)

    num_data = len(x)
    exceed_prob = 1 - np.arange(1, num_data + 1) / num_data

    return x, exceed_prob

subplot_idx = ["(a)", "(b)", "(c)"]
x_label = ["10-minute Precipitation Total", "aaa", "cc"]
t1 = "2023-01-01T00:00:00"
t2 = "2025-12-31T23:50:00"
col_idx = [2, 3, 4] # precp, tmeperature, radiation

precp_path = Path(project_root) / "data/SedCas_input/climate_2004_2025_t.txt"
df = pd.read_csv(precp_path, header=0)
date_str = np.array(df.iloc[:, 1])


# run all to get cache
t0_list = [1, 2, 3, 5, 7, 9]
cp_list = [30, 60, 90, 120]
# region all plots
fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(1, 3)

for idx, col in enumerate(col_idx):
    ax = plt.subplot(gs[idx])
    ax.set_title(f"{subplot_idx[idx]}", fontweight='bold', fontsize=7, loc='left')
    
    data_obs = np.array(df.iloc[:, col])

    # all MVE
    cache_path = Path(current_dir) / f"cache/mve_all_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs, sum_value = temp["x_obs"], temp["p_obs"], temp["sum_value"]
    except:
        x_obs, p_obs = exceedance_curve_all(x=data_obs)
        sum_value = np.sum(data_obs)
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs, sum_value=sum_value)
    ax.scatter(x_obs, p_obs, color="black", alpha=0.25, label="MVE 2004-2025")



    # 2023-2025 MVE
    cache_path = Path(current_dir) / f"cache/mve_2023-2025_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs, sum_value = temp["x_obs"], temp["p_obs"], temp["sum_value"]
    except:
        id1 = np.where(date_str == t1)[0][0]
        id2 = np.where(date_str == t2)[0][0]
        x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
        sum_value = np.sum(data_obs[id1:id2])
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs, sum_value=sum_value)
    ax.scatter(x_obs, p_obs, color="C2", alpha=0.25, label="MVE 2023-2025")



    # 2023-2025 Storm-to-Drought
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t0_list) * len(cp_list))) # type: ignore
    colors = [mcolors.to_hex(c) for c in colors]
    i = 0
    for t0 in t0_list:
        for cp in cp_list:
            
            cache_path = Path(current_dir) / f"cache/sed_2023-2025_{col}_t0={t0}_cp={cp}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                temp = np.load(cache_path)
                x_obs, p_obs, sum_value = temp["x_obs"], temp["p_obs"], temp["sum_value"]
            except:
                precp_path = Path(project_root) / f"data/SedCas_whatif_input/climate_2023_2026_t_whatif_CP={cp}_R=0.050_M={t0}_D=1.txt"
                df = pd.read_csv(precp_path, header=0)
                date_str = np.array(df.iloc[:, 1])
                
                data_obs = np.array(df.iloc[:, col])

                id1 = np.where(date_str == "2023-01-01T00:00:00")[0][0]
                id2 = np.where(date_str == "2025-12-31T23:50:00")[0][0]

                x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
                sum_value = np.sum(data_obs)
                np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs, sum_value=sum_value)

            ax.scatter(x_obs, p_obs, alpha=0.25, color=colors[i], label=f"S2D 2023-2025 " + rf"($M=${t0}, $1/f=${cp})")

            i = i + 1

    ax.set_yscale("log")
    # ax.legend()

    # ax.set_ylim(1e-7, 1e0)
    # ax.set_xlim(-2, 30)

    ax.set_xlabel(f"{x_label[idx]} " + r"$x$" + f" [mm]", fontweight='bold')
    ax.set_ylabel("Exceedance Probability " + r"$P(X > x)$", fontweight='bold')
    ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    print(f"Done: {idx}")

png_path = Path(current_dir) / f"exceedance_probability.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
# endregion



# in main
subplot_idx = ["(a) Fix " + r"$M$=2", "(a) Fix " + r"$1/f$=60"]
x_label = "10-minute Precipitation Total "  + r"$x$" + f" [mm]"
col_idx = [2, 2]
# region
fig = plt.figure(figsize=(6, 3.5))
gs = gridspec.GridSpec(1, 2)

for idx, col in enumerate(col_idx):
    ax = plt.subplot(gs[idx])
    ax.set_title(f"{subplot_idx[idx]}", fontweight='bold', fontsize=7, loc='left')
    
    data_obs = np.array(df.iloc[:, col])

    # all MVE
    cache_path = Path(current_dir) / f"cache/mve_all_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs, sum_value = temp["x_obs"], temp["p_obs"], temp["sum_value"]
    except:
        x_obs, p_obs = exceedance_curve_all(x=data_obs)
        sum_value = np.sum(data_obs)
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs, sum_value=sum_value)
    # ax.scatter(x_obs, p_obs, color="black", alpha=0.25, label="MVE 2004-2025")
    ax.plot(x_obs, p_obs, color="black", alpha=0.75, ls="-", label="MVE 2004-2025" + r"($\Sigma P_\text{mve}(t)$" + f"={sum_value:.0f})", zorder=20)


    # 2023-2025 MVE
    cache_path = Path(current_dir) / f"cache/mve_2023-2025_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs, sum_value = temp["x_obs"], temp["p_obs"], temp["sum_value"]
    except:
        id1 = np.where(date_str == t1)[0][0]
        id2 = np.where(date_str == t2)[0][0]
        x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
        sum_value = np.sum(data_obs[id1:id2])
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs, sum_value=sum_value)
    ax.plot(x_obs, p_obs, color="C2", alpha=0.75, ls="--", label="MVE 2023-2025" + r"($\Sigma P_\text{mve}(t)$" + f"={sum_value:.0f})", zorder=21)
    # ax.scatter(x_obs, p_obs, color="C2", alpha=0.25, label="MVE 2023-2025")
    sum_mve_2023_2025 = sum_value

    if idx == 0:
        t0_list = [2]
        cp_list = [30, 60, 90, 120]
    else:
        t0_list = [1, 3, 5, 7, 9]
        cp_list = [60]

    # 2023-2025 Storm-to-Drought
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t0_list) * len(cp_list))) # type: ignore
    colors = [mcolors.to_hex(c) for c in colors]
    i = 0
    for t0 in t0_list:
        for cp in cp_list:
            
            cache_path = Path(current_dir) / f"cache/sed_2023-2025_{col}_t0={t0}_cp={cp}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                temp = np.load(cache_path)
                x_obs, p_obs, sum_value = temp["x_obs"], temp["p_obs"], temp["sum_value"]
            except:
                precp_path = Path(project_root) / f"data/SedCas_whatif_input/climate_2023_2026_t_whatif_CP={cp}_R=0.050_M={t0}_D=1.txt"
                df = pd.read_csv(precp_path, header=0)
                date_str = np.array(df.iloc[:, 1])
                
                data_obs = np.array(df.iloc[:, col])

                id1 = np.where(date_str == "2023-01-01T00:00:00")[0][0]
                id2 = np.where(date_str == "2025-12-31T23:50:00")[0][0]

                x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
                sum_value = np.sum(data_obs[id1:id2])
                np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs, sum_value=sum_value)
            
            
            ratio_temp = sum_value / sum_mve_2023_2025
            
            # ax.plot(x_obs, p_obs, alpha=0.75, color=colors[i], 
            #         label=f"S2D 2023-2025 " + rf"($M=${t0}, $1/f=${cp})")
            
            if idx == 0:
                label = f"s2d 2023-2025 " + rf"($1/f={cp}$" + ", " + r"$\frac{\Sigma P_\text{s2d}(t)}{\Sigma P_\text{mve}(t)}$" + f"={ratio_temp:.2f})"
            else:
                label = f"s2d 2023-2025 " + rf"($M={t0}$" + ", " + r"$\frac{\Sigma P_\text{s2d}(t)}{\Sigma P_\text{mve}(t)}$" + f"={ratio_temp:.2f})"
            ax.scatter(x_obs, p_obs, marker="o", alpha=0.35, color=colors[i], label=label)

            i = i + 1

    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=6)

    ax.set_ylim(1e-7, 1e0)
    ax.set_xlim(-2, 60)

    ax.set_xlabel(x_label, fontweight='bold')
    
    if idx == 0:
        ax.set_ylabel("Exceedance Probability " + r"$P(X > x)$", fontweight='bold')
    else:
        ax.set_ylabel("")
    ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    print(f"Done: {idx}")

png_path = Path(current_dir) / f"exceedance_probability_precp.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
# endregion


exit()



# in SI
x_label = "Temperature "  + r"$x$" + rf" [$^\circ$C]"
col_idx = [3, 3]
subplot_idx = ["(a)", "(b)"]
fig = plt.figure(figsize=(6, 7))
gs = gridspec.GridSpec(2, 2)

# region
for idx, col in enumerate(col_idx):
    ax = plt.subplot(gs[idx])
    ax.set_title(f"{subplot_idx[idx]}", fontweight='bold', fontsize=7, loc='left')
    
    data_obs = np.array(df.iloc[:, col])
    id1 = np.where(date_str == t1)[0][0]
    id2 = np.where(date_str == t2)[0][0]

    # all MVE
    cache_path = Path(current_dir) / f"cache/mve_all_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs = temp["x_obs"], temp["p_obs"]
    except:
        x_obs, p_obs = exceedance_curve_all(x=data_obs)
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs)
    ax.plot(x_obs, p_obs, color="black", alpha=0.75, label="MVE 2004-2025", zorder=10)



    # 2023-2025 MVE
    cache_path = Path(current_dir) / f"cache/mve_2023-2025_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs = temp["x_obs"], temp["p_obs"]
    except:
        x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs)
    ax.plot(x_obs, p_obs, color="C2", alpha=0.75, ls="--", label="MVE 2023-2025", zorder=11)


    if idx == 0:
        t0_list = [2]
        cp_list = [30, 60, 90, 120]
    else:
        t0_list = [1, 3, 5, 7, 9]
        cp_list = [60]

    # 2023-2025 Storm-to-Drought
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t0_list) * len(cp_list))) # type: ignore
    colors = [mcolors.to_hex(c) for c in colors]
    i = 0
    for t0 in t0_list:
        for cp in cp_list:
            
            cache_path = Path(current_dir) / f"cache/sed_2023-2025_{col}_t0={t0}_cp={cp}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                temp = np.load(cache_path)
                x_obs, p_obs = temp["x_obs"], temp["p_obs"]
            except:
                precp_path = Path(project_root) / f"data/SedCas_whatif_input/climate_2023_2026_t_whatif_CP={cp}_R=0.050_M={t0}_D=1.txt"
                df = pd.read_csv(precp_path, header=0)
                date_str = np.array(df.iloc[:, 1])
                
                data_obs = np.array(df.iloc[:, col])

                id1 = np.where(date_str == "2023-01-01T00:00:00")[0][0]
                id2 = np.where(date_str == "2025-12-31T23:50:00")[0][0]

                x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
                np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs)

            ax.scatter(x_obs, p_obs, marker="o", alpha=0.75, color=colors[i], 
                       label=f"S2D 2023-2025 " + rf"($M=${t0}, $1/f=${cp})")

            i = i + 1

    ax.set_yscale("log")
    ax.legend(loc="lower left", fontsize=6)

    ax.set_ylim(1e-6, 5e0)
    ax.set_xlim(-20, 30)

    ax.set_xlabel(x_label, fontweight='bold')
    
    if idx == 0:
        ax.set_ylabel("Exceedance Probability " + r"$P(X > x)$", fontweight='bold')
    else:
        ax.set_ylabel("")
    ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    print(f"Done: {idx}")
# endregion


# in SI
x_label = "Sun Radiation "  + r"$x$" + r" [$Wm^{-2}$]"
col_idx = [4, 4]
subplot_idx = ["(c)", "(d)"]
# region
for idx, col in enumerate(col_idx):
    
    ax = plt.subplot(gs[idx + 2])
    ax.set_title(f"{subplot_idx[idx]}", fontweight='bold', fontsize=7, loc='left')
    
    data_obs = np.array(df.iloc[:, col])
    id1 = np.where(date_str == t1)[0][0]
    id2 = np.where(date_str == t2)[0][0]

    # all MVE
    cache_path = Path(current_dir) / f"cache/mve_all_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs = temp["x_obs"], temp["p_obs"]
    except:
        x_obs, p_obs = exceedance_curve_all(x=data_obs)
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs)
    ax.plot(x_obs, p_obs, color="black", alpha=0.75, label="MVE 2004-2025", zorder=10)



    # 2023-2025 MVE
    cache_path = Path(current_dir) / f"cache/mve_2023-2025_{col}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp = np.load(cache_path)
        x_obs, p_obs = temp["x_obs"], temp["p_obs"]
    except:
        x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
        np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs)
    ax.plot(x_obs, p_obs, color="C2", alpha=0.75, ls="--", label="MVE 2023-2025", zorder=11)


    if idx == 0:
        t0_list = [2]
        cp_list = [30, 60, 90, 120]
    else:
        t0_list = [1, 3, 5, 7, 9]
        cp_list = [60]

    # 2023-2025 Storm-to-Drought
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t0_list) * len(cp_list))) # type: ignore
    colors = [mcolors.to_hex(c) for c in colors]
    i = 0
    for t0 in t0_list:
        for cp in cp_list:
            
            cache_path = Path(current_dir) / f"cache/sed_2023-2025_{col}_t0={t0}_cp={cp}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                temp = np.load(cache_path)
                x_obs, p_obs = temp["x_obs"], temp["p_obs"]
            except:
                precp_path = Path(project_root) / f"data/SedCas_whatif_input/climate_2023_2026_t_whatif_CP={cp}_R=0.050_M={t0}_D=1.txt"
                df = pd.read_csv(precp_path, header=0)
                date_str = np.array(df.iloc[:, 1])
                
                data_obs = np.array(df.iloc[:, col])

                id1 = np.where(date_str == "2023-01-01T00:00:00")[0][0]
                id2 = np.where(date_str == "2025-12-31T23:50:00")[0][0]

                x_obs, p_obs = exceedance_curve_all(x=data_obs[id1:id2])
                np.savez_compressed(cache_path, x_obs=x_obs, p_obs=p_obs)

            ax.scatter(x_obs, p_obs, marker="o", alpha=0.75, color=colors[i], 
                       label=f"S2D 2023-2025 " + rf"($M=${t0}, $1/f=${cp})")

            i = i + 1

    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=6)

    ax.set_ylim(1e-6, 1e0)
    ax.set_xlim(-2, 4000)

    ax.set_xlabel(x_label, fontweight='bold')
    
    if idx == 0:
        ax.set_ylabel("Exceedance Probability " + r"$P(X > x)$", fontweight='bold')
    else:
        ax.set_ylabel("")
    ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    print(f"Done: {idx}")
# endregion


png_path = Path(current_dir) / f"exceedance_probability_temperature_radiation.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)

