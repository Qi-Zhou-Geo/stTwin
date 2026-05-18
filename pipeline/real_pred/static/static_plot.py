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
project_root = current_dir.parent.parent.parent

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

    # shape as (time step 1 -> N, scenarios 1 -> N)
    arr = np.hstack(temp_l)

    return time_str, arr


def run_load_data():
    key_type = "sed"
    for key in ["sed_transport_real_Q50", "channel_storage_Q50"]:
        time_str, sed_arr = load_data(key_type, key)
        np.savez(f"./{key}.npz", time_str=time_str, key=sed_arr)
    
    
sed_transport_real_Q50 = np.load(f"./sed_transport_real_Q50.npz", allow_pickle=True)
time_str = sed_transport_real_Q50["time_str"]
idx = np.where(time_str=='2026-01-01T00:00:00')[0][0] + 1
time_str = time_str[:idx]
sed_transport_real_Q50 = sed_transport_real_Q50["key"]
sed_transport_real_Q50 = sed_transport_real_Q50[:idx, :]
print(sed_transport_real_Q50.shape)

channel_storage_Q50 = np.load(f"./channel_storage_Q50.npz", allow_pickle=True)
channel_storage_Q50 = channel_storage_Q50["key"]
channel_storage_Q50 = channel_storage_Q50[:idx, :]
print(channel_storage_Q50.shape)


# what if 1
sed_transport_real_Q50_whatif = np.load(f"/Users/qizhou/#python/stTwin/pipeline/what_if/sed_transport_real_Q50_1.npz", allow_pickle=True)
time_str1 = sed_transport_real_Q50_whatif["time_str"]
idx = np.where(time_str1=='2025-12-31T23:50:00')[0][0] + 1
time_str1 = time_str1[:idx]
sed_transport_real_Q50_whatif = sed_transport_real_Q50_whatif["key"]
sed_transport_real_Q50_whatif = sed_transport_real_Q50_whatif[:idx, :]
print(sed_transport_real_Q50_whatif.shape)

channel_storage_Q50_whatif = np.load(f"/Users/qizhou/#python/stTwin/pipeline/what_if/channel_storage_Q50_1.npz", allow_pickle=True)
channel_storage_Q50_whatif = channel_storage_Q50_whatif["key"]
channel_storage_Q50_whatif = channel_storage_Q50_whatif[:idx, :]
print(channel_storage_Q50_whatif.shape)


# what if2 
sed_transport_real_Q50_whatif2 = np.load(f"/Users/qizhou/#python/stTwin/pipeline/what_if/sed_transport_real_Q50_2.npz", allow_pickle=True)
time_str1 = sed_transport_real_Q50_whatif2["time_str"]
idx = np.where(time_str1=='2025-12-31T23:50:00')[0][0] + 1
time_str1 = time_str1[:idx]
sed_transport_real_Q50_whatif2 = sed_transport_real_Q50_whatif2["key"]
sed_transport_real_Q50_whatif2 = sed_transport_real_Q50_whatif2[:idx, :]
print(sed_transport_real_Q50_whatif2.shape)

channel_storage_Q50_whatif2 = np.load(f"/Users/qizhou/#python/stTwin/pipeline/what_if/channel_storage_Q50_2.npz", allow_pickle=True)
channel_storage_Q50_whatif2 = channel_storage_Q50_whatif2["key"]
channel_storage_Q50_whatif2 = channel_storage_Q50_whatif2[:idx, :]
print(channel_storage_Q50_whatif2.shape)


x_ticks_idx = []
x_ticks_label = []
for year in [2023, 2024, 2025, 2026]:
    for month in [1, 6]:
        
        if year == 2026 and month > 1:
            continue
    
        t = UTCDateTime(year=year, month=month, day=1).strftime("%Y-%m-%dT%H:%M:%S")
        idx = np.where(time_str == t)[0][0]
        
        x_ticks_idx.append(idx)
        t = UTCDateTime(year=year, month=month, day=1).strftime("%Y-%m-%d")
        x_ticks_label.append(t)
        
idx_2023_10_20 = np.where(time_str=="2023-10-24T06:00:00")[0][0]
x = np.arange(len(time_str))




fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 1)

# precp
ax = plt.subplot(gs[0])
data = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2023_2026_t.txt", header=0)
idx1 = np.where(data.iloc[:, 1]=='2023-01-01T00:00:00')[0][0]
idx2 = np.where(data.iloc[:, 1]=='2026-01-01T00:00:00')[0][0] + 1
y = data.iloc[idx1:idx2, 2].values
ax.plot(x, y, lw=1, color="black", zorder=3, label="Ten Minutes Total Precip. (MVE station)")
ax.axvline(x=idx_2023_10_20, ls="--", lw=0.5, color="black", label="2023-10-20T06:00:00", zorder=0)

ax.set_ylabel("Precipitation [mm]", fontweight="bold")
ax.set_xlim(x[0], x[-1])
ax.set_xlim(0, 7)
ax.set_xticks(x_ticks_idx, x_ticks_label)
ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.2, zorder=0)
ax.legend(loc="upper left", fontsize=6)



# channel storage
data = channel_storage_Q50
data = (data - data.min()) / (data.max() - data.min())
y = np.mean(data, axis=1)
x = np.arange(len(y))
y[0] = y[1]
std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[1])
ax.plot(x, y, lw=1, color="black", zorder=3, label="Mean")
ax.fill_between(x, y1, y2, color="black", alpha=0.2, zorder=2, label="Mean +- Std.")
ax.axvline(x=idx_2023_10_20, ls="--", lw=0.5, color="black", zorder=0)




# what-if1
data = channel_storage_Q50_whatif
data = (data - data.min()) / (data.max() - data.min())
y = np.mean(data, axis=1)
x = np.arange(len(y))
y[0] = y[1]
std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[1])
ax.plot(x, y, lw=1, color="C3", zorder=3, label="Mean (what-if, \nt0=1st April, f=60 days, Rs2d=0.1)")
ax.fill_between(x, y1, y2, color="C3", alpha=0.2, zorder=2, label="Mean +- Std.")



# what-if2
data = channel_storage_Q50_whatif2
data = (data - data.min()) / (data.max() - data.min())
y = np.mean(data, axis=1)
x = np.arange(len(y))
y[0] = y[1]
std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[1])
ax.plot(x, y, lw=1, color="C2", zorder=3, label="Mean (what-if, \nt0=1st April, f=120 days, Rs2d=0.1)")
ax.fill_between(x, y1, y2, color="C2", alpha=0.2, zorder=2, label="Mean +- Std.")





ax.set_ylabel("Normalized Channel Storage", fontweight="bold")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.05, 0.5)
ax.set_xticks(x_ticks_idx, x_ticks_label)
ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.2, zorder=0)
ax.legend(loc="upper left", fontsize=6, ncols=3)



# sediments
data = sed_transport_real_Q50
data = (data - data.min()) / (data.max() - data.min())
y = np.mean(data, axis=1)
x = np.arange(len(y))
std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[2])
ax.plot(x, y, lw=1, color="black", zorder=3, label="Mean")
ax.fill_between(x, y1, y2, color="black", alpha=0.2, zorder=2, label="Mean +- Std.")
print("real y", np.sum(y))
print("real y2 max", np.max(y2))

# what if 1
data = sed_transport_real_Q50_whatif
data = (data - data.min()) / (data.max() - data.min())
y = np.mean(data, axis=1)
x = np.arange(len(y))
std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[2])
ax.plot(x, y, lw=1, color="C3", zorder=3, label="Mean (what-if)")
ax.fill_between(x, y1, y2, color="C3", alpha=0.2, zorder=2, label="Mean +- Std.")
print("what y", np.sum(y))
print("what y2 max", np.max(y2))



# what if 2
data = sed_transport_real_Q50_whatif2
data = (data - data.min()) / (data.max() - data.min())
y = np.mean(data, axis=1)
x = np.arange(len(y))
std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[2])
ax.plot(x, y, lw=1, color="C2", zorder=3, label="Mean (what-if)")
ax.fill_between(x, y1, y2, color="C2", alpha=0.2, zorder=2, label="Mean +- Std.")
print("what y", np.sum(y))
print("what y2 max", np.max(y2))




ax.axvline(x=idx_2023_10_20, ls="--", lw=0.5, color="black", label="2023-10-20T06:00:00", zorder=0)
ax.set_ylabel("Normalized Sediment Yield", fontweight="bold")
ax.set_xlabel("Time", fontweight="bold")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 0.5)
ax.set_xticks(x_ticks_idx, x_ticks_label)
ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.2, zorder=0)


plt.tight_layout()
plt.savefig(f"./2023-2025.png", dpi=600, transparent=True)
plt.show()
plt.close(fig=fig)










# /Users/qizhou/#python/Flow-Alert/demo/ILL2023/2023-10-24_pro.txt
t1 = '2023-10-24T03:00:00'
t2 = '2023-10-24T18:00:00'

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 1)

# precp
ax = plt.subplot(gs[0])
data = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2023_2026_t.txt", header=0)
time_str = data.iloc[:, 1]
idx1 = np.where(time_str==t1)[0][0]
idx2 = np.where(time_str==t2)[0][0] + 1


time_str = time_str[idx1:idx2].values
y = data.iloc[idx1:idx2, 2].values
x = np.arange(len(y))

ax.plot(x, y, lw=1, color="black", zorder=3, label="Ten Minutes Total Precip. (MVE station)")
ax.set_ylabel("Precipitation [mm]", fontweight="bold")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 0.8)

x_ticks_idx, x_ticks_label = [], []
for ticks in ['2023-10-24T03:00:00', '2023-10-24T06:00:00', 
              '2023-10-24T09:00:00', '2023-10-24T12:00:00', 
              '2023-10-24T15:00:00', '2023-10-24T18:00:00']:
    idx = np.where(time_str == ticks)[0][0]
    x_ticks_idx.append(idx)
    x_ticks_label.append(ticks)

ax.set_xticks(x_ticks_idx, x_ticks_label)

ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.2, zorder=0)
ax.legend(loc="upper left", fontsize=6)



# pro
ax = plt.subplot(gs[1])
ax_twin = ax.twinx()
data = pd.read_csv("/Users/qizhou/#python/Flow-Alert/demo/ILL2023/2023-10-24_pro.txt", header=0)
time_str = data.iloc[:, 1].values
idx1 = np.where(time_str==t1)[0][0]
idx2 = np.where(time_str==t2)[0][0] + 1


time_str = time_str[idx1:idx2]
y = np.array(data.iloc[idx1:idx2, -2], dtype=float)
x = np.arange(len(y))


std = np.array(data.iloc[idx1:idx2, -1], dtype=float)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[1])
ax.plot(x, y, lw=1, color="black", zorder=3, label="Mean (Flow-Alert v1.3)")
ax.fill_between(x, y1, y2, color="black", alpha=0.2, zorder=2, label="Mean +- Std.")
ax.set_ylabel("Debris Flow Probability", fontweight="bold")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 1)

iqr = np.load("/Users/qizhou/#python/Flow-Alert/demo/ILL2023/iqr.npy")
ax_twin.plot(x, iqr, color="C0", zorder=3)
ax_twin.set_ylabel("Amplitude IQR [m/s]", fontweight="bold", color="C0")
ax_twin.tick_params(axis="y", colors="C0")

x_ticks_idx, x_ticks_label = [], []
for ticks in ['2023-10-24T03:00:00', '2023-10-24T06:00:00', 
              '2023-10-24T09:00:00', '2023-10-24T12:00:00', 
              '2023-10-24T15:00:00', '2023-10-24T18:00:00']:
    idx = np.where(time_str == ticks)[0][0]
    x_ticks_idx.append(idx)
    x_ticks_label.append(ticks)

ax.set_xticks(x_ticks_idx, x_ticks_label)
ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.2, zorder=0)




# channel storage
sed_transport_real_Q50 = np.load(f"./sed_transport_real_Q50.npz", allow_pickle=True)
time_str = sed_transport_real_Q50["time_str"]
idx1 = np.where(time_str==t1)[0][0]
idx2 = np.where(time_str==t2)[0][0] + 1

sed_transport_real_Q50 = sed_transport_real_Q50["key"]
sed_transport_real_Q50 = (sed_transport_real_Q50 - sed_transport_real_Q50.min()) / (sed_transport_real_Q50.max() - sed_transport_real_Q50.min())

sed_transport_real_Q50 = sed_transport_real_Q50[idx1:idx2, :]
time_str = time_str[idx1:idx2]
data = sed_transport_real_Q50

y = np.mean(data, axis=1)
x = np.arange(len(y))

std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[2])
ax.plot(x, y, lw=1, color="black", zorder=3, label="Mean")
ax.fill_between(x, y1, y2, color="black", alpha=0.2, zorder=2, label="Mean +- Std.")
print("subplot_real", np.max(y), np.max(y2))


# what-if1
sed_transport_real_Q50 = np.load(f"/Users/qizhou/#python/stTwin/pipeline/what_if/channel_storage_Q50_1.npz", allow_pickle=True)
time_str = sed_transport_real_Q50["time_str"]
idx1 = np.where(time_str==t1)[0][0]
idx2 = np.where(time_str==t2)[0][0] + 1

sed_transport_real_Q50 = sed_transport_real_Q50["key"]
sed_transport_real_Q50 = (sed_transport_real_Q50 - sed_transport_real_Q50.min()) / (sed_transport_real_Q50.max() - sed_transport_real_Q50.min())

sed_transport_real_Q50 = sed_transport_real_Q50[idx1:idx2, :]
time_str = time_str[idx1:idx2]
data = sed_transport_real_Q50
# data = (data - data.min()) / (data.max() - data.min())

y = np.mean(data, axis=1)
x = np.arange(len(y))

std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[2])
# ax.plot(x, y, lw=1, color="C3", zorder=3, label="Mean")
# ax.fill_between(x, y1, y2, color="C3", alpha=0.2, zorder=2, label="Mean +- Std.")
print("subplot_whatif1", np.max(y), np.max(y2))

# what-if2
sed_transport_real_Q50 = np.load(f"/Users/qizhou/#python/stTwin/pipeline/what_if/channel_storage_Q50_2.npz", allow_pickle=True)
time_str = sed_transport_real_Q50["time_str"]
idx1 = np.where(time_str==t1)[0][0]
idx2 = np.where(time_str==t2)[0][0] + 1

sed_transport_real_Q50 = sed_transport_real_Q50["key"]
sed_transport_real_Q50 = (sed_transport_real_Q50 - sed_transport_real_Q50.min()) / (sed_transport_real_Q50.max() - sed_transport_real_Q50.min())

sed_transport_real_Q50 = sed_transport_real_Q50[idx1:idx2, :]
time_str = time_str[idx1:idx2]
data = sed_transport_real_Q50

y = np.mean(data, axis=1)
x = np.arange(len(y))

std = np.std(data, axis=1, ddof=1)
y1 = y - std
y1 = np.clip(y1, a_max=y1.max(), a_min=0)
y2 = y + std
ax = plt.subplot(gs[2])
# ax.plot(x, y, lw=1, color="C2", zorder=3, label="Mean")
# ax.fill_between(x, y1, y2, color="C2", alpha=0.2, zorder=2, label="Mean +- Std.")
print("subplot_whatif2", np.max(y), np.max(y2))




ax.set_ylabel("Normalized Sediment Yield", fontweight="bold")
ax.set_xlabel("Time", fontweight="bold")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 0.03)

x_ticks_idx, x_ticks_label = [], []
for ticks in ['2023-10-24T03:00:00', '2023-10-24T06:00:00', 
              '2023-10-24T09:00:00', '2023-10-24T12:00:00', 
              '2023-10-24T15:00:00', '2023-10-24T18:00:00']:
    idx = np.where(time_str == ticks)[0][0]
    x_ticks_idx.append(idx)
    x_ticks_label.append(ticks)

ax.set_xticks(x_ticks_idx, x_ticks_label)
ax.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.2, zorder=0)


plt.tight_layout()
plt.savefig(f"./2023-10-24.png", dpi=600, transparent=True)
plt.show()
plt.close(fig=fig)

