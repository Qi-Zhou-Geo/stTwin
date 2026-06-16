#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-15T19:04:59
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


from obspy import UTCDateTime, read

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
from func.visulize.visualize_seismic import psd_plot
from func.toolkit.physical_unit_converter import unit_converter


plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 3),
                     'axes.formatter.use_mathtext': True})

t1 = "2020-06-29T03:30:00"
t0 = UTCDateTime(t1)
t2 = "2020-06-29T07:30:00" 


fig = plt.figure(figsize=(6, 5))
gs = gridspec.GridSpec(3, 1, height_ratios=[10, 10, 0.5])


ax = plt.subplot(gs[0])
ax.set_title(label=f"(a)", loc="left", fontsize=7, fontweight='bold')

# precp
df_path = Path(project_root) / "data/SedCas_input/climate_2004_2025_t.txt"
df = pd.read_csv(df_path, header=0)

time_str = df.iloc[:, 1].values
id1 = np.where(time_str==t1)[0][0]
id2 = np.where(time_str==t2)[0][0] + 1

mve_precp = np.array(df.iloc[id1:id2, 2])
x = np.arange(len(mve_precp))
ax.plot(x, mve_precp, color="C0", zorder=3, label="Total Precip.\nin 10 minutes")

ax.set_yticks([0, 0.4, 0.8, 1.2], labels=[0, 0.4, 0.8, 1.2]) # type: ignore
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 1.2)
ax.set_ylabel("Precipitation [mm]", fontweight='bold')
# ax.tick_params(axis="y", colors="C0")
# ax.spines["left"].set_color("C0")


# mve sed
key = "sed_transport_real"
sed_mve_path = Path(project_root) / f"pipeline/run_2004_2025_posterior/v0dot4/theta_001/sed_container.nc"
sed_mve = xr.load_dataset(sed_mve_path)

mask = (sed_mve.time_str >= t1) & (sed_mve.time_str <= t2)
sed_mve = sed_mve.isel(time=mask)

data = sed_mve[key].values
data = unit_converter(input=data, catchment_area=4.83, method="area-aggregated")
q05 = np.quantile(a=data, q=0.05, axis=1)
q50 = np.quantile(a=data, q=0.50, axis=1)
q95 = np.quantile(a=data, q=0.95, axis=1)
ax_twin = ax.twinx()
ax_twin.plot(x, q50, color="black", zorder=4, label="Q50")
ax_twin.fill_between(x, q05, q95, color="black", alpha=0.2, zorder=2, label="Q5 to Q95")
ax_twin.set_ylim(0, 6000)
ax_twin.set_ylabel("Sediment Yield " + r"[$\mathrm{m}^3$]", fontweight='bold', color="black")
ax_twin.xaxis.set_major_locator(MultipleLocator(6))
ax.set_xticks([0, 6, 12, 18, 24], labels=["03:30", "04:30", "05:30", "06:30", "07:30"]) # type: ignore

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=6, ncol=1)



ax = plt.subplot(gs[1])
ax.set_title(label=f"(b)", loc="left", fontsize=7, fontweight='bold')
cbar_ax = plt.subplot(gs[2])

st_path = Path(project_root) / f"data/open_access/qz2020/cooked_ILL12_EHZ_2020.mseed"
st = read(st_path)
st = st.trim(UTCDateTime(t1), UTCDateTime(t2))

pro_path = Path(project_root) / f"data/open_access/qz2020/model_prediction.npy"
pro = np.load(pro_path, allow_pickle=True)
time_str = pro[:, 1]
id1 = np.where(time_str == t1)[0][0]
id2 = np.where(time_str == t2)[0][0] + 1

pro_time = pro[id1:id2, 1]
pro_mean = np.array(pro[id1:id2, -2], dtype=float)
pro_std = np.array(pro[id1:id2, -1], dtype=float)



ax, data_sps = psd_plot(fig, ax, cbar_ax, st, fix_colorbar=True, per_lap=0.5, wlen=60, x_interval=1)
ax.set_xlabel("UTC+0 Time", fontweight='bold', color="black")

ax_prob = ax.twinx()
x = np.array([UTCDateTime(t) - t0 for t in pro_time], dtype=float)
ax_prob.plot(x, pro_mean, color="white", zorder=10, linewidth=2, label="Probability")
ax_prob.set_ylim(0, 1)
ax_prob.set_ylabel("Debris Flow Probability", fontweight='bold', color="black")
ax_prob.legend(loc="upper right", fontsize=6, ncol=1)



png_path = Path(current_dir) / f"seismic_vs_yield.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
