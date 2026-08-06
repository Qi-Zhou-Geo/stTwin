#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-28T09:10:22
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
from func.visulize.visualize_seismic import psd_plot, rewrite_x_ticks
from func.toolkit.physical_unit_converter import unit_converter


plt.rcParams.update({'font.size': 7, # 7pt 
                     'axes.formatter.limits': (-3, 3), # 1e-3 to 1e3 >> liner scale, otherwise, log scale
                     'axes.formatter.use_mathtext': True}) # scientific style 1 \times 10^3, rather than 1e3

t1 ="2022-06-05T10:30:00"
t0 = UTCDateTime(t1)
t2 ="2022-06-05T13:30:00" 


fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(4, 1, height_ratios=[10, 10, 10, 0.5])


# region
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
ax.plot(x, mve_precp, color="C0", zorder=3, label="MetroSwiss Montana")

ax.set_yticks([0, 0.4, 0.8, 1.2], labels=[0, 0.4, 0.8, 1.2]) # type: ignore
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 1.2)
ax.set_ylabel("Precipitation\n" + r"$[\mathrm{mm}$ / $\mathrm{10-min}]$", fontweight='bold')


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

id1 = np.where(sed_mve.time_str == "2022-06-05T11:00:00")[0][0]
id2 = np.where(sed_mve.time_str == "2022-06-05T12:00:00")[0][0]
print(f"np.sum(q05[id1:id2]) = {np.sum(q05[id1:id2])} [m**3 per 10 minutes]," 
      f"np.sum(q50[id1:id2]) = {np.sum(q50[id1:id2])}," 
      f"np.sum(q95[id1:id2]) = {np.sum(q95[id1:id2])}")


ax_twin = ax.twinx()
ax_twin.plot(x, q50, color="black", zorder=4, label="Q50")
ax_twin.fill_between(x, q05, q95, color="black", alpha=0.2, zorder=2, label="Q5 to Q95")
ax_twin.set_ylim(0, 6000)
ax_twin.set_ylabel("Sediment Yield\n" + r"[$\mathrm{m}^3$ / $\mathrm{10-min}$]", fontweight='bold', color="black")
rewrite_x_ticks(ax=ax_twin, data_start=t1, data_end=t2, data_sps=1/600, x_interval=1)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=6, ncol=1)
# endregion


# region
ax = plt.subplot(gs[1])
ax.set_title(label=f"(b)", loc="left", fontsize=7, fontweight='bold')
# raw data sampling frequency as 0.1 s, m**3 / s
threshold = 30 # m**3 / s
data_map = {"Gazoduc":"fig3a.csv", "CD27":"fig3b.csv", "CD29":"fig3c.csv"}
t1_pd = pd.to_datetime(t1)
t2_pd = pd.to_datetime(t2)

full_time = pd.date_range(start=t1_pd, end=t2_pd, freq="100ms")
x_raw = (full_time - t1_pd).total_seconds()

x = np.arange(len(full_time))
for label, filename in data_map.items():
    discharge_path = Path(project_root) / f"data/open_access/aj2025/Figure3/{filename}"
    discharge = pd.read_csv(discharge_path, header=0)
    
    discharge["Time"] = pd.to_datetime(discharge["Time"],format="%d-%b-%Y %H:%M:%S.%f")
    discharge["Time"] = discharge["Time"].dt.round("100ms")
    
    discharge = discharge.set_index("Time").sort_index()
    discharge = discharge.groupby(discharge.index).mean()
    df_win = discharge.reindex(full_time)
    
    ax.plot(x_raw, df_win["discharge"], label=f"Measured at: {label}", alpha=0.75)
    
    
    mask = df_win["discharge"] >= threshold
    if mask.any():
        first_time = df_win.index[mask.argmax()] # type: ignore
        first_value = df_win.loc[first_time, "discharge"]
        print(label, first_time, first_value)
    else:
        print(label, "never reaches threshold")

ax.set_xlim(0, (t2_pd - t1_pd).total_seconds())
ax.set_ylim(0, 150)
rewrite_x_ticks(ax=ax, data_start=t1, data_end=t2, data_sps=1, x_interval=1)
ax.set_ylabel("Measured Discharge\n" + r"[$\mathrm{m}^3$ / $s$]", fontweight='bold', color="black")


ax_twin = ax.twinx()
key = "Qs"
hydro_mve_path = Path(project_root) / f"pipeline/run_2004_2025_posterior/v0dot4/theta_001/hydro_output.nc"
hydro_mve = xr.load_dataset(hydro_mve_path)

mask = (hydro_mve.time_str >= t1) & (hydro_mve.time_str <= t2)
hydro_mve = hydro_mve.isel(time=mask)

data = hydro_mve[key].values
data = unit_converter(input=data, catchment_area=4.83, method="area-aggregated") # return as m^3 per 10 minutes
print(f"np.max(data) = {np.max(data)} [m**3 per 10 minutes], np.max(data) / 600 = {np.max(data) / 600 } [m**3 per second]")

hydro_time = pd.to_datetime(hydro_mve.time_str.values)
x_qs = (hydro_time - t1_pd).total_seconds()
x = np.arange(len(data))
ax_twin.plot(x_qs, data, color="black", label="Modelled", alpha=0.75)
ax_twin.set_ylabel("Modeled Surface Discharge\n" + r"[$\mathrm{m}^3$ / $\mathrm{10-min}$]", fontweight='bold', color="black")
ax_twin.set_ylim(0, 1.5 * 1e3)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax_twin.get_legend_handles_labels()
ax_twin.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=6, ncol=1)
# endregion


# region
ax = plt.subplot(gs[2])
ax.set_title(label=f"(c)", loc="left", fontsize=7, fontweight='bold')
cbar_ax = plt.subplot(gs[3])

st_path = Path(project_root) / f"data/open_access/qz2022/cooked_2022-06-05.mseed"
st = read(st_path)
st = st.trim(UTCDateTime(t1), UTCDateTime(t2))

pro_path = Path(project_root) / f"data/open_access/qz2022/model_prediction_2022-06-05.txt"
pro = pd.read_csv(pro_path, header=0)
time_str = pro.iloc[:, 1]
id1 = np.where(time_str == t1)[0][0]
id2 = np.where(time_str == t2)[0][0] + 1

pro_time = pro.iloc[id1:id2, 1]
pro_mean = np.array(pro.iloc[id1:id2, -2], dtype=float)
pro_std = np.array(pro.iloc[id1:id2, -1], dtype=float)

t = 55
print(pro_time.values[t], pro_mean[t])
t = 56
print(pro_time.values[t], pro_mean[t])


ax, data_sps = psd_plot(fig, ax, cbar_ax, st, fix_colorbar=True, per_lap=0.5, wlen=60, x_interval=1)
ax.set_xlabel("UTC+0 Time", fontweight='bold', color="black")

ax_prob = ax.twinx()
x = np.array([UTCDateTime(t) - t0 for t in pro_time], dtype=float)
ax_prob.plot(x, pro_mean, color="white", zorder=10, linewidth=2, label="Probability")
ax_prob.set_ylim(0, 1)
ax_prob.set_ylabel("Debris Flow Probability", fontweight='bold', color="black")
ax_prob.legend(loc="upper right", fontsize=6, ncol=1)
# endregion


png_path = Path(current_dir) / f"seismic_vs_yield_2022.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
