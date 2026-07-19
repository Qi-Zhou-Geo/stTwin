#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-16T11:06:48
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
from func.visulize.visualize_seismic import psd_plot, rewrite_x_ticks, waveform_plot
from func.toolkit.physical_unit_converter import unit_converter


plt.rcParams.update({'font.size': 7, # 7pt 
                     'axes.formatter.limits': (-3, 3), # 1e-3 to 1e3 >> liner scale, otherwise, log scale
                     'axes.formatter.use_mathtext': True}) # scientific style 1 \times 10^3, rather than 1e3

t1 = "2026-07-15T09:00:00"
t0 = UTCDateTime(t1)
t2 = "2026-07-15T15:00:00"


fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(2, 1)


# region
ax = plt.subplot(gs[0])
# ax.set_title(label=f"(a)", loc="left", fontsize=7, fontweight='bold')

# precp
df_path = Path(project_root) / "deploy/liveshow_cache/climate/climate_2026_t.txt"
df = pd.read_csv(df_path, header=0)

time_str = df.iloc[:, 1].values
id1 = np.where(time_str==t1)[0][0]
id2 = np.where(time_str==t2)[0][0] + 1


mve_precp = np.array(df.iloc[id1:id2, 2])
x = np.arange(len(mve_precp))
ax.plot(x, mve_precp, color="C0", zorder=3, label="MetroSwiss Montana")

ax.set_yticks([0, 1, 2, 3, 4], labels=[0, 1, 2, 3, 4]) # type: ignore
ax.set_xlim(x[0], x[-1])
ax.set_ylabel("Precipitation\n[mm, 10-min total]", fontweight='bold')


# mve sed
key = "sed_transport_real"
sed_mve_path = Path(project_root) / f"deploy/liveshow_cache/monitoring/sed_output.nc"
sed_mve = xr.load_dataset(sed_mve_path)

mask = (sed_mve.time_str >= t1) & (sed_mve.time_str <= t2)
sed_mve = sed_mve.isel(time=mask)

q01 = sed_mve["sed_transport_real_Q1"].values
q50 = sed_mve["sed_transport_real_Q50"].values
q99 = sed_mve["sed_transport_real_Q99"].values
q01 = unit_converter(input=q01, catchment_area=4.83, method="area-aggregated") # return as m^3 per 10 minutes
q50 = unit_converter(input=q50, catchment_area=4.83, method="area-aggregated") # return as m^3 per 10 minutes
q99 = unit_converter(input=q99, catchment_area=4.83, method="area-aggregated") # return as m^3 per 10 minutes


ax_twin = ax.twinx()
ax_twin.plot(x, q50, color="black", zorder=4, label="Q50")
ax_twin.fill_between(x, q01, q99, color="black", alpha=0.2, zorder=2, label="Q1 to Q99")
ax_twin.set_ylim(0, 2e4)
ax_twin.set_ylabel("Modelled Sediment Yield\n" + r"[$\mathrm{m}^3$" + ", 10-min total]", fontweight='bold', color="black")
rewrite_x_ticks(ax=ax_twin, data_start=t1, data_end=t2, data_sps=1/600, x_interval=1)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=6, ncol=1)
# endregion


# region
ax = plt.subplot(gs[1])
# ax.set_title(label=f"(b)", loc="left", fontsize=7, fontweight='bold')

key = "Qs"
hydro_mve_path = Path(project_root) / f"deploy/liveshow_cache/monitoring/hydro_output.nc"
hydro_mve = xr.load_dataset(hydro_mve_path)

mask = (hydro_mve.time_str >= t1) & (hydro_mve.time_str <= t2)
hydro_mve = hydro_mve.isel(time=mask)

data = hydro_mve[key].values
data = unit_converter(input=data, catchment_area=4.83, method="area-aggregated") # return as m^3 per 10 minutes

x = np.arange(len(data))
ax.plot(x, data, color="black", label="Modelled: Surface Discharge", alpha=0.75)
ax.set_ylabel("Modelled Discharge\n" + r"[$\mathrm{m}^3$" + ", 10-min total]", fontweight='bold', color="black")
ax.set_ylim(0, 7 * 1e3)
rewrite_x_ticks(ax=ax, data_start=t1, data_end=t2, data_sps=1/600, x_interval=1)
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("UTC+0 Time", fontweight='bold')

# endregion


png_path = Path(current_dir) / f"sed_yield_2026.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
