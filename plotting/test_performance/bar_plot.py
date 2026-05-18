#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

#region ### add the sys.path to search for custom modules ###
from pathlib import Path

from numpy.core.records import record
from obspy import UTCDateTime

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.toolkit.round_timestamp import round_time
from func.toolkit.loss_func import likehood_loss
from func.toolkit.physical_unit_converter import unit_converter

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

# load the observed debris flow volume
event_catalog = pd.read_csv(f"{project_root}/data/event_catalog/debris_flow_volume_2004_2022.txt",
                            skiprows=6, header=0)
event_catalog = event_catalog.iloc[:63, :]
volume = event_catalog.iloc[:, 2].values
idx = []
for i in range(len(volume)):
    volume_obs = volume[i]
    if np.isnan(volume_obs):
        pass
    else:
        idx.append(i)

y_obs = event_catalog.iloc[idx, :]
t_s, t_e = [], []
for event_id in range(len(y_obs)):
    t1 = round_time(y_obs.iloc[event_id, 0])
    t_s.append(t1)

    t2 = round_time(y_obs.iloc[event_id, 1])
    t_e.append(t2)
y_obs.iloc[:, 0] = t_s
y_obs.iloc[:, 1] = t_e




# (1) calaulate the loss for default
# this result is use the default params and data
sed_container = xr.open_dataset(f"{project_root}/functions/SedCas_default/sed_container_2004_2018.nc")
sed_output = xr.open_dataset(f"{project_root}/functions/SedCas_default/sed_output_2004_2018.nc")

dfs = sed_container["dfs"]
catchment_area = 4.83  # unit is km^2
y_pred = unit_converter(input=dfs, catchment_area=catchment_area, method="area-aggregated")
buffer_time = 1
default_loss = 1e10
total_loss, details_loss, num_failured_pred = likehood_loss(y_obs, y_pred, buffer_time, default_loss)

## this is based on the all interations
# select the ratio of predicted volume / obsvered volume
temp_l = details_loss[1:] # first column is meta
predicted_gap1 = []
for i in range(len(temp_l)):
    temp = temp_l[i]
    ratio = float(temp.split(", ")[-1])
    predicted_gap1.append(ratio)

## this is based on the Q50
dfs_Q50 = sed_output["dfs_Q50"]
dfs_Q50 = unit_converter(input=dfs_Q50, catchment_area=catchment_area, method="area-aggregated")
predicted_gap2 = []
for i in range(len(y_obs)):
    t1, t2 = y_obs.iloc[i, 0], y_obs.iloc[i, 1]
    volume_obs = y_obs.iloc[i, 2]

    if np.isnan(volume_obs):
        continue
    else:
        temp = dfs_Q50.copy()
        mask = (temp.time_str >= t1) & (temp.time_str < t2)
        temp = temp.isel(time=mask)
        predicted_volume = np.sum(temp.values)

        if predicted_volume == 0:
            predicted_volume = np.nan

        ratio = predicted_volume / volume_obs
        predicted_gap2.append(ratio)



# plot it
x = np.arange(1, len(predicted_gap1)+1)
width = 0.25

fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

ax.bar(x - width / 2, predicted_gap1, width, color="C0", alpha=0.7, label="1h Default", zorder=2)
ax.bar(x + width / 2, predicted_gap2 , width, color="C1", alpha=0.7, label="1h from 2004", zorder=2)

ax.set_xlim(1-width, x[-1]+width)
ax.set_xticks([1, 10, 20, 30, 40, 50, 60], [1, 10, 20, 30, 40, 50, 60])
ax.legend(loc="best", fontsize=6)
ax.set_yscale('log')
ax.grid(axis='y', color='red', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

ax.set_ylabel("Volume Pre. / OBs.", fontweight='bold')
ax.set_xlabel("Event Index [from 2004 to 2017]", fontweight='bold')
ax.set_title(f"")

plt.tight_layout()
plt.savefig(f"{current_dir}/error_ratio.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)
