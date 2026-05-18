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
from func.toolkit.loss_func import likehood_loss, ratio_loss
from func.toolkit.physical_unit_converter import unit_converter

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})



# (0) load the observed debris flow volume
event_catalog = pd.read_csv(f"{project_root}/"
                            f"data/event_catalog/debris_flow_volume_2004_2022.txt",
                            skiprows=6, header=0)
event_catalog = event_catalog.iloc[:63, :]

# round the time
y_obs = event_catalog
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
sed_output = xr.open_dataset(f"{project_root}/functions/SedCas/10min/sed_output_2004_2018.nc")
sed_transport_real = sed_output["sed_transport_real_Q50"] # mean values
sed_transport_real = sed_transport_real.to_dataframe().reset_index()
y_pred = sed_transport_real[['time_str', 'sed_transport_real_Q50']]

# convert the volume unit
catchment_area = 4.83  # unit is km^2
volume = unit_converter(input=y_pred.iloc[:, 1], catchment_area=catchment_area, method="area-aggregated")
y_pred.iloc[:, 1] = volume

# calculate the loss
predicted_gap = ratio_loss(y_obs, y_pred, buffer_time=3, ratio_no_obs=np.nan, ratio_no_prediction=np.inf)





# (2) plot it
x = np.arange(1, len(predicted_gap)+1)
width = 0.25

fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

num_no_prediction = 0
for key, value in predicted_gap.items():
    ratio = value[-1]

    if np.isnan(ratio):
        # no observed volume
        ratio = 1
        color = "black"
        label = "No Observation"
    elif np.isinf(ratio):
        # no detection
        ratio = 1
        color = "red"
        label = "No Volume Prediction"
        num_no_prediction = num_no_prediction + 1
    else:
        # succssed
        color = "C0"
        label = "With Volume Prediction"

    ax.bar(key+1, ratio, width, color=color, alpha=0.7, label=label, zorder=2)

ax.set_title(label=f"No Volume Prediction: {num_no_prediction}", fontsize=7)
ax.set_ylim(1e-2, 1e2)
ax.set_xlim(0, 70)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=3)

ax.set_yscale('log')
ax.grid(axis='y', color='red', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_xticks([1, 10, 20, 30, 40, 50, 60], [1, 10, 20, 30, 40, 50, 60])

ax.set_ylabel("Ratio of Predicted to Observed Volume", fontweight='bold')
ax.set_xlabel("Event Index [from 2004 to 2017]", fontweight='bold')

plt.tight_layout()
plt.savefig(f"{current_dir}/error_ratio_BO_10min.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)


# calculate the likehood loss
sed_output = xr.open_dataset(f"{project_root}/functions/SedCas/10min/sed_container_2004_2018.nc")
sed_transport_real = sed_output["sed_transport_real"] # mean values
# convert the volume unit
catchment_area = 4.83  # unit is km^2
sed_transport_real = unit_converter(input=sed_transport_real, catchment_area=catchment_area, method="area-aggregated")
y_pred = sed_transport_real
total_loss, details_loss = likehood_loss(y_obs, y_pred, buffer_time=3, loss_no_obs=np.nan, low_value=0, high_value=1e4)

