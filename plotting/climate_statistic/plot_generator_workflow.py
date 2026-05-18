#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

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

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

data_resolution = "t"
climate_forcing_file = f"{project_root}/data/SedCas_input/climate_2004_2023_{data_resolution}.txt"
climate_forcing = pd.read_csv(climate_forcing_file, header=0)

precipitation = climate_forcing.iloc[:, 2].values # [mm per time_step]
temperature = climate_forcing.iloc[:, 3].values # [degree]
sun_radiation = climate_forcing.iloc[:, 4].values # [W per squared m]

temperature_min = math.floor(np.min(temperature))
temperature_max = math.floor(np.max(temperature))
temperature_delta = 0.1 # unit same as temperature

mapping = []
for t in np.arange(temperature_min,
                   temperature_max + temperature_delta,
                   temperature_delta):

    # (1) scane the temperature with range [t, t + delta_t]
    idx = np.where((temperature >= t) & (temperature <= t + temperature_delta))[0]

    # (2) calculate the mean and std for the given temperature range
    precipitation_mean = np.mean(precipitation[idx])
    precipitation_std = np.std(precipitation[idx], ddof=1)

    sun_radiation_mean = np.mean(sun_radiation[idx])
    sun_radiation_std = np.std(sun_radiation[idx], ddof=1)

    # pack together
    mapping.append([t, precipitation_mean, precipitation_std, sun_radiation_mean, sun_radiation_std])

mapping = np.array(mapping)
temperature_fluctuation = mapping[:, 0]
precipitation_fluctuation_mean = mapping[:, 1]
precipitation_fluctuation_std = mapping[:, 2]
sun_radiation_fluctuation_mean = mapping[:, 3]
sun_radiation_fluctuation_std = mapping[:, 4]

if data_resolution == "t":
    np.save(f"{project_root}/data/climate_statistic/10minutes/"
            f"map_temperature_to_precp_radiation.npy",
            mapping)
elif data_resolution == "h":
    np.save(f"{project_root}/data/climate_statistic/1h/"
            f"map_temperature_to_precp_radiation.npy",
            mapping)
else:
    raise ValueError("data_resolution must be 't' or 'h'")


fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(2, 1)


temperature_archive = (f"{project_root}/data/climate_statistic/10minutes/"
                       f"climate_forcing_daily_Temperature.npz")
# data_dict with key as julday from 1 to 365, and values with (mean, std)
temp = np.load(temperature_archive, allow_pickle=True)
data_dict = temp["data_dict"].item()
arr = np.array([[k, v[0], v[1]] for k, v in data_dict.items()])

mu = arr[:, 1]
sigma = arr[:, 2]
sigma_scaler = 1
lower_bound = mu - sigma_scaler * sigma  # mean - sigma_scaler * std
upper_bound = mu + sigma_scaler * sigma  # mean + sigma_scaler * std

ax = plt.subplot(gs[0])
x = np.arange(0, len(lower_bound))
ax.plot(x, arr[:, 1], color='black', label=r'Temperature $\mu \pm \sigma$')
ax.fill_between(x, lower_bound, upper_bound, color='black', alpha=0.3)
ax.set_ylabel("Temperature [Degree]", fontweight='bold')
ax.set_xlabel(f"Julday Day", fontweight='bold')



ax = plt.subplot(gs[1])
ax_twin = ax.twinx()

ax.plot(temperature_fluctuation, precipitation_fluctuation_mean, color='C0', label=r'Precipitation $\mu \pm \sigma$')
y1 = precipitation_fluctuation_mean - precipitation_fluctuation_std
y1 = np.clip(y1, a_min=0, a_max=np.nanmax(y1))
y2 = precipitation_fluctuation_mean + precipitation_fluctuation_std
ax.fill_between(temperature_fluctuation, y1, y2, color='C0', alpha=0.3)
ax.grid(axis='x', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.set_ylabel(f"Precipitation [mm per 10 minutes]", fontweight='bold')
ax.set_xlabel("Temperature [Degree]", fontweight='bold')
ax.set_xlim(-25, 35)

ax_twin.plot(temperature_fluctuation, sun_radiation_fluctuation_mean, color='black', label=r'Radiation $\mu \pm \sigma$')
y1 = sun_radiation_fluctuation_mean - sun_radiation_fluctuation_std
y1 = np.clip(y1, a_min=0, a_max=np.nanmax(y1))
y2 = sun_radiation_fluctuation_mean + sun_radiation_fluctuation_std
ax_twin.fill_between(temperature_fluctuation, y1, y2, color='black', alpha=0.3)
ax_twin.set_ylabel(f"Sun Radiation [W per squared m]", fontweight='bold')

# get handles and labels from both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6)

plt.tight_layout()
plt.savefig(f"{current_dir}/climate_{data_resolution}.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)
