#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-19T17:25:37
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import argparse

import numpy as np
import xarray as xr
import pandas as pd

from obspy import UTCDateTime
import matplotlib.pyplot as plt

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the func. from the same folder
from func.generator.main_generator import s2d_workflow


df = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2023_2026_t.txt", 
                 header=0, nrows=157824)
sum_2023_2025 = np.sum(df.iloc[:, 2])

temp = {}

storm2drought_ratio = 0.05
for cycle_period in [30, 45, 60, 75, 90, 105, 120]:
    for storm_onset_month in np.arange(1, 10):
        
        file_format, data = s2d_workflow(
            year_list=(2023, 2024, 2025),
            cycle_period=int(cycle_period),
            storm2drought_ratio=float(storm2drought_ratio),
            storm_onset_month=int(storm_onset_month),
            storm_onset_day=1,
            sigma_scale=3,
            plot=False,
            seed=None,
            ref_history=True,
        )
        
        sum_data = np.sum(data[:, 2])
        temp[f"{int(cycle_period)}__{int(storm_onset_month)}"] = sum_data / sum_2023_2025



data = temp
# split keys into x (left) and y (right)
records = [{'y': int(k.split('__')[0]), 'x': int(k.split('__')[1]), 'val': v}
           for k, v in data.items()]
df = pd.DataFrame(records)

# pivot: rows = y, cols = x (so x runs along the horizontal axis)
grid = df.pivot(index='y', columns='x', values='val')

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(grid.values, cmap='viridis', aspect='auto', origin='lower')

ax.set_xticks(range(len(grid.columns)))
ax.set_xticklabels(grid.columns)
ax.set_yticks(range(len(grid.index)))
ax.set_yticklabels(grid.index)
ax.set_xlabel('x (left of __)')
ax.set_ylabel('y (right of __)')

# annotate cell values
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        val = grid.values[i, j]
        if not pd.isna(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()