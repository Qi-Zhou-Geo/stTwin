#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-18T00:44:35
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


# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import custom func.
# from func.SedCas.mass_balance_checker import mass_balance_checker
from func.generator.main_generator import workflow

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})


cycle_period_l = [120]
storm2drought_ratio = 0.05
storm_onset_month = 1
for cycle_period in cycle_period_l:
    file_format = workflow(
        year_list=(2023, 2024, 2025),
        cycle_period=int(cycle_period),
        storm2drought_ratio=float(storm2drought_ratio),
        storm_onset_month=int(storm_onset_month),
        storm_onset_day=1,
        plot=True
    )
    print(file_format)



 
s_l = ["2023-01-01T00:00:00", "2024-01-01T00:00:00", "2025-01-01T00:00:00"]
e_l = ["2024-01-01T00:00:00", "2025-01-01T00:00:00", "2025-12-31T23:50:00"]

temp_l = []
obs_precp_l = []

for cycle_period in cycle_period_l:
    
    df_path = Path(project_root) / f"data/SedCas_whatif_input/climate_2023_2026_t_whatif_CP={cycle_period}_R={storm2drought_ratio}_M={storm_onset_month}_D=1.txt"
    df = pd.read_csv(df_path, header=0)
    
    date_str = df.iloc[:, 1]
    s, e = s_l[0], e_l[0]
    id1 = np.where(date_str == s)[0][0]
    id2 = np.where(date_str == e)[0][0]
    
    precp_sum = np.sum(df.iloc[id1:id2, 2])
    temp_l.append(precp_sum)
    
    
    if cycle_period == 15:
        df_path = Path(project_root) / f"data/SedCas_input/climate_2004_2025_t.txt"
        df = pd.read_csv(df_path, header=0)

        date_str = df.iloc[:, 1]
        
        for s, e in zip(s_l, e_l):
            id1 = np.where(date_str == s)[0][0]
            id2 = np.where(date_str == e)[0][0]

            obs_precp = np.sum(df.iloc[id1:id2, 2])
            obs_precp_l.append(obs_precp)


fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

x = np.arange(len(cycle_period_l))
ax.bar(x, temp_l, color="C0", linewidth=1, zorder=1)

# ax.set_ylim(0, 2000)
ax.set_xticks(x, cycle_period_l) # type: ignore

ax.axhline(y=513.8, color="C1", linestyle="--", label="Historical Minimum (513.8 mm)", zorder=2)
ax.axhline(y=919.7, color="C2", linestyle="--", label="Historical Mean (919.7 mm)", zorder=2)
ax.axhline(y=1608.6, color="C3", linestyle="--", label="Historical Maximum (1608.6 mm)", zorder=2)

ls = ["--", "-", "-."]
for idx, year in enumerate([2023, 2024, 2025]):
    obs_precp = obs_precp_l[idx]
    ax.axhline(y=obs_precp, color=f"black", linestyle=ls[idx], label=f"Obs. of {year} ({obs_precp:.1f} mm)", zorder=2)


ax.legend(fontsize=6)

ax.set_xlabel("Cycle Period [Days]", fontweight="bold")
ax.set_ylabel("Annual Precipitation [mm]", fontweight="bold")

png_path = Path(current_dir) / f"plots/annual_precipitation_example.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)
