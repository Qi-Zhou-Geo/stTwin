#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

from numpy.core.records import record
from obspy import UTCDateTime

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# </editor-fold>

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

df1 = pd.read_csv(f'{current_dir}/default_results.txt', header=None) # "1h Default"
df2 = pd.read_csv(f'{current_dir}/default_results2.txt', header=None) # "1h from 2004"
df3 = pd.read_csv(f'{current_dir}/default_results_BO.txt', header=None) # "1h from 2004"

fill_nan = 1e4
ratio1 = pd.to_numeric(df1.iloc[:63, -1], errors="coerce") \
         .replace([np.inf, -np.inf, np.nan], fill_nan) \
         .to_numpy()

ratio2 = pd.to_numeric(df2.iloc[:63, -1], errors="coerce") \
         .replace([np.inf, -np.inf, np.nan], fill_nan) \
         .to_numpy()

ratio3 = pd.to_numeric(df3.iloc[:63, -1], errors="coerce") \
         .replace([np.inf, -np.inf, np.nan], fill_nan) \
         .to_numpy()

color_1 = []
for i in ratio1:
    if i == fill_nan:
        color_1.append('black')
    else:
        color_1.append('C0')

color_2 = []
for i in ratio2:
    if i == fill_nan:
        color_2.append('black')
    else:
        color_2.append('C1')

x = np.arange(len(ratio1))
width = 0.25

fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

ax.bar(x - width/2, ratio1, width, color=color_1, alpha=0.7, label="1h Default", zorder=2)
ax.bar(x + width/2, ratio2, width, color=color_2, alpha=0.7, label="1h from 2004", zorder=2)

ax.legend(loc="best", fontsize=6)
ax.set_yscale('log')
ax.set_ylim(1, fill_nan)
ax.grid(axis='y', color='red', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

ax.set_ylabel("Volume Obs / Pre", fontweight='bold')
ax.set_xlabel("Event Index [from 2004 to 2017]", fontweight='bold')

plt.tight_layout()
plt.savefig(f"{current_dir}/error_ratio.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)

