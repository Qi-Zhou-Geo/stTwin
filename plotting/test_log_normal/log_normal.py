#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from scipy.stats import norm
import scipy.stats as stats
from scipy.stats import t

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.toolkit.round_timestamp import round_time
from functions.toolkit.loss_func import likehood_loss, ratio_loss
from functions.toolkit.physical_unit_converter import unit_converter

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})


# <editor-fold desc="(0) load the observed debris flow volume">
event_catalog = pd.read_csv(f"{project_root}/"
                            f"data/event_catalog/debris_flow_volume_2000_2022.txt",
                            skiprows=6, header=0)

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

df_volume = []
for volume in y_obs["Volume[m3]"]:
    if np.isnan(volume):
        pass
    else:
        df_volume.append(volume)

# </editor-fold>


df_volume = np.array(df_volume)
df_volume_log = np.log10(df_volume)


# log-normal
fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 2)

# histogram plot
ax = plt.subplot(gs[0])
mu, sigma = np.mean(df_volume_log), np.std(df_volume_log, ddof=1)
plt.hist(df_volume_log, color="black", bins=20, density=True, alpha=0.6)

x = np.linspace(min(df_volume_log), max(df_volume_log), 200)
plt.plot(x, norm.pdf(x, mu, sigma), color="C0")

ax.set_ylabel("Probability Density", fontweight='bold')
ax.set_xlabel("Log10(volume)", fontweight='bold')

# Q-Q plot
ax = plt.subplot(gs[1])
res = stats.probplot(df_volume_log, dist="norm", plot=ax)
lines = ax.get_lines()
lines[0].set_markerfacecolor('black')
lines[0].set_markeredgecolor('black')
lines[0].set_alpha(0.5)
lines[1].set_color('C0')
slope, intercept, r = res[-1]
ax.text(-2, 5, f"$R^2 = {r**2:.4f}$")

ax.set_title("")

ax.set_ylabel("Ordered Log10(volume)", fontweight='bold')
ax.set_xlabel("Theoretical Quantiles", fontweight='bold')


plt.tight_layout()
plt.savefig(f"{current_dir}/test_log_normal.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)



# student-t
fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 2)

# histogram plot
ax = plt.subplot(gs[0])
mu, sigma = np.mean(df_volume_log), np.std(df_volume_log, ddof=1)
plt.hist(df_volume_log, color="black", bins=20, density=True, alpha=0.6)

x = np.linspace(min(df_volume_log), max(df_volume_log), 200)
plt.plot(x, norm.pdf(x, mu, sigma), color="C0")

ax.set_ylabel("Probability Density", fontweight='bold')
ax.set_xlabel("Log10(volume)", fontweight='bold')

# Q-Q plot
ax = plt.subplot(gs[1])
res = stats.probplot(df_volume_log, dist="norm", plot=ax)
lines = ax.get_lines()
lines[0].set_markerfacecolor('black')
lines[0].set_markeredgecolor('black')
lines[0].set_alpha(0.5)
lines[1].set_color('C0')
slope, intercept, r = res[-1]
ax.text(-2, 5, f"$R^2 = {r**2:.4f}$")

ax.set_title("")

ax.set_ylabel("Ordered Log10(volume)", fontweight='bold')
ax.set_xlabel("Theoretical Quantiles", fontweight='bold')


plt.tight_layout()
plt.savefig(f"{current_dir}/test_log_normal.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)