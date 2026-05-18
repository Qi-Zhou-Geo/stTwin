#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd
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


# import the custom functions
from func.visulize.plotly_visualize import plotly_multi_time_series

df1 = pd.read_csv(f"{current_dir}/Hydro.out", header=0)
df2 = pd.read_csv(f"{current_dir}/Sediment.out", header=0)


time_coord = "time_str"
t1, t2 = "2000-01-01 00:00:00", "2017-10-31 00:00:00"

date1 = df1.iloc[:, 0]
id1 = np.where(date1 == t1)[0][0]
id2 = np.where(date1 == t2)[0][0]
df1 = df1.iloc[id1:id2, :]

date2 = df2.iloc[:, 0]
id1 = np.where(date2 == t1)[0][0]
id2 = np.where(date2 == t2)[0][0]
df2 = df2.iloc[id1:id2, :]

list_of_tuples = [(df1, "D", "Q"), (df1, "D", "Qs"), (df2, "D", "Q50")]


fig = plotly_multi_time_series(list_of_tuples)
fig.show()
