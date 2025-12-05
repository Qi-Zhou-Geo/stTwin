#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import pandas as pd
import numpy as np

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

import plotly.express as px
import plotly.io as pio

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>


plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-2, 3),
                     'axes.formatter.use_mathtext': True})

pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.size = 7
fig.update_yaxes(exponentformat="power")  # show scientific notation
fig.update_xaxes(exponentformat="power")

# selected_period = ["2017-05-01 00:00:00", "2017-07-01 00:00:00"]
selected_period = ["2018-06-01 00:00:00", "2018-08-15 00:00:00"]

df1 = pd.read_csv(f"{project_root}/data/SedCas_output/Hydro_2017-2025.txt")
df1_arr = np.array(df1)
df1_label = df1.columns

df2 = pd.read_csv(f"{project_root}/data/SedCas_output/Sediment_2017-2025.txt")
df2_arr = np.array(df2)
df2_label = df2.columns

df_arr_selected = []
for idx, df in enumerate([df1_arr, df2_arr]):
    date = np.array(df1.iloc[:, 0])

    id1 = np.where(date == selected_period[0])[0][0]
    id2 = np.where(date == selected_period[1])[0][0]

    df_arr_selected.append(df[id1:id2, :])

df1_arr, df2_arr = df_arr_selected[0], df_arr_selected[1]

# model input
df3 = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2017_2025.txt")
df3_arr = np.array(df3)
df3_label = df3.columns
date = df3_arr[:, 1]
id1 = np.where(date == selected_period[0].replace(" ", "T"))[0][0]
id2 = np.where(date == selected_period[1].replace(" ", "T"))[0][0]
df3_arr = df3_arr[id1:id2, :]



fig = px.line(df3, x='timestamp [UTC+0]', y='precipitation [mm per Hourly]')

fig.update_traces(line_color='black', line_width=2)

fig.update_layout(
    title="Precipitation [mm/hr]",
    xaxis=dict(tickformat="%Y-%m-%dT%H:%M:%S"),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

fig.update_xaxes(
    title_text="Time [UTC+0]",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
    griddash="dash",
    layer="below traces",
)

fig.update_yaxes(
    title_text="Precipitation [mm/hr]",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
    griddash="dash",
    layer="below traces",
)

fig.write_image(f"./precipitation.png", width=550, height=300)
fig.show()
