#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-12-15
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions


plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-2, 3),
                     'axes.formatter.use_mathtext': True})

pio.templates.default = "plotly_white"
# pio.templates["plotly_white"].layout.font.size = 7
pio.renderers.default = "chrome"


# <editor-fold desc="load CasSed in-out">
files_list = ["2013_IGB02_HHZ_all_B.txt", "2014_IGB02_HHZ_all_B.txt", "2017_ILL02_EHZ_all_B.txt",
              "2018_ILL12_EHZ_all_B.txt", "2019_ILL12_EHZ_all_B.txt", "2020_ILL12_EHZ_all_B.txt",
              "2022_ILL12_EHZ_all_B.txt"]
feature_name = "env_max_to_duration" #'ES_1', "env_max_to_duration, "MeanFFT"

for idx, file in enumerate(files_list):

    df = pd.read_csv(f"{project_root}/data/seismic_temp/seis_energy/{file}", header=0)
    df = df.rename(columns={"time_window_start": "timestamp [UTC+0]"})

    if idx == 0:
        temp_df = df
    else:
        temp_df = pd.concat([temp_df, df], axis=0)
# </editor-fold>


df1 = temp_df

fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

# Panel 1
fig.add_trace(
    go.Scatter(
        x=df1['timestamp [UTC+0]'],
        y=df1[feature_name],
        mode='lines',
        line=dict(color='black', width=2),
        name=feature_name
    ),
    row=1, col=1
)

# update layout
fig.update_layout(
    title=f"Feature {feature_name} VS Flow-Alert Prediction",
    xaxis=dict(tickformat="%Y-%m-%dT%H:%M:%S"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    autosize=True,
    width=2000,
    height=800,  # adjust for 4 stacked panels
)

# update axes if needed
fig.update_xaxes(
    title_text="Time [UTC+0]",
    tickformat="%Y-%m-%dT%H:%M:%S",
    hoverformat="%Y-%m-%dT%H:%M:%S",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
    griddash="dash",
    layer="below traces",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text=feature_name,
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
    griddash="dash",
    layer="below traces",
    row=2,
    col=1
)

# show in browser
fig.show(renderer="browser")
