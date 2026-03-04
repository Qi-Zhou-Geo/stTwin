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


file = f"/Users/qizhou/#python/Flow-Alert/pipeline/cross_catchments_test/output/Illgraben-9S-2022-ILL12-EHZ-F-testing-True-v3model-F-b=-256-s-64-9.txt"
df = pd.read_csv(file, header=0)

column1 = "t_str"
column2 = "pro_mean"

fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

# Panel 1
fig.add_trace(
    go.Scatter(
        x=df[column1],
        y=df[column2],
        mode='lines',
        line=dict(color='black', width=2),
        name=column2
    ),
    row=1, col=1
)

# update layout
fig.update_layout(
    title=f"{file}",
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
    title_text=column2,
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
