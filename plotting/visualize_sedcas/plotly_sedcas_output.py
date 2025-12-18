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
from functions.seismic.cal_ES import load_ES_energy

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-2, 3),
                     'axes.formatter.use_mathtext': True})

pio.templates.default = "plotly_white"
# pio.templates["plotly_white"].layout.font.size = 7
pio.renderers.default = "chrome"

# <editor-fold desc="load the dataset">
# selected_period = ["2017-05-18T00:00:00", "2017-07-01T00:00:00"]
# selected_period = ["2018-06-01T00:00:00", "2018-08-15T00:00:00"]
# selected_period = ["2019-05-25T00:00:00", "2019-08-25T00:00:00"]
selected_period = ["2020-05-29T00:00:00", "2020-09-25T00:00:00"]
# selected_period = ["2022-06-01T00:00:00", "2022-09-15T00:00:00"]

# <editor-fold desc="load CasSed in-out">
df0 = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2017_2025.txt")
df0['timestamp [UTC+0]'] = df0['timestamp [UTC+0]'].str.replace(' ','T')

id1 = df0.index[df0['timestamp [UTC+0]'] == selected_period[0]][0]
id2 = df0.index[df0['timestamp [UTC+0]'] == selected_period[1]][0] + 1
df0 = df0.iloc[id1:id2]



df1 = pd.read_csv(f"{project_root}/data/SedCas_output/Hydro_2017-2025.txt")
df1['timestamp [UTC+0]'] = df1['timestamp [UTC+0]'].str.replace(' ','T')

id1 = df1.index[df1['timestamp [UTC+0]'] == selected_period[0]][0]
id2 = df1.index[df1['timestamp [UTC+0]'] == selected_period[1]][0] + 1
df1 = df1.iloc[id1:id2]


df2 = pd.read_csv(f"{project_root}/data/SedCas_output/Sediment_2017-2025.txt")
df2['timestamp [UTC+0]'] = df2['timestamp [UTC+0]'].str.replace(' ','T')

id1 = df2.index[df2['timestamp [UTC+0]'] == selected_period[0]][0]
id2 = df2.index[df2['timestamp [UTC+0]'] == selected_period[1]][0] + 1
df2 = df2.iloc[id1:id2]


# load ES
# network, station, channel, year = "9S", "ILL12", "EHZ", 2018
# for idx, julday in enumerate(range(145, 250+1)):
#
#     output_temp, output_header = load_ES_energy(network, station, channel, year, julday)
#
#     if idx == 0:
#         arr = output_temp
#     else:
#         arr = np.vstack((arr, output_temp))
#
# df3 = pd.DataFrame(arr, columns=output_header)
# df3 = df3.rename(columns={"t_str": "timestamp [UTC+0]"})
# df3['timestamp [UTC+0]'] = (
#     df3['timestamp [UTC+0]']
#         .str.replace(' ', 'T')
#         .str.replace('+00:00', '')
# )
#
# id1 = df3.index[df3['timestamp [UTC+0]'] == selected_period[0]][0]
# id2 = df3.index[df3['timestamp [UTC+0]'] == selected_period[1]][0] + 1
# df3 = df3.iloc[id1:id2]

# </editor-fold>


# - 36  # MeanFFT
# - 37  # MaxFFT
# - 38  # FmaxFFT
# - 39  # FCentroid

feature_name = "MaxFFT" #"env_max_to_duration"#'ES_1' # df3.columns
df3 = pd.read_csv(f"{project_root}/data/seismic_temp/seis_energy/{selected_period[0][:4]}_ILL12_EHZ_all_B.txt", header=0)
df3 = df3.rename(columns={"time_window_start": "timestamp [UTC+0]"})
df3 = df3.rename(columns={feature_name: feature_name})
id1 = df3.index[df3['timestamp [UTC+0]'] == selected_period[0]][0]
id2 = df3.index[df3['timestamp [UTC+0]'] == selected_period[1]][0] + 1
df3 = df3.iloc[id1:id2]
# </editor-fold>


fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

# Panel 1
fig.add_trace(
    go.Scatter(
        x=df0['timestamp [UTC+0]'],
        y=df0['precipitation [mm per Hourly]'],
        mode='lines',
        line=dict(color='black', width=2),
        name='Precipitation'
    ),
    row=1, col=1
)

# Panel 2
fig.add_trace(
    go.Scatter(
        x=df1['timestamp [UTC+0]'],  # make sure column names match
        y=df1['Q'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Q'
    ),
    row=2, col=1
)


# Panel 3
fig.add_trace(
    go.Scatter(
        x=df2['timestamp [UTC+0]'],  # make sure column names match
        y=df2['Qstl'],
        mode='lines',
        line=dict(color='brown', width=2),
        name='Qstl'
    ),
    row=3, col=1
)


# Panel 4
fig.add_trace(
    go.Scatter(
        x=df3['timestamp [UTC+0]'],  # make sure column names match
        y=df3[feature_name].astype(float),
        mode='lines',
        line=dict(color='red', width=2),
        name=feature_name
    ),
    row=4, col=1
)


# update layout
fig.update_layout(
    updatemenus=[dict(type="buttons",
                      showactive=False,
                      buttons=[dict(label="Reset Y",
                                    method="relayout",
                                    args=[{"yaxis.autorange": True}])],
                      x=1.1,
                      y=1.05)],
    title="Precipitation [mm/hr]",
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
    title_text="Precipitation [mm/hr]",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
    griddash="dash",
    layer="below traces",
    row=1,
    col=1
)


fig.update_xaxes(
    title_text="Time [UTC+0]",
    tickformat="%Y-%m-%dT%H:%M:%S",
    hoverformat="%Y-%m-%dT%H:%M:%S",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
    griddash="dash",
    layer="below traces",
    row=2,
    col=1
)

fig.update_yaxes(
    title_text="Sediemnt [mm/hr]",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
    griddash="dash",
    layer="below traces",
    row=2,
    col=1
)

# Panel 3 X-axis
fig.update_xaxes(
    title_text="Time [UTC+0]",
    tickformat="%Y-%m-%dT%H:%M:%S",
    hoverformat="%Y-%m-%dT%H:%M:%S",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",
    griddash="dash",
    layer="below traces",
    row=3,
    col=1
)

# Panel 3 Y-axis
fig.update_yaxes(
    title_text="Qstl",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",
    griddash="dash",
    layer="below traces",
    row=3,
    col=1
)

# Panel 4 X-axis
fig.update_xaxes(
    title_text="Time [UTC+0]",
    tickformat="%Y-%m-%dT%H:%M:%S",
    hoverformat="%Y-%m-%dT%H:%M:%S",
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",
    griddash="dash",
    layer="below traces",
    row=4,
    col=1
)

# Panel 4 Y-axis
fig.update_yaxes(
    title_text=feature_name,
    showgrid=True,
    gridwidth=1,
    gridcolor="rgba(128,128,128,0.5)",
    griddash="dash",
    layer="below traces",
    row=4,
    col=1
)


# show in browser
fig.show(renderer="browser")
