#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-22T09:12:12
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import json

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the custom functions
# Do not need


font_global = dict(family="Arial, sans-serif", size=12, color="#2d2d2d")

def plotly_multi_time_series_xr(xr_dataset, 
                                list_of_col_names, 
                                vars_dict,
                                xr_dataset_whatif=None):

    pio.templates.default = "plotly_white"
    pio.renderers.default = "browser"

    if list_of_col_names is None:
        x_col = "time_str"
        list_of_col_names = [(x_col, y_col) for y_col in xr_dataset.data_vars]

    n = len(list_of_col_names)

    subplot_titles = []
    subplot_titles_abbr = []
    for x_col, y_col in list_of_col_names:
        y_temp = vars_dict[y_col]
        y_abbr, y_label = str(y_temp).split(": ")
        
        subplot_titles.append(y_label)
        subplot_titles_abbr.append(y_abbr)

    fig = make_subplots(rows=n, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=subplot_titles)

    # color palette — one color per subplot
    colors = px.colors.qualitative.Plotly  # or any palette you like

    for i, (x_col, y_col) in enumerate(list_of_col_names, start=1):
        color = colors[i % len(colors)]

        # solid line — monitoring
        fig.add_trace(
            go.Scatter(
                x=xr_dataset[x_col].values,
                y=xr_dataset[y_col].values,
                mode="lines",
                name=f"{subplot_titles[i-1]} (monitoring)",
                line=dict(color=color, dash="solid"),
            ),
            row=i, col=1
        )

        # dashed line — whatif (only if provided)
        if xr_dataset_whatif is not None:
            y_col_whatif = y_col
            fig.add_trace(
                go.Scatter(
                    x=xr_dataset_whatif["time_str"].values,
                    y=xr_dataset_whatif[y_col_whatif].values,
                    mode="lines",
                    name=f"{subplot_titles[i-1]} (whatif)",
                    line=dict(color="rgba(0,0,0,0.7)", dash="dash"),
                ),
                row=i, col=1
            )

        fig.update_yaxes(
            title_text=subplot_titles_abbr[i-1], row=i, col=1, showgrid=True,
            gridcolor="rgba(128,128,128,0.5)", griddash="dash"
        )

    rangeselector = dict(buttons=[
        dict(count=1, label="1d", step="day", stepmode="backward"),
        dict(count=7, label="1w", step="day", stepmode="backward"),
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ])

    fig.update_xaxes(
        tickformat="%Y-%m-%dT%H:%M:%S", hoverformat="%Y-%m-%dT%H:%M:%S",
        showgrid=True, gridcolor="rgba(128,128,128,0.5)", griddash="dash",
        rangeselector=rangeselector,
    )
    fig.update_xaxes(title_text="Time [UTC+0]", row=n, col=1)

    last_metro_swiss = f"Latest MetroSwiss Data: {xr_dataset.coords['time_str'].values[-1]} [UTC+0]"
    json_path = Path(project_root) / f"deploy/liveshow_cache/monitoring/last_stTwin_update.json"
    with open(json_path, "r") as f:
        temp = json.load(f)
    last_update = temp["last_update"]

    fig.update_layout(
        autosize=True,
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=f"{last_metro_swiss}<br>{last_update}", x=0.5, xanchor="center", font=dict(size=12)),
        font=font_global,
    )

    return fig
