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


def plotly_multi_time_series(list_of_tuples,
                             shared_title=None):
    """
    list_of_tuples: list of tuples
        [(df, x_col, y_col), ...]
    """

    pio.templates.default = "plotly_white"
    pio.renderers.default = "browser"

    n = len(list_of_tuples)
    if n == 0:
        raise ValueError("No series provided")

    subplot_titles = []
    for df, x_col, y_col in list_of_tuples:
        subplot_titles.append(y_col)

    fig = make_subplots(rows=n, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=subplot_titles)

    for i, (df, x_col, y_col) in enumerate(list_of_tuples, start=1):
        fig.add_trace(
            go.Scatter(x=df[x_col], y=df[y_col], mode="lines", name=y_col),
            row=i, col=1
        )

        fig.update_yaxes(title_text="111", row=i, col=1, showgrid=True,
                         gridcolor="rgba(128,128,128,0.5)", griddash="dash")

    fig.update_xaxes(tickformat="%Y-%m-%dT%H:%M:%S", hoverformat="%Y-%m-%dT%H:%M:%S",
                     showgrid=True, gridcolor="rgba(128,128,128,0.5)", griddash="dash")

    fig.update_layout(autosize=True,
                      showlegend=False, plot_bgcolor="white",
                      paper_bgcolor="white",
                      title=dict(text=shared_title, x=0.5, xanchor="center")
                      )

    fig.show()




def plotly_multi_time_series_xr(xr_dataset, list_of_col_names, shared_title=None):
    """
    list: list_of_col_names
        [(x1_col_name, y1_col_name),
        ...,
        (x2_col_name, y2_col_name)]
    """

    pio.templates.default = "plotly_white"
    pio.renderers.default = "browser"


    if list_of_col_names is None:
        print("No column name provided,\nall columns in the <xr_dataset> will be plotted.")
        x_col = "time_str"
        list_of_col_names = [(x_col, y_col) for y_col in xr_dataset.data_vars]
    else:
        pass
    n = len(list_of_col_names)

    # you can use this as y label or subplot title
    subplot_titles = []
    for x_col, y_col in list_of_col_names:

        unit = xr_dataset[y_col].attrs['units']
        y_col = y_col.replace("_", " ")
        y_col = y_col.title()
        y_label = f"{y_col} [{unit}]"

        subplot_titles.append(y_label)


    fig = make_subplots(rows=n, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03)

    for i, (x_col, y_col) in enumerate(list_of_col_names, start=1):
        fig.add_trace(
            go.Scatter(x=xr_dataset[x_col].values, y=xr_dataset[y_col].values,
                       mode="lines", name=subplot_titles[i-1]),
            row=i, col=1
        )

        fig.update_yaxes(
            title_text=subplot_titles[i-1], row=i, col=1, showgrid=True,
            gridcolor="rgba(128,128,128,0.5)", griddash="dash"
        )

    fig.update_xaxes(
        tickformat="%Y-%m-%dT%H:%M:%S", hoverformat="%Y-%m-%dT%H:%M:%S",
        showgrid=True, gridcolor="rgba(128,128,128,0.5)", griddash="dash"
    )

    fig.update_xaxes(title_text="Time [UTC+0]", row=n, col=1)


    fig.update_layout(
        autosize=True,
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=shared_title, x=0.5, xanchor="center")
    )

    return fig

def plotly_multi_time_series_shade_xr(xr_dataset, list_of_col_names, shared_title=None):
    """
    list: list_of_col_names
        [(x1_col_name, y1_col_name),
        ...,
        (x2_col_name, y2_col_name)]
    """

    pio.templates.default = "plotly_white"
    pio.renderers.default = "browser"

    n = len(list_of_col_names)
    if n == 0:
        raise ValueError("No column name provided, all columns in the <xr_dataset> will be plotted.")

    # you can use this as y label or subplot title
    subplot_titles = []
    for x_col, y_col, lower_b, mean_x, upper_b in list_of_col_names:
        col_name = f"{y_col}_{mean_x}"

        unit = xr_dataset[col_name].attrs['units']
        y_col = col_name.replace("_", " ").replace(mean_x, "")
        y_col = y_col.title()
        y_label = f"{y_col} [{unit}]"

        subplot_titles.append(y_label)

    fig = make_subplots(rows=n, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03)

    for i, (x_col, y_col, lower_b, mean_x, upper_b) in enumerate(list_of_col_names, start=1):

        # lower boundary
        col_name = f"{y_col}_{lower_b}"
        fig.add_trace(
            go.Scatter(x=xr_dataset[x_col].values,
                       y=xr_dataset[col_name].values,
                       mode="lines",
                       line=dict(width=0),
                       name=col_name,
                       showlegend=False
                       ),
            row=i, col=1
        )

        # upper boundary
        col_name = f"{y_col}_{upper_b}"
        fig.add_trace(
            go.Scatter(x=xr_dataset[x_col].values,
                       y=xr_dataset[col_name].values,
                       mode="lines",
                       line=dict(width=0),
                       name=col_name,
                       showlegend=False,
                       # fill color
                       fill='tonexty',
                       fillcolor="rgba(0,100,80,0.2)",
                       ),
            row=i, col=1
        )

        # mean boundary
        col_name = f"{y_col}_{mean_x}"
        fig.add_trace(
            go.Scatter(x=xr_dataset[x_col].values,
                       y=xr_dataset[col_name].values,
                       mode="lines",
                       line=dict(width=1),
                       name=col_name,
                       showlegend=True
                       ),
            row=i, col=1
        )

        fig.update_yaxes(
            title_text=subplot_titles[i-1], row=i, col=1, showgrid=True,
            gridcolor="rgba(128,128,128,0.5)", griddash="dash"
        )

    fig.update_xaxes(
        tickformat="%Y-%m-%dT%H:%M:%S", hoverformat="%Y-%m-%dT%H:%M:%S",
        showgrid=True, gridcolor="rgba(128,128,128,0.5)", griddash="dash"
    )

    fig.update_xaxes(title_text="Time [UTC+0]", row=n, col=1)


    fig.update_layout(
        autosize=True,
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=shared_title, x=0.5, xanchor="center")
    )

    fig.show()
