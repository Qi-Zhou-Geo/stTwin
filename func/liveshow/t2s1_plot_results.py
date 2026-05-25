#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import json

import numpy as np
import pandas as pd
import xarray as xr

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from obspy import UTCDateTime

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

def load_cache(data_type):

    if data_type in ["hydro", "hydro_output"]:
        data = f"{project_root}/data/liveshow_cache/results/hydro_output.nc"
        
        vars_dict = {
            "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            "modelled_SWE": "SWE: Modelled Snow-Water-Equivalent\n[mm]",
            "Qs": "Qs: Surface Discharge\n[mm]",
            "Qss": "Qss: Sub-Surface Discharge\n[mm]"
        }
        
    elif data_type in ["sed", "sed_output"]:
        data = f"{project_root}/data/liveshow_cache/results/sed_output.nc"
        
        vars_dict = {
             "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            "hillslope_storage_Q50": "HS: Hillslope Storage\n[mm]",
            "channel_storage_Q50": "CS: Channel Storage\n[mm]",
            "sed_transport_real_Q50": "SY: Sediments Yield\n[mm]"
        }
    else:
        raise ValueError(f"Please check the input <data_type> {data_type}")
    
    ds1 = xr.load_dataset(data)
    ds2 = xr.load_dataset(f"{project_root}/data/liveshow_cache/results/climate_forcing.nc")
    
    ds_merged = xr.merge([ds1, ds2])
    vars_to_keep  = list(vars_dict.keys())
    ds_sub = ds_merged[vars_to_keep]
    
    print(f"{UTCDateTime.now().isoformat()}\n"
          f"<load_cache> with latest data <{ds_sub.coords['time_str'].values[-1]}> \n")
    
    return ds_sub, vars_dict


def plotly_multi_time_series_xr(xr_dataset, list_of_col_names, vars_dict):
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

    for i, (x_col, y_col) in enumerate(list_of_col_names, start=1):
        fig.add_trace(
            go.Scatter(x=xr_dataset[x_col].values, 
                       y=xr_dataset[y_col].values, 
                       mode="lines", name=subplot_titles[i - 1]),
            row=i, col=1
        )

        fig.update_yaxes(
            title_text=subplot_titles_abbr[i-1], row=i, col=1, showgrid=True,
            gridcolor="rgba(128,128,128,0.5)", griddash="dash"
        )

    # x-axis style + range-selector buttons
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
    with open(f"{project_root}/data/liveshow_cache/results/last_stTwin_update.json", "r") as f:
        temp = json.load(f)
        # last_update = f"Latest stTwin Update: {UTCDateTime().strftime('%Y-%m-%dT%H:%M:%S')} [UTC+0]"
    last_update = temp["last_update"]
    
    
    fig.update_layout(
        autosize=True,
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=f"{last_metro_swiss}<br>{last_update}", x=0.5, xanchor="center", font=dict(size=12)),
        font=font_global,
    )


    return fig


# data_type = "hydro"
# ds_sub, vars_dict = load_cache(data_type)


# xr_dataset = ds_sub
# list_of_col_names = []
# for key in vars_dict.keys():
#     list_of_col_names.append(("time_str", key))

# plotly_multi_time_series_xr(xr_dataset, list_of_col_names, vars_dict)
