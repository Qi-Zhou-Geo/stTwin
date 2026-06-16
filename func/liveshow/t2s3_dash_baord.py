#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import yaml
from dash import Dash, dcc, html, Output, Input, State

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# import the custom functions
from func.liveshow.t2s1_load_cache import load_cache_monitoring, load_cache_whatif
from func.liveshow.t2s2_plotly import plotly_multi_time_series_xr
from func.SedCas_whatif.create_bound import load_what_if_bound

font_global = {"fontFamily": "Arial, sans-serif", "fontSize": "12px", "color": "#2d2d2d"}
cfg, cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range = load_what_if_bound()

cycle_period_range = [int(v) for v in cycle_period_range]
storm2drought_ratio_range = [round(float(v), 1) for v in storm2drought_ratio_range]
storm_onset_month_range = [int(v) for v in storm_onset_month_range]
storm_onset_day_range = [int(v) for v in storm_onset_day_range]

# app layout
app = Dash(__name__)

app.layout = html.Div(
    children=[
        
        # Level 1: Monitoring or WhatIf
        dcc.RadioItems(
            id="level1-toggle",
            options=[
                {"label": "Monitoring (colorful-line)", "value": "monitoring"},
                {"label": "WhatIf (black-line)", "value": "whatif"},
            ],
            value="monitoring",
            inline=True,
            style={"marginBottom": "3px", "fontSize": "14px", "gap": "1px"},
        ),

        # Level 2: Hydro or Sed
        dcc.RadioItems(
            id="level2-toggle",
            options=[
                {"label": "Hydro", "value": "hydro"},
                {"label": "Sed",   "value": "sed"},
            ],
            value="sed",
            inline=True,
            style={"marginBottom": "3px", "fontSize": "14px", "gap": "1px"},
        ),

        # WhatIf sliders (hidden until level1 == whatif)
        html.Div(
            id="whatif-panel",
            children=[
                html.Div([
                    html.Label("cycle_period"),
                    dcc.Slider(id="cycle_period",
                               min=cfg["cycle_period"]["value_min"],
                               max=cfg["cycle_period"]["value_max"],
                               step=None,
                               value=float(cycle_period_range[0]),
                               marks={v: str(v) for v in cycle_period_range},
                               ),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("cstorm2drought_ratio"),
                    dcc.Slider(id="storm2drought_ratio",
                               min=cfg["storm2drought_ratio"]["value_min"],
                               max=cfg["storm2drought_ratio"]["value_max"],
                               step=None,
                               value=float(storm2drought_ratio_range[0]),
                               marks={v: str(v) for v in storm2drought_ratio_range},
                               ),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("storm_onset_month"),
                    dcc.Slider(id="storm_onset_month",
                               min=cfg["storm_onset_month"]["value_min"],
                               max=cfg["storm_onset_month"]["value_max"],
                               step=None,
                               value=float(storm_onset_month_range[0]),
                               marks={v: str(v) for v in storm_onset_month_range},
                               ),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("storm_onset_day"),
                    dcc.Slider(id="storm_onset_day",
                               min=cfg["storm_onset_day"]["value_min"],
                               max=cfg["storm_onset_day"]["value_max"],
                               step=None,
                               value=float(storm_onset_day_range[0]),
                               marks={v: str(v) for v in storm_onset_day_range},
                               ),
                ], style={"marginBottom": "10px"}),
            ],
            style={"display": "none", "padding": "10px", "border": "1px solid #ddd"},
        ),

        # shows the actual subplots
        dcc.Graph(id="chart", style={"height": "100vh"}),
        # auto-refresh every 10 minutes
        dcc.Interval(id="timer", interval=10 * 60 * 1000),
    ]
)


# show/hide whatif slider panel
@app.callback(
    Output("whatif-panel", "style"),
    Input("level1-toggle", "value"),
)

def toggle_whatif_panel(level1):
    if level1 == "whatif":
        output_dict = {"display": "block", "padding": "10px", "border": "1px solid #ddd"}
    else:
        output_dict = {"display": "none"}

    return output_dict


# callback: chart refreshes on timer OR toggle switch
@app.callback(
    # The result of this function should be sent to the figure property 
    # of the component with id='chart'
    Output("chart", "figure"),

    # update the plot by the following two actions
    Input("timer", "n_intervals"),
    Input("level1-toggle", "value"),
    Input("level2-toggle", "value"),
    
    Input("cycle_period", "value"),
    Input("storm2drought_ratio", "value"),
    Input("storm_onset_month", "value"),
    Input("storm_onset_day", "value"),
)


def refresh_chart(n_intervals, level1, level2,
                  cycle_period, storm2drought_ratio,
                  storm_onset_month, storm_onset_day):
    
    print(f"CP={cycle_period} R={storm2drought_ratio} M={storm_onset_month} D={storm_onset_day}")
    
    # we will have four combinations
    # level1 in ["monitoring", "whatif"]
    # level2 in ["hydro", "sed"]
    
    application_type, data_type = level1, level2
    
    if level1 == "monitoring":
        ds_sub, vars_dict = load_cache_monitoring(data_type, t1="2025-01-01T00:00:00", t2="2036-01-01T00:00:00")
        list_of_col_names = [("time_str", key) for key in vars_dict.keys()]
        
        fig = plotly_multi_time_series_xr(ds_sub, list_of_col_names, vars_dict, xr_dataset_whatif=None)
    elif level1 == "whatif":
        whatif_type = f"CP={float(cycle_period)}_R={float(storm2drought_ratio)}_M={float(storm_onset_month)}_D={float(storm_onset_day)}"
        
        ds_whatif, vars_dict_whatif, ds_monitoring, vars_dict = load_cache_whatif(data_type, whatif_type, t1="2023-01-01T00:00:00", t2="2026-01-01T00:00:00")
        list_of_col_names = [("time_str", key) for key in vars_dict.keys()]
        
        fig = plotly_multi_time_series_xr(ds_monitoring, list_of_col_names, vars_dict, xr_dataset_whatif=ds_whatif)

    return fig

if __name__ == "__main__":
    # host="127.0.0.1"
    # Binds the app only to localhost (same machine access only)

    # port=8050
    # Local service port

    # debug=False
    # Disable Flask debug mode for stability in production
    
    app.run(host="127.0.0.1", port=8050, debug=True)
