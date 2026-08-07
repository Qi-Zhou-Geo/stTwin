#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-08-07T11:41:29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import argparse
from dash import Dash, dcc, html, Output, Input, State

import numpy as np
import pandas as pd

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
from func.liveshow.t2s1_load_cache import load_cache_monitoring, load_cache_whatif, load_cache_pro
from func.liveshow.t2s2_plotly import plotly_multi_time_series_xr
from func.SedCas_whatif.create_bound import load_what_if_bound
from func.toolkit.logger_printer import setup_logger


font_global = {"fontFamily": "Arial, sans-serif", "fontSize": "12px", "color": "#2d2d2d"}

LEVEL2_MONITORING_OPTIONS = [
    {"label": "Discharge Magnitude", "value": "hydro"},
    {"label": "Sediment Magnitude", "value": "sed"},
    {"label": "Debris-Flow Probability", "value": "seis"},
]

LEVEL2_WHATIF_OPTIONS = [
    {"label": "Discharge Magnitude", "value": "hydro"},
    {"label": "Sediment Magnitude", "value": "sed"},
]


def load_whatif_slider(method="from_text"):

    if method == "from_yaml":
        cfg, cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range = load_what_if_bound()

        cycle_period_range = [int(v) for v in cycle_period_range]
        storm2drought_ratio_range = [round(float(v), 1) for v in storm2drought_ratio_range]
        storm_onset_month_range = [int(v) for v in storm_onset_month_range]
        storm_onset_day_range = [int(v) for v in storm_onset_day_range]
    
    elif method == "from_text":
        scenario_path = Path(project_root) / "config/scenario_bound.txt"
        df = pd.read_csv(scenario_path, header=0)
        
        cycle_period_range = np.unique(df["cycle_period"]).tolist()
        storm2drought_ratio_range = np.unique(df["storm2drought_ratio"]).tolist()
        storm_onset_month_range = np.unique(df["storm_onset_month"]).tolist()
        storm_onset_day_range = np.unique(df["storm_onset_day"]).tolist()
        
    else:
        raise ValueError(f"method got unexception value: {method}")
         
    return cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range


def build_layout(whatif_slider_cfg):
    
    cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range = whatif_slider_cfg
    
    # app layout
    layout = html.Div(
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

            # Level 2: Hydro or Sed or Seis
            dcc.RadioItems(
                id="level2-toggle",
                options=LEVEL2_MONITORING_OPTIONS,
                value="sed",
                inline=True,
                style={"marginBottom": "3px", "fontSize": "14px", "gap": "1px"},
            ),

            # Level 2: WhatIf sliders (hidden until level1 == whatif)
            html.Div(
                id="whatif-panel",
                children=[
                    html.Div([
                        html.Label("Cycle Period"),
                        dcc.RadioItems(
                            id="cycle_period",
                            options=[{"label": str(v), "value": v} for v in cycle_period_range],
                            value=cycle_period_range[0],
                            inline=True
                            ),
                    ], style={"marginBottom": "10px"}),

                    html.Div([
                        html.Label("Storm to Drought Ratio"),
                        dcc.RadioItems(
                            id="storm2drought_ratio",
                            options=[{"label": str(v), "value": v} for v in storm2drought_ratio_range],
                            value=storm2drought_ratio_range[0],
                            inline=True),
                    ], style={"marginBottom": "10px"}),

                    html.Div([
                        html.Label("Storm Onset Month"),
                        dcc.RadioItems(
                            id="storm_onset_month",
                            options=[{"label": str(v), "value": v} for v in storm_onset_month_range],
                            value=storm_onset_month_range[0],
                            inline=True),
                    ], style={"marginBottom": "10px"}),

                    html.Div([
                        html.Label("Storm Onset Day"),
                        dcc.RadioItems(
                            id="storm_onset_day",
                            options=[{"label": str(v), "value": v} for v in storm_onset_day_range],
                            value=storm_onset_day_range[0],
                            inline=True),
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

    return layout


def update_level2_options(level1, current_level2):

    if level1 == "monitoring":
        options = LEVEL2_MONITORING_OPTIONS
    else:
        options = LEVEL2_WHATIF_OPTIONS

    valid_values = {opt["value"] for opt in options}

    if current_level2 in valid_values:
        return options, current_level2

    return options, "sed"


def toggle_whatif_panel(level1):
    
    if level1 == "whatif":
        output_dict = {"display": "block", "padding": "10px", "border": "1px solid #ddd"}
    else:
        output_dict = {"display": "none"}

    return output_dict


def refresh_chart(n_intervals, level1, level2,
                  cycle_period, storm2drought_ratio,
                  storm_onset_month, storm_onset_day):
    
    # we will have these combinations
    # level1 in ["monitoring", "whatif"]
    # level2 in ["hydro", "sed", "seis"]
    
    # (1) level 1 >> ["monitoring", "whatif"]
    if level1 == "monitoring":
        
        # (1-1) level2 >> ["hydro", "sed", "seis"]
        # results from SedCas
        if level2 in ["hydro", "sed"]:
            msg, ds_monitoring, vars_dict = load_cache_monitoring(data_type=level2, 
                                                           t1="2025-01-01T00:00:00", 
                                                           t2="2036-01-01T00:00:00")
            list_of_col_names = [("time_str", key) for key in vars_dict.keys()]
            xr_dataset_whatif = None
            
            # fig = plotly_multi_time_series_xr(ds_sub, list_of_col_names, vars_dict, xr_dataset_whatif=None)
        
        # (1-2) level2 >> ["hydro", "sed", "seis"]
        # results from Flow-Alert
        elif level2 in ["seis"]:
            pro_dir = Path(project_root.parent) / "Flow-Alert/deploy/liveshow_cache/pro"
            msg, ds_monitoring, vars_dict = load_cache_pro(pro_dir) # type: ignore
            list_of_col_names = [("time_str", key) for key in vars_dict.keys()]
            xr_dataset_whatif = None
            
        # (1-3) level2 >> error
        else:
            raise ValueError(f"level 2 got unexcepted value: level2={level2}")
    
    # (2) level 1 >> ["monitoring", "whatif"]
    elif level1 == "whatif":
        
        # (2-1) level2 >> do not need
        whatif_type = f"CP={int(cycle_period)}_R={storm2drought_ratio:.3f}_M={int(storm_onset_month)}_D={int(storm_onset_day)}"
        
        msg, xr_dataset_whatif, vars_dict_whatif, ds_monitoring, vars_dict = load_cache_whatif(data_type=level2, 
                                                                                       whatif_type=whatif_type, 
                                                                                       t1="2023-01-01T00:00:00", 
                                                                                       t2="2026-01-01T00:00:00")
        list_of_col_names = [("time_str", key) for key in vars_dict.keys()]

    # (3) level 1 >> error
    else:
        raise ValueError(f"level 1 got unexcepted value: level2={level1}")
    
    fig = plotly_multi_time_series_xr(xr_dataset=ds_monitoring, 
                                      list_of_col_names=list_of_col_names,
                                      vars_dict=vars_dict, 
                                      xr_dataset_whatif=xr_dataset_whatif)
    
    return fig


def register_callbacks(app):
    
    # update the Level-2 choices when switching between Monitoring and WhatIf
    # Monitoring supports hydro, sed, and seis
    # WhatIf supports only hydro and sed
    level2_callback = app.callback(
        Output("level2-toggle", "options"),
        Output("level2-toggle", "value"),
        Input("level1-toggle", "value"),
        State("level2-toggle", "value"),
    )
    level2_callback(update_level2_options)
    
    
    # show/hide whatif slider panel, return as Python decorator
    toggle_callback = app.callback(
        Output("whatif-panel", "style"),
        Input("level1-toggle", "value"),
    )
    toggle_callback(toggle_whatif_panel)
    
    
    # callback: chart refreshes on timer OR toggle switch, return as Python decorator
    refresh_callback = app.callback(
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
    refresh_callback(refresh_chart)


def create_app():
    whatif_slider_cfg= load_whatif_slider()

    app = Dash(__name__)
    app.layout = build_layout(whatif_slider_cfg)

    register_callbacks(app)

    return app

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    # host="127.0.0.1" >> Binds the app only to localhost (same machine access only)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    # port=8050 >> Local service port
    parser.add_argument("--port", type=int, default=8050)
    
    parser.add_argument("--output_dir", type=str, default=f"{project_root}/deploy/liveshow_cache/logs")
    parser.add_argument("--log_filename", type=str, default="t2_main.log")
    args = parser.parse_args()
    
    # run app
    app = create_app()

    # debug=False >> Disable Flask debug mode for stability in production
    app.run(host=args.host, port=args.port, debug=False)
