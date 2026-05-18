#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

from dash import Dash, dcc, html, Output, Input, State

# region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.liveshow.t2s1_plot_results import load_cache, plotly_multi_time_series_xr


font_global = {"fontFamily": "Arial, sans-serif", "fontSize": "12px", "color": "#2d2d2d"}



# app layout
app = Dash(__name__)

# define the layout
app.layout = html.Div(
    children=[
        
        dcc.RadioItems(
            id="data-toggle",
            options=[
                {"label": "Hydro Status", "value": "hydro"},
                {"label": "Sediment Status", "value": "sed"},
            ],
            value="sed",  # default option to show "Sediment Status"
            inline=True,
            style={"marginBottom": "3px", "fontSize": "14px", "gap": "1px"},
        ),
        
        # shows the actual subplots
        dcc.Graph(id="chart", style={"height": "100vh"}),
        
        # auto-refresh every 10 minutes
        dcc.Interval(id="timer", interval=10 * 60 * 1000),  # 10 min in ms
    ],
    
    style=font_global
)


# callback: chart refreshes on timer OR toggle switch
@app.callback(
    # The result of this function should be sent to the figure property 
    # of the component with id='chart'
    Output("chart", "figure"),
    
    # update the plot by the following two actions
    Input("timer", "n_intervals"),
    Input("data-toggle", "value"),
)

def refresh_chart(n_intervals, data_type):

    ds_sub, vars_dict = load_cache(data_type)

    xr_dataset = ds_sub
    list_of_col_names = [( "time_str", key) for key in vars_dict.keys()]

    fig = plotly_multi_time_series_xr(xr_dataset, list_of_col_names, vars_dict)

    return fig


if __name__ == "__main__":
    # (1) use 127.0.0.1 so the app isn't exposed to your local network, host="0.0.0.0"
    # (2) turn debug = False for real-live-demo
    app.run(host="0.0.0.0", port=8050, debug=False)
