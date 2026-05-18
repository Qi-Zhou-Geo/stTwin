#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-12-15
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import tempfile
import webbrowser

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from obspy import UTCDateTime

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# endregion

# import the custom functions

# all the following functions are stored under the same path
from plot_uncertainty import load_data


def plotly_multi_time_series_std(list_of_tuples, sigma_scale=1, shared_title=None,
                                  search_time=None, window_hours=24, show_plot=False):
    """
    list_of_tuples: list of tuples
        [(x, y_mean, y_std, label), ...]
        x can be float timestamps, datetime objects, or time strings
        y_std can be None to skip the shaded band
    """

    PLOTLY_COLORS = [
        "99,110,250",  # blue-purple
        "239,85,59",  # red-orange
        "0,204,150",  # teal
        "171,99,250",  # purple
        "255,161,90",  # orange
        "25,211,243",  # cyan
        "255,102,146",  # pink
        "182,232,128",  # light green
        "255,151,255",  # lavender
        "254,203,82",  # yellow
    ]


    pio.templates.default = "plotly_white"
    pio.renderers.default = "browser"

    n = len(list_of_tuples)
    if n == 0:
        raise ValueError("No series provided")

    fig = make_subplots(rows=n, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03)

    for i, (x, y_mean, y_std, label) in enumerate(list_of_tuples, start=1):

        rgb = PLOTLY_COLORS[(i - 1) % len(PLOTLY_COLORS)]

        # --- shaded std band ---
        if y_std is not None:
            y_lower = y_mean - sigma_scale * y_std
            y_lower = np.clip(y_lower, a_min=0, a_max=None)
            y_upper = y_mean + sigma_scale * y_std

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({rgb},0.3)",  # same color, transparent
                    line=dict(color=f"rgba({rgb},0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{label}_band",
                ),
                row=i, col=1,
            )

        # --- mean line ---
        fig.add_trace(
            go.Scatter(x=x, y=y_mean, mode="lines",
                       line=dict(color=f"rgb({rgb})"),
                       name=label),
            row=i, col=1,
        )

        fig.update_yaxes(
            title_text=label, row=i, col=1, showgrid=True,
            gridcolor="rgba(128,128,128,0.5)", griddash="dash",
        )

    # ── x-axis + range selector ───────────────────────────────────────────────
    rangeselector = dict(buttons=[
        dict(count=1,  label="1d", step="day",   stepmode="backward"),
        dict(count=7,  label="1w", step="day",   stepmode="backward"),
        dict(count=1,  label="1m", step="month", stepmode="backward"),
        dict(count=6,  label="6m", step="month", stepmode="backward"),
        dict(count=1,  label="1y", step="year",  stepmode="backward"),
        dict(step="all", label="All"),
    ])

    fig.update_xaxes(
        tickformat="%Y-%m-%dT%H:%M:%S", hoverformat="%Y-%m-%dT%H:%M:%S",
        showgrid=True, gridcolor="rgba(128,128,128,0.5)", griddash="dash",
        rangeselector=rangeselector,
    )
    fig.update_xaxes(title_text="Time [UTC+0]", row=n, col=1)

    # ── optional search_time marker + zoom ───────────────────────────────────
    if search_time is not None:
        t_center = UTCDateTime(search_time)
        t_start  = t_center - window_hours * 3600
        t_end    = t_center + window_hours * 3600

        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=t_center.isoformat(), x1=t_center.isoformat(),
            y0=0, y1=1,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.update_xaxes(range=[t_start.isoformat(), t_end.isoformat()])

    fig.update_layout(
        autosize=True,
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=shared_title, x=0.5, xanchor="center"),
    )

    # ── show / return ─────────────────────────────────────────────────────────
    if show_plot:
        html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)

        plotly_search_box = f"{project_root}/functions/toolkit/plotly_search_box.html"
        with open(plotly_search_box, "r", encoding="utf-8") as f:
            search_box_html = f.read()

        html_str = html_str.replace("</body>", search_box_html + "\n</body>")

        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
        tmp.write(html_str)
        tmp.close()
        webbrowser.open(f"file://{tmp.name}")

    return fig

def main(model_version, output_dir):

    key_type_l = ["hydro", "hydro", "hydro", "sed", "sed"]
    key_list_l = [
        ["Q", "Qs", "Qss"],
        ["modelled_SWE", "snow_delta_depth", "snow_acc", "snow_melt"],
        ["albedo", "PET", "snow_acc", "snow_melt"],
        ["ls_Q1", "ls_Q50", "ls_Q99"],
        ["hillslope_storage_Q50", "channel_storage_Q50", "sed_transport_real_Q50"]
    ]
    for key_type, key_list in zip(key_type_l, key_list_l):

        list_of_tuples = []  # [(x, y_mean, y_std, label), ...]
        for key in key_list:
            print(key_type, key)
            time_str, arr = load_data(key_type, key, model_version, num_draw=100)
            y_mean = np.mean(arr, axis=1)  # by row
            y_std = np.std(arr, axis=1)  # by row

            list_of_tuples.append((time_str, y_mean, y_std, key))

        fig = plotly_multi_time_series_std(list_of_tuples, sigma_scale=1, shared_title=None)
        fig.write_html(f"{output_dir}/{key_type}_{key_list[0]}.html")


if __name__ == "__main__":
    model_version = "bayesian_inference0dot4"
    output_dir = f"{project_root}/pipeline/real_pred/{model_version}"

    main(model_version, output_dir)