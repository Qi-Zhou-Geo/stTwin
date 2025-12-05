#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-10-14
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import numpy as np
import pandas as pd

import time
import json

import threading

from flask import Flask, render_template_string, jsonify
from datetime import datetime, timezone


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.download_MeteoSwiss.fetch_data import fetch_data4SedCas
from demo.live.templates import INDEX_PAGE



app = Flask(__name__)

# Global variable to store the latest data
latest_data = {"timestamp": None, "data": []}


def dataframe_to_json(df):

    """Convert DataFrame to JSON-safe format, replacing NaN with None"""

    data = df.to_dict('records')

    for record in data:
        for key, value in record.items():
            # Replace NaN with None (which becomes null in JSON)
            if isinstance(value, float) and np.isnan(value):
                record[key] = -10

    return data


def fetch_and_update_data():
    while True:
        try:
            # Fetch data from your function
            df = fetch_data4SedCas(station="mve", time_resolution="10 minutes", time_period="Today")
            print(df.columns)
            print(np.array(df.iloc[-1, :]))

            # Convert dataframe to JSON format (handles NaN)
            data_json = dataframe_to_json(df)

            # Use the latest timestamp from the data (already in UTC+0)
            latest_timestamp = df.iloc[-1]['timestamp [UTC+0]']
            latest_data["timestamp"] = latest_timestamp
            latest_data["data"] = data_json

            print(f"Data fetched at: UTC+0 {latest_data['timestamp']}")

        except Exception as e:
            print(f"Error fetching data: {e}")

        # Wait 10 minutes (600 seconds)
        time.sleep(600)


@app.route('/')
def index():
    return render_template_string(INDEX_PAGE)


@app.route('/api/data')
def get_data():

    return jsonify(latest_data)


if __name__ == '__main__':

    # Fetch initial data BEFORE starting the app
    try:
        df = fetch_data4SedCas(station="mve", time_resolution="10 minutes", time_period="Today")
        latest_data["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        latest_data["data"] = dataframe_to_json(df)
        print(f"Initial data loaded: {latest_data['timestamp']}")
    except Exception as e:
        print(f"Error fetching initial data: {e}")

    # Start background thread for updates
    data_thread = threading.Thread(target=fetch_and_update_data, daemon=True)
    data_thread.start()

    app.run(debug=False, use_reloader=False)
