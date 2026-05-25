#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-29
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import pandas as pd
import numpy as np

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
from func.download_MeteoSwiss.fetch_data import fetch_data4SedCas, replace_nan

def request_latest_10min_data(station="mve", time_resolution="10 minutes"):

    df1 = fetch_data4SedCas(station=station, 
                           time_resolution=time_resolution, 
                           time_period="Current year")

    df2 = fetch_data4SedCas(station=station, 
                            time_resolution=time_resolution, 
                            time_period="Today")
    
    df = pd.concat([df1, df2], ignore_index=True)
    df = replace_nan(df, default_value = 0)

    df['timestamp'] = pd.to_datetime(df['timestamp [UTC+0]'], utc=True)
    df = df.sort_values(by='timestamp')
    df = df.reset_index(drop=True)
    df.drop(columns=['timestamp'], inplace=True)

    meta_data = df.columns
    latest_data = str(df.iloc[-1, 1]) # time str

    
    p_dir = f"{project_root}/data/liveshow_cache/climate"
    os.makedirs(p_dir, exist_ok=True)
    p_path = f"{p_dir}/climate_2026_t.txt"

    # incase there are no such file when call this func at the first time
    if os.path.exists(p_path):
        # load the exist downlaoded data
        df0 = pd.read_csv(p_path, header=0)
        archived_last = str(df0.iloc[-1, 1])
        
        no_new_data = latest_data == archived_last # True -> same; False -> no save
    else:
        df.to_csv(p_path, index=False)
        no_new_data = False
    

    # check whether get new data
    if no_new_data is True:
        print(f"{UTCDateTime.now().isoformat()}\n"
              f"No new data available.\nLatest record remains <{latest_data}> from station <{station}>.")
    else:
        df.to_csv(f"{p_dir}/climate_2026_t.txt", index=False)
        print(f"{UTCDateTime.now().isoformat()}\n"
              f"Downloaded new data: <{latest_data}> from station <{station}>.")
    
    return no_new_data


if __name__ == "__main__":
    no_new_data =request_latest_10min_data()
