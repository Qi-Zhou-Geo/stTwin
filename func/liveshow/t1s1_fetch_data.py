#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-21T22:49:05
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

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

    
    # prepare the output dir
    time_year = UTCDateTime.now().year
    txt_dir = Path(project_root) / f"deploy/liveshow_cache/climate"
    txt_path = Path(txt_dir) / f"climate_{time_year}_t.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)


    # incase there are no such file when call this func at the first time
    if txt_path.exists():
        # load the exist downlaoded data
        df0 = pd.read_csv(txt_path, header=0)
        archived_last = str(df0.iloc[-1, 1])
        
        latest_matches_archive = latest_data == archived_last
        
        # is_new_data: True >> find new data, False >> no new data
        is_new_data = not latest_matches_archive
    else:
        is_new_data = True
    

    # check whether get new data
    if is_new_data is True:
        df.to_csv(txt_path, index=False)
        msg = f"Downloaded new data: <{latest_data}> from station <{station}>."
    else:
        msg = (f"No new data available.\n"
               f"Latest record remains <{latest_data}> from station <{station}>.")

    
    return is_new_data, msg


if __name__ == "__main__":
    is_new_data, msg =request_latest_10min_data()
    print(msg)
