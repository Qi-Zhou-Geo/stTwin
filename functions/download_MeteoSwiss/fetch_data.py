#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-29
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd


import io
import requests
import pandas as pd
import numpy as np

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions

def check_available_sta():
    # check by links
    # https://www.meteoswiss.admin.ch/services-and-publications/applications/measurement-values-and-measuring-networks.html#param=messwerte-lufttemperatur-10min&lang=en&table=false&station=VSSIE&chart=day

    # download by click
    # https://www.meteoswiss.admin.ch/services-and-publications/applications/ext/download-data-without-coding-skills.html#lang=en&mdt=normal&pgid=&sid=&col=&di=&tr=&hdr=

    end_point = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn-precip/items"

    resp = requests.get(end_point)
    resp.raise_for_status()
    data = resp.json()

    for item in data['features']:
        try:
            url = end_point.replace("items", item['id'])
            csv_url = f"{url}/ogd-smn-precip_{item['id']}_t_now.csv" # t_now denotes the latest 10 minutes resolution data
            r = requests.get(csv_url)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep=';')

            print(f"Sta <{item['id']}> is available.")
        except Exception as e:
            print(e)

def replace_nan(df, default_value = 0, time_index = 1):
    '''
    Replace the df if there is NaN.

    Attention: you may introduce error for temperature if there is NaN.

    Args:
        df: pandas data frame, the row is time stamp and the column is different observation.
        default_value: float or int, value to replace the NaN

    Returns:
        df_temp: pandas data frame, the replaced data frame
    '''
    df_temp = df.copy()

    for col in df:  # loop col by name
        status = df[col].isna().any()

        if status:  # check if status is Ture
            nan_indices = df[df[col].isna()].index

            # convert to datatime and print the unique julday number
            s = pd.to_datetime(df.iloc[nan_indices, time_index])
            unique_days = s.dt.strftime('%Y-%m-%d').unique()

            print(f"Attention!\n"
                  f"NaN in column <{col}>, \n"
                  f"the time stamp ('%Y-%m-%d') is: \n"
                  f"{unique_days} \n")

            df_temp[col] = df[col].fillna(default_value)

    return df_temp

def fetch_data4SedCas(station="mve", time_resolution="10 minutes",  time_period="Today"):
    '''
    Fetch the climate data from MeteoSwiss

    # download by click
    # https://www.meteoswiss.admin.ch/services-and-publications/applications/ext/download-data-without-coding-skills.html#lang=en&mdt=normal&pgid=&sid=&col=&di=&tr=&hdr=

    Args:
        station: str, station name, the most cloest station to Illgraben.
        One of the following parameters,
        ["vssie", "mve"]

        time_resolution: str, tempromal resolution of the data.
        One of the following parameters,
        ["10 minutes", "Hourly", "Daily", "Monthly", "Yearly"]

        time_period: str, the time period that you prefered
        One of the following parameters,
        ["Today", "Current year", "2010-2019", "2020-2029"]

    Returns:
        df: pandas data frame, row by time, column by "data_type"
    '''

    time_resolution_mapping = {"10 minutes":"t", "Hourly":"h"}
    time_period_mapping = {"Today":"now", "Current year":"recent",
                           "2010-2019":"historical_2010-2019", "2020-2029":"historical_2020-2029"}

    # t_now denotes the latest 10 minutes resolution data since midnight UT+0 2025-03-27T00:00:00
    # t_recent denotes the latest 10 minutes resolution data since this year UT+0 2025-01-01T00:00:00

    csv_url = f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/{station}/" \
              f"ogd-smn_{station}_{time_resolution_mapping.get(time_resolution)}_{time_period_mapping.get(time_period)}.csv"


    if time_resolution == "10 minutes":
        data_type = {
            "station" : "station_abbr", # Station name
            "timestamp [UTC+0]": "reference_timestamp", # time stamp
            f"precipitation [mm per {time_resolution}]":"rre150z0", # Precipitation; ten minutes total, unit by mm
            "temperature [degree]":"tre200s0", # Air temperature 2 m above ground; current value, unit by degree C
            "sun radiation [W per squared m]":"gre000z0" # Global radiation; ten minutes mean, unit by W/m^2
        }
    elif time_resolution == "Hourly":
        data_type = {
            "station" : "station_abbr", # Station name
            "timestamp [UTC+0]": "reference_timestamp", # time stamp
            f"precipitation [mm per {time_resolution}]":"rre150h0", # Precipitation; hourly total, unit by mm/h
            "temperature [degree]":"tre200h0", # Air temperature 2 m above ground; hourly mean, unit by degree C
            "sun radiation [W per squared m]":"gre000h0" # Global radiation; hourly mean, unit by W/m^2
        }
    else:
        print(f"Error! please check the time_resolution {time_resolution}.")

    try:
        r = requests.get(csv_url)
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text), sep=';')

    except Exception as e:
        print(e)

    # select only the columns in data_type values
    selected_cols = [col for col in data_type.values() if col in df.columns]
    df = df[selected_cols]

    # Rename columns to friendly names (keys of data_type)
    rename_dict = {v: k for k, v in data_type.items() if v in df.columns}
    df.rename(columns=rename_dict, inplace=True)

    # convert time format
    df["timestamp [UTC+0]"] = pd.to_datetime(df["timestamp [UTC+0]"], format='%d.%m.%Y %H:%M')
    df["timestamp [UTC+0]"] = df["timestamp [UTC+0]"].dt.strftime('%Y-%m-%dT%H:%M:%S') #'%Y-%m-%dT%H:%M:%S'

    return df

# prec data
# https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn-precip/vsbas/ogd-smn-precip_vsbas_t_now.csv

# temperature data
# https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/mve/ogd-smn_mve_t_now.csv

# snow
# https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/mve/ogd-nime_mve_d_recent.csv

# radiation
# https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/mve/ogd-smn_mve_t_now.csv

# find the station and choose which station you want to use