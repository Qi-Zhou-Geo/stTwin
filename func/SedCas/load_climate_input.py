#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-07T12:56:41
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd
import xarray as xr

from obspy import UTCDateTime


def load_climate_input4model(df_climate, climate_resolution, 
                             data_source="MeteoSwiss", 
                             station="Montana (MVE)"
                             ):
    """
    Convert a climate pandas DataFrame into an xarray Dataset with metadata for SedCas.

    This function reformats raw climate forcing data into an xarray Dataset
    and attaches relevant metadata for SedCas.

    Args:
        df_climate (pandas.DataFrame): Input climate forcing data. 
            The DataFrame must contain the following columns:
                - "station"
                - "timestamp [UTC+0]"
                - "precipitation [mm per time_step]"
                - "temperature [degree]"
                - "sun radiation [W m-2]"
                
        climate_resolution (float or int): Temporal resolution of the dataset (time step size).
        data_source (str, optional): Source of the climate data. Defaults to "MeteoSwiss".
        station (str, optional): Name of the measurement station. Defaults to "Montana (MVE)".

    Returns:
        xarray.Dataset: Climate forcing data reformatted as an xarray Dataset with metadata.
    """
    
    # copy 
    df = df_climate.copy()

    # prepare time stamps
    time_float = [UTCDateTime(i).timestamp for i in df.iloc[:, 1]]
    time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in df.iloc[:, 1]]

    # extract variables
    precipitation = df.iloc[:, 2].values
    temperature = df.iloc[:, 3].values
    sun_radiation = df.iloc[:, 4].values

    # create a time stamp
    time_now = UTCDateTime().isoformat()

    # warp as xr.dataset
    climate_forcing = xr.Dataset(
        coords={
            "time": ("time", np.array(time_float)),  # numeric UTC+0 time
            "time_str": ("time", np.array(time_str)),  # string UTC+0 time
        },
        
        data_vars={
            "precipitation": ("time", precipitation,
                              {"units": f"mm per {climate_resolution} s", "description": "Total precipitation"}),

            "temperature": ("time", temperature,
                            {"units": f"°C per {climate_resolution} s", "description": "Air temperature"}),

            "sun_radiation": ("time", sun_radiation,
                              {"units": "W/m^2", "description": "Incoming solar radiation"})
        },
        
        attrs={
            "data_source": data_source,
            "station": station,
            "resolution": climate_resolution,
            "resolution_unit": f"seconds",
            "create_time": time_now
        }
    )

    return climate_forcing
