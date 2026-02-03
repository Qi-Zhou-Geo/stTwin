#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# __note__ = This code is adapted from SedCas (Author: Jacob Hirschberg, Created: 2022-02-03, Source: https://github.com/jacobhirschberg/SedCas)
#           and is distributed under the terms of the GNU General Public License v3.0 (GPL-3.0).

import math
import random
from itertools import dropwhile

import pandas as pd
import numpy as np



from obspy import UTCDateTime
import xarray as xr


data = pd.read_csv(f"/Users/qizhou/#python/stTwin/data/SedCas_input/climate_2004_2017_h.txt", header=0)

time = [UTCDateTime(i).timestamp for i in data.iloc[:, 1]]
time_str = [UTCDateTime(i).isoformat() for i in data.iloc[:, 1]]

# Optional: human-readable strings
time_str = [str(UTCDateTime(i)) for i in data.iloc[:, 1]]

# Extract variables
precipitation = data.iloc[:, 2].values
temperature = data.iloc[:, 3].values
sun_radiation = data.iloc[:, 4].values


data_source = "MeteoSwiss"
station = data.iloc[0, 0]  # station name
resolution = time[1] - time[0] # unit is second
time_now = UTCDateTime().isoformat()

climate_forcing = xr.Dataset(
    coords={
        "time": ("time", np.array(time)),            # numeric UTC+0 time
        "time_str": ("time", np.array(time_str)),    # string UTC+0 time
    },
    data_vars={
        "precipitation": ("time", precipitation,
                          {"units": f"mm per {resolution} s", "description": "Total precipitation"}),

        "temperature": ("time", temperature,
                        {"units": f"°C per {resolution} s", "description": "Air temperature"}),

        "sun_radiation": ("time", sun_radiation,
                          {"units": "W/m^2", "description": "Incoming solar radiation"})
    },
    attrs={
        "source": data_source,
        "station": station,
        "resolution": resolution,
        "resolution_unit": f"seconds",
        "create_time": time_now
    }
)


num_data = len(precipitation)
time_now = UTCDateTime().isoformat()

num_HUR = 2
hydro_output = xr.Dataset(
    coords={
        "time": ("time", np.array(time)),  # numeric UTC+0 time
        "time_str": ("time", np.array(time_str)),  # string UTC+0 time
        "HRU_id": np.arange(num_HUR),
    },
    data_vars={
        # (1) snow_water_equivalent
        # snow depth SWE
        "modelled_SWE": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                       {"units": f"mm per {resolution} s", "description": "Modelled snow-water-equivalent depth"}),

        # Snow pack changes
        "delta_depth": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                         {"units": f"mm per {resolution} s", "description": "Snow pack changes"}),

        # snow accumulation
        "snow_acc": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                              {"units": f"mm per {resolution} s", "description": "Snow accumulation"}),
        # snow melt
        "snow_melt": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                     {"units": f"mm per {resolution} s", "description": "Snow melting"}),

        # albedo
        "albedo": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                      {"units": "mm", "description": "Snow (when covered with snow) or soil (when no more snow existing) albedo"}),

        # (2)
        # Potential evapotranspiration [mm]
        "potential_ET": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                         {"units": "mm", "description": "Potential evapotranspiration"}),

        # Actual evapotranspiration [mm]
        "actual_ET": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                      {"units": "mm", "description": "Actual evapotranspiration"}),

        # discharge [mm]
        "discharge": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                      {"units": "mm", "description": "Total discharge"}),

        # discharge from overland flow [mm]
        "discharge_surface": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                              {"units": "mm", "description": "Discharge from overland flow"}),

        # discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
        "discharge_sub_surface": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                                  {"units": "mm", "description": "Discharge from subsurface flow"}),

        # state of soil water storage [mm]
        "soil_water_storage": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                               {"units": "mm", "description": "Soil water storage"}),

    },
    attrs={
        "source": data_source,
        "station": station,
        "resolution": resolution,
        "resolution_unit": f"seconds",
        "create_time": time_now
    }
)


# Snow Degree-Day-Module
def snow_water_equivalent(HRU_id, climate_forcing, hydro_output, snow_melt_r, T_theta_a, T_theta_m, snow_albedo, soil_albedo):

    # the temperature, precipitation are from climate_forcing (xarray.Dataset)
    temperature = climate_forcing.temperature.values # unit by degree per delta_t
    precipitation = climate_forcing.precipitation.values # unit by mm per delta_t


    # snow accumulation
    cold_cond = temperature <= T_theta_a # cold condition
    warm_cond = temperature > T_theta_a
    snow_acc = precipitation.copy()
    snow_acc[warm_cond] = 0.0  # no snow pack accumulation where temperature is above


    # snow melt
    melt_cond = temperature > T_theta_m
    non_melt_cond = temperature <= T_theta_m
    T_grad = temperature - T_theta_m  # melting gradient
    T_grad[non_melt_cond] = 0.0  # no snowmelt where temperature is below
    snow_melt = snow_melt_r * T_grad  # potential snowmelt based on temperature


    # compute actual snow depth
    modelled_SWE = np.zeros(len(precipitation)) # modelled snow water equivalent [mm]
    delta_depth = np.zeros(len(precipitation)) # snow pack change SWE

    for i in range(1, len(precipitation)):
        # modelled_SWE[i] is total snow water equivalent (SWE) stored on the ground at time step i
        available_snow = modelled_SWE[i - 1] + snow_acc[i]

        # incase the snow_melt[i] > available_snow,
        # the melt snow bigger than exit snow, we will select the min one
        actual_melt = min(snow_melt[i], available_snow)
        snow_melt[i] = actual_melt

        # update the SWE
        modelled_SWE[i] = available_snow - actual_melt

        # delta_depth is change in SWE during one time step i
        delta_depth[i] = modelled_SWE[i] - modelled_SWE[i - 1]


    # Albedo
    albedo = np.zeros(len(modelled_SWE))

    with_snow_cond = modelled_SWE > 0
    albedo[with_snow_cond] = snow_albedo

    without_snow_cond = modelled_SWE <= 0
    albedo[without_snow_cond] = soil_albedo

    # update the xarray.Dataset
    hydro_output["modelled_SWE"][:, HRU_id] = modelled_SWE
    hydro_output["delta_depth"][:, HRU_id] = delta_depth
    hydro_output["snow_acc"][:, HRU_id] = snow_acc
    hydro_output["snow_melt"][:, HRU_id]  = snow_melt
    hydro_output["albedo"][:, HRU_id] = albedo

    return hydro_output

HRU_id = 0
snow_melt_r, T_theta_a, T_theta_m = 0.08, 0.5, 0.6
snow_albedo, soil_albedo = [0.4, 0.65], [0.15, 0.25]
snow_water_equivalent(HRU_id, climate_forcing, hydro_output, snow_melt_r, T_theta_a, T_theta_m, snow_albedo[HRU_id], soil_albedo[HRU_id])

# Potential Evapotranspiration Module
def actual_evap(HRU_id, climate_forcing, hydro_output, cloud_cover_r, elevation, U):


    # the temperature, precipitation are from climate_forcing (xarray.Dataset)
    temperature = climate_forcing.temperature.values # unit by degree per delta_t
    sun_radiation = climate_forcing.sun_radiation.values # unit by mm per delta_t
    albedo = hydro_output.albedo[:, HRU_id].values
    sps_temperature = climate_forcing.attrs["resolution"]

    esat = 611 * np.exp(17.27 * temperature / (237.3 + temperature))  ## Vapor Pressure Saturation
    ea = U * esat

    #### Net Radiation
    si = 5.6704 * (10 ** -8)  # Stefan-Boltzman Constant [W/m**2.K**4]
    ######
    K = 0.1 + 0.9 * (1 - 0.6 * (cloud_cover_r ** 2.5))  #### Emissiivty coefficient cloud
    ei = 0.34 - 0.14 * np.sqrt(ea / 1000)  ## Net emissivity humidity
    ######
    D_Rlw = ei * K * si * ((temperature - 273.15) ** 4)  ## Net Longwave Radiatio W/m^2
    Rn = sun_radiation * (1 - albedo) - D_Rlw  ## Net Radiation  W/m^2

    #################################################
    ##### COMBINED ENERGETIC AND AERODYNAMIC METHOD
    ### PARAMETERS
    # p_p0   = exp(-elev/8434.5)  ### correction for differences in pressure between basin and seal level
    # Pre = P0*p_p0
    Pre0 = 101325  ##[Pa]
    Pre = Pre0 * np.exp((-9.81 / 287) * (elevation- 0) / (temperature + 273.15))  # [Pa]
    #############################5
    row = 1000  # water density [kg/m^3]
    cp = 1005 + ((temperature + 23.15) ** 2) / 3364  ## specific heat air  [J/kg K]
    L = 1000 * (2501.3 - 2.361 * (temperature))  ### Latent heat vaporization/condensaition [J/kg]
    ##############
    ######################################

    EP_en = 1000 * 3600 * Rn / (L * row)  ## [mm/h] evaporation
    G = cp * Pre / (0.622 * L)  ## Pa/c  psicrometric costant
    D = (4098 * esat) / ((237.3 + temperature) ** 2)  ## Pa/C
    PET = 1.26 * (D / (D + G)) * EP_en  ## [mm/h] Priestly-Taylor
    EP_aer = np.nan
    Epot = np.nan
    Tpot = np.nan

    # Is sps_temperature almost equal to 1.0?
    # the unit of 3600 is second
    if not np.isclose(a = sps_temperature, b = 3600):
        PET = PET * sps_temperature  ##[mm/sps_temperature]
        EP_aer = EP_aer * sps_temperature
        EP_en = EP_en * sps_temperature
        Tpot = Tpot * sps_temperature
        Epot = Epot * sps_temperature

    # PET may be negative due to dew in the winter.
    # However, we do not consider dew and are just interested in positive values.
    PET[PET < 0] = 0

    hydro_output["PET"][:, HRU_id] = PET

    return hydro_output


# Hydrological Model
def h_model(snow, PET, precipitation, temperature, alpha, num_reservoir, params):
    '''
    SedCas hydrological model

    Args:
        snow: data frame from degree-day-model
        PET: potential ET from cal_actual_evap model
        precipitation: data frame, precipitation, physical unit: mm/h
        temperature: data frame, temperature, physical unit: degree C
        alpha: parameter for efficiency of ET dependent on saturation of upper storage,
        num_reservoir: int or float, number of reservoirs, physical unit: none
        params: dict containing {'k':k, 'Scap':Scap, 'S0':S0}
                k : factor for release from linera reservoir, i.e. residence time
                Scap : water storage capacity [mm]
                S0 : initial condition [mm]
                e.g., if n=2 the input shoud be params = dict('k' : [k1, k2], 'Scap' : [sc1, sc2], 'S0' : [s01, s02])
    Returns:
        hyd : dataframe containing time series of...
            - Q : dischagre [mm]
            - Qs : discharge from overland flow [mm]
            - Qss : discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
            - Vw : state of soil water storage [mm]
            - snow : snow depth SWE (snow_water_equivalent) [mm], already in input
            - PET : Potential ET [mm]
            - AET : Actual ET [mm]
    '''

    #  initialization
    index = snow.index
    num_data = len(snow)

    # convert to numpy array
    snow_depth = np.array(snow.modelled_s_depth, dtype=float)
    dsdepth =  np.array(snow.delta_depth, dtype=float)
    snow_acc = np.array(snow.snow_acc, dtype=float)
    snow_melt = np.array(snow.snow_melt, dtype=float)

    PET = np.array(PET, dtype=float)
    precipitation = np.array(precipitation, dtype=float)
    temperature = np.array(temperature, dtype=float)

    Q = np.zeros(num_data)  # dischagre
    Qss = np.zeros(num_data) # discharge from subsurface flow
    Qs = np.zeros(num_data)  # discharge from overland (surface) flow

    # percolation between storages the first column is for the flow between the most upper and the second...
    Qper = np.zeros(shape=(num_data, num_reservoir - 1))

    # assign storage properties
    class storage(object):
        '''
        makes class from parameter inputs
        '''

        def __init__(self, dictionary):
            for key in dictionary:
                setattr(self, key, dictionary[key])

    s = storage(params)

    # array for storage time series, each column represents one reservoir (from bottom to top)
    Vw = np.zeros((num_data, num_reservoir))
    AET = np.zeros(num_data)
    for i in range(num_reservoir):
        Vw[0, i] = s.S0[i]  # initial condition

    # transient storage, discharge computation
    # changes from precipitation and snowmelt (both positive, because they are inputs to the storage)
    # to keep this order is important because there can be snow accumulation and melt on the same day
    dVw = precipitation.copy()  # add precipitation
    dVw = dVw - snow_acc  # when snow accumulation, precipitation is not added to the storage
    dVw = dVw + snow_melt  # where snow melt, add it

    # loop through each time step
    ## scheme:
    # 1) add everything to the most upper storage
    # 2) compute all internal and outflows based on state afer input
    # 3) subtract outflows, account for internal flow
    # 4) if lower is full, it is pushed to the upper
    # 5) check that capacity of upper is not exceeded, else more surface runoff

    # loop through time steps
    for i in range(1, num_data):
        # 1)
        Vw[i, 0] = Vw[i - 1, 0] + dVw[i]

        # 2)
        # AET
        if temperature[i] > 0:
            b = 1.0 - np.exp(-alpha * Vw[i, 0] / float(s.Scap[0]))
            AET[i] = b * PET[i]

        # this is out of temperature loop, should not make a difference, but ensures continuity of water just in case
        if Vw[i, 0] > s.Scap[0]:
            Qs[i] = Vw[i, 0] - s.Scap[0]  # the amount exceeding the capacity is surface runoff

        if temperature[i] > 0:  # condition for subsurface flow, else frozen
            # percolation
            # j --> storage index
            for j in range(num_reservoir - 1):
                Qper[i, j] = Vw[i, j] * 1 / s.k[j]

        # 3)
        # most upper bucket
        if num_reservoir == 1:
            Vw[i, 0] = Vw[i, 0] - AET[i] - Qs[i]
        elif num_reservoir == 2:
            Vw[i, 0] = Vw[i, 0] - AET[i] - Qs[i] - Qper[i]
        else:
            Vw[i, 0] = Vw[i, 0] - AET[i] - Qs[i] - Qper[i, 0]

        # check for continuity of mass
        if Vw[i, 0] < 0:
            d = abs(Vw[i, 0])
            Vw[i, 0] = Vw[i, 0] + AET[i]
            AET[i] = AET[i] - d
            if AET[i] < 0:
                AET[i] = 0
            Vw[i, 0] = Vw[i, 0] - AET[i]
            if Vw[i, 0] < 0:
                d = abs(Vw[i, 0])
                Vw[i, 0] = Vw[i, 0] + Qs[i]
                Qs[i] = Qs[i] - d
                Vw[i, 0] = Vw[i, 0] - Qs[i]
                if Vw[i, 0] < 0:
                    Qs[i] = 0
                    Vw[i, 0] = 0

        # lower buckets
        # for only two buckets, this does nothing
        if num_reservoir > 1:
            if num_reservoir == 2:
                Vw[i, 1] = Vw[i - 1, 1] + Qper[i]

                # check for continuity of mass
                if Vw[i, 1] < 0:
                    d = abs(Vw[i, 0])
                    Vw[i, 1] = Vw[i, 1] + Qper[i, 2]
                    AET[i] = Qper[i, 2] - d
                    if Qper[i, 2] < 0:
                        Qper[i, 2] = 0
                    Vw[i, 1] = Vw[i, 1] - Qper[i, 2]
                    if Vw[i, 1] < 0:
                        Qper[i, 2] = 0
                        Vw[i, 1] = 0

            elif num_reservoir > 2:
                for j in range(1, num_reservoir):
                    Vw[i, j] = Vw[i - 1, j] + Qper[i, j] - Qper[i, j + 1]

                    # check for continuity of mass
                    if Vw[i, j] < 0:
                        d = abs(Vw[i, 0])
                        Vw[i, j] = Vw[i, j] + Qper[i, j + 1]
                        AET[i] = Qper[i, j + 1] - d
                        if Qper[i, j + 1] < 0:
                            Qper[i, j + 1] = 0
                        Vw[i, j] = Vw[i, j] - Qper[i, j + 1]
                        if Vw[i, j] < 0:
                            Qper[i, j + 1] = 0
                            Vw[i, j] = 0

        ## for lowest bucket

        # release from lowest
        Qss[i] = Vw[i, -1] * 1 / s.k[-1]

        if num_reservoir == 1:
            Vw[i, 0] = Vw[i, 0] - Qss[i]
        elif num_reservoir == 2:
            Vw[i, -1] = Vw[i, -1] - Qss[i]
        else:
            Vw[i, -1] = Vw[i, -1] - Qss[i]

        # check for continuity of mass
        if Vw[i, -1] < 0:
            d = abs(Vw[i, -1])
            Vw[i, -1] = Vw[i, -1] + Qss[i]
            Qss[i] = Qss[i] - d
            if Qss[i] < 0:
                Qss[i] = 0
            Vw[i, -1] = Vw[i, -1] - Qss[i]
            if Vw[i, -1] < 0:
                Qs[i] = 0
                Vw[i, -1] = 0

        # 4)
        # loop from lowest to most upper to check for capacitiy exceedence
        for j in reversed(range(1, num_reservoir)):
            if Vw[i, j] > s.Scap[j]:
                up = Vw[i, j] - s.Scap[j]
                Vw[i, j] = s.Scap[j]
                Vw[i, j - 1] = Vw[i, j - 1] + up

        # 5) for most upper bucket
        if num_reservoir == 1:
            pass
        elif Vw[i, 0] > s.Scap[0]:
            q = Vw[i, 0] - s.Scap[0]
            Qs[i] = Qs[i] + q
            Vw[i, 0] = s.Scap[0]

        Q[i] = Qs[i] + Qss[i]

    # total system storage
    Vw_tot = np.sum(Vw, axis=1)

    data = {'Q': Q, # dischagre [mm]
            'Qs': Qs, # discharge from overland (surface) flow [mm]
            'Qss': Qss, # discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
            'Vw': Vw_tot, # state of soil water storage [mm]
            'snow_depth': snow_depth, # snow depth SWE [mm], already in input
            'snowacc': dsdepth, # snow accumulation [mm]
            'PET': PET, # Potential ET [mm]
            'AET': AET, # Actual ET [mm]
            'Pr': precipitation, # rainfall [mm]
            'temperature': temperature # [degree C]
            }


    # single bucket storage
    i = 0
    for col in Vw.T:
        name = 'Vw%i' % i
        data[name] = col
        i += 1

    hyd = pd.DataFrame(data=data, index=index)

    return hyd


# Lumped Hydrological Model Results
def lump_h_model(HYM, num_HRU, shares, log_print):

    # create an empty df with the same column and index name as hydro
    hydro = pd.DataFrame(columns=HYM[0].columns, index=HYM[0].index)

    for column in hydro.columns:

        try:
            lumped = []
            for HRU_id in range(num_HRU):
                # sum the different HRU part based on the shares values
                temp = np.array(HYM[HRU_id][column], dtype=float) * shares[HRU_id] # return as
                lumped.append(temp)

            lumped = np.array(lumped)
            hydro[column] = np.sum(lumped, axis=0)

        except KeyError as e:
            # careful the classic Python gotcha, e != 'Vw1'
            if e.args[0] == 'Vw1':
                # the model do not have 'Vw1' wh HRU=1
                # ingore this
                pass
            else:
                print(e)


        # check the column contains string "Vw" or not
        if "Vw" in column:
            # contain something like, "Vw", 'Vw0', 'Vw1'
            if column == "Vw":
                # only keep the column 'Vw'
                pass
            else:
                # drop the column like 'Vw0', 'Vw1'
                hydro.drop(columns=[column], inplace=True)  # drop
        else:
            # Do not contain
            pass  # keep

    return hydro