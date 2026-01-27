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


# Snow Degree-Day-Module
def snow_water_equivalent(temperature, precipitation, melt_rate_f, T_theta_a, T_theta_m, snow_albedo, soil_albedo):
    '''
    Computing snow depth in snow-water-equivalent (SWE) based on a degree-day-snow-melt-model

    Args:
        temperature : Temperature time series [degree C]
        precipitation : Precipitation time series [mm]
        melt_rate_f : melt rate factor [mm/degreeC/t]
        T_theta_a : temperature threshold for snow accumulation [degree C]
        T_theta_m : temperature threshold for snow melt [degree C]
        snow_albedo : snow albedo [-]
        soil_albedo : soil albedo [-]

    Returns:
        df : dataframe containing
            - ddepth : daily snow pack change SWE [mm]
            - depth : modelled SWE snow depth SWE [mm]
            - acc : snow pack accumulation SWE [mm]
            - melt : melt from snow pack SWE [mm]

    Note: acc and melt is not necessary equal to ddepth because there can be both
    at the same time step (if accumulation and melt temperature threshold are not equal)
    '''

    index = temperature.index
    temperature = np.array(temperature, dtype=float) # unit by degree
    precipitation = np.array(precipitation, dtype=float) # unit by mm per hour, Note this may change to mm per minutes

    # snow accumulation
    cold_cond = temperature <= T_theta_a # cold condition
    warm_cond = temperature > T_theta_a
    snow_acc = precipitation.copy()
    snow_acc[warm_cond] = 0.0  # no snow pack accumulation where temperature is above

    # snow melt
    # melt_cond = temperature > T_theta_m
    non_melt_cond = temperature <= T_theta_m

    T_grad = temperature - T_theta_m  # melting gradient
    T_grad[non_melt_cond] = 0.0  # no snowmelt where temperature is below
    snow_melt = melt_rate_f * T_grad  # potential snowmelt based on temperature

    # compute actual snow depth
    modelled_s_depth = np.zeros(len(precipitation)) # modelled snow depth SWE [mm]
    delta_depth = np.zeros(len(precipitation)) # daily (?? really ?? QZ) snow pack change SWE

    for i in range(1, len(precipitation)):
        modelled_s_depth[i] = modelled_s_depth[i - 1] + snow_acc[i] - snow_melt[i]

        if modelled_s_depth[i] < 0:
            snow_melt[i] = snow_melt[i] - abs(modelled_s_depth[i])
            modelled_s_depth[i] = 0

        delta_depth[i] = modelled_s_depth[i] - modelled_s_depth[i - 1]

    # Albedo
    albedo = np.zeros(len(modelled_s_depth))

    with_snow_cond = modelled_s_depth > 0
    albedo[with_snow_cond] = snow_albedo

    without_snow_cond = modelled_s_depth <= 0
    albedo[without_snow_cond] = soil_albedo

    # prepare the output
    data = {'modelled_s_depth': modelled_s_depth,
            'delta_depth': delta_depth,
            'snow_acc': snow_acc,
            'snow_melt': snow_melt,
            'albedo': albedo
            }

    df = pd.DataFrame(data=data, index=index)

    return df


# Potential Evapotranspiration Module
def cal_actual_evap(temperature, sps_temperature, radiation, cloud_cover_r, albedo, elevation, U):
    '''
    Calculate the daily potential evapotranspiration (PET) by Priestly Taylor method.
    E(t) = \gammar \cdot PET(t)

    Note:
        the Penmann-Monteith method can be found at the source code <https://github.com/jacobhirschberg/SedCas.git>.

    Args:
        temperature: data array, time series temperature, physical unit: degree C
        sps_temperature: int or float, temporal resolution of the temperature data, physical unit: None
                         e.g., sps_temperature=24 for dayily data;
                               sps_temperature=1 for hourly data,
        radiation: radiation, physical unit: W/m^2 (watt per squared meter)
        cloud_cover_r: cloud cover fraction (ratio), physical unit: None
        albedo: data array, time series albedo, physical unit: None
        elevation: data array, meters above sea level (elevation), physical unit: m
        U: ??, what's this?

    Returns:
        PET: data array, daily potential evapotranspiration, physical unit: mm/sps_temperature,
             the temporal resolution is same as temperature.
    '''

    # convert to numpy array
    temperature = np.array(temperature, dtype=float)
    radiation = np.array(radiation, dtype=float)
    albedo = np.array(albedo, dtype=float)

    esat = 611 * np.exp(17.27 * temperature / (237.3 + temperature))  ## Vapor Pressure Saturation
    ea = U * esat

    #### Net Radiation
    si = 5.6704 * (10 ** -8)  # Stefan-Boltzman Constant [W/m**2.K**4]
    ######
    K = 0.1 + 0.9 * (1 - 0.6 * (cloud_cover_r ** 2.5))  #### Emissiivty coefficient cloud
    ei = 0.34 - 0.14 * np.sqrt(ea / 1000)  ## Net emissivity humidity
    ######
    D_Rlw = ei * K * si * ((temperature - 273.15) ** 4)  ## Net Longwave Radiatio W/m^2
    Rn = radiation * (1 - albedo) - D_Rlw  ## Net Radiation  W/m^2

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
    if not np.isclose(a = sps_temperature, b = 1.0):
        PET = PET * sps_temperature  ##[mm/sps_temperature]
        EP_aer = EP_aer * sps_temperature
        EP_en = EP_en * sps_temperature
        Tpot = Tpot * sps_temperature
        Epot = Epot * sps_temperature

    # PET may be negative due to dew in the winter.
    # However, we do not consider dew and are just interested in positive values.
    PET[PET < 0] = 0

    return PET


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
    Qss = np.zeros(num_data) # discharge from overland (surface) flow
    Qs = np.zeros(num_data)  # discharge from subsurface flow

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
            'Qs': Qs, # discharge from overland flow [mm]
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
            print(f"Error={e} \n \n")


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