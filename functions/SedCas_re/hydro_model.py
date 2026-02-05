#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# __note__ = This code is adapted from SedCas (Author: Jacob Hirschberg, Created: 2022-02-03, Source: https://github.com/jacobhirschberg/SedCas)
#           and is distributed under the terms of the GNU General Public License v3.0 (GPL-3.0).

import math
import random

import pandas as pd
import numpy as np

from obspy import UTCDateTime
import xarray as xr

data = pd.read_csv(f"/Users/qizhou/#python/stTwin/data/SedCas_input/climate_2004_2017_h.txt", header=0)

time = [UTCDateTime(i).timestamp for i in data.iloc[:, 1]]
time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in data.iloc[:, 1]]

# Extract variables
precipitation = data.iloc[:, 2].values
temperature = data.iloc[:, 3].values
sun_radiation = data.iloc[:, 4].values

data_source = "MeteoSwiss"
station = data.iloc[0, 0]  # station name
resolution = time[1] - time[0]  # unit is second
time_now = UTCDateTime().isoformat()

climate_forcing = xr.Dataset(
    coords={
        "time": ("time", np.array(time)),  # numeric UTC+0 time
        "time_str": ("time", np.array(time_str)),  # string UTC+0 time
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
                   {"units": "mm??",
                    "description": "Snow (when covered with snow) or soil (when no more snow existing) albedo"}),

        # (2)
        # Potential evapotranspiration [mm]
        "PET": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                {"units": "mm??", "description": "Potential evapotranspiration"}),

        # Actual evapotranspiration [mm]
        "AET": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                {"units": "mm??", "description": "Actual evapotranspiration"}),

        # (3)
        # discharge [mm]
        "Q": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
              {"units": "mm??", "description": "Total discharge"}),

        # discharge from overland flow [mm]
        "Qs": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
               {"units": "mm??", "description": "Discharge from overland (surface) flow"}),

        # discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
        "Qss": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                {"units": "mm??", "description": "Discharge from subsurface flow"}),

        # state of soil water storage [mm]
        "soil_water_storage": (("time", "HRU_id"), np.zeros((num_data, num_HUR)),
                               {"units": "mm??", "description": "State of soil water storage"}),

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
def snow_water_equivalent(temperature, precipitation,
                          snow_melt_r, T_theta_a, T_theta_m, snow_albedo, soil_albedo):
    # snow accumulation
    cold_cond = temperature <= T_theta_a  # cold condition
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
    modelled_SWE = np.zeros(len(precipitation))  # modelled snow water equivalent [mm]
    delta_depth = np.zeros(len(precipitation))  # snow pack change SWE

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

    return modelled_SWE, delta_depth, snow_acc, snow_melt, albedo


# Potential Evapotranspiration Module
def potential_et(temperature, sun_radiation, albedo, sps_temperature, cloud_cover_r, elevation, U):
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
    Pre = Pre0 * np.exp((-9.81 / 287) * (elevation - 0) / (temperature + 273.15))  # [Pa]
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

    # Is sps_temperature almost equal to 1.0?
    # the unit of 3600 is second
    if not np.isclose(a=sps_temperature, b=3600):
        PET = PET * sps_temperature  ##[mm/sps_temperature]

    # PET may be negative due to dew in the winter.
    # However, we do not consider dew and are just interested in positive values.
    PET[PET < 0] = 0

    return PET


# Hydrological Model
def single_reservoir_o(w_in, initial_w_storage, temperature,
                       PET, alpha,
                       w_storage_cap, w_residence_time):
    '''
    Simulate water discharge for a single reservoir at one time step.
    All the input and output is 1D.

    Args:
        w_in: float, net water input (precipitation - snow_acc + snow_melt) [mm]
        initial_w_storage: float, water storage from previous timestep [mm]
        temperature: float, air temperature [°C]
        PET: float, potential evapotranspiration [mm]
        alpha: float, parameter controlling AET efficiency (dimensionless)
        w_storage_cap: float, reservoir storage capacity [mm]
        w_residence_time: float, mean residence time of water [hours or timesteps]

    Returns:
        Qs: float, surface runoff [mm]
        Qss: float, subsurface discharge [mm]
        AET: float, actual evapotranspiration [mm]
        c_w_storage: float, updated water storage for next timestep [mm]
    '''

    # (1) update the current water storage for the reservoir
    c_w_storage = initial_w_storage + w_in

    # (2) update the AET
    # when no forzen, there will AET, Qss, but NO Qpercolation
    if temperature > 0:
        # theoretical AET
        b = 1.0 - np.exp(-alpha * c_w_storage / w_storage_cap)
        AET = b * PET
        # theoretical Qss
        Qss = c_w_storage / w_residence_time
    else:
        AET = 0
        Qss = 0

    # (3) check the theoretical valus for AET, Qss
    if AET + Qss > c_w_storage:
        # not enoungh water, then reduce the AET and Qss with priority:
        # (a) AET takes priority (gets what it needs first)
        # (b) Qss gets whatever remains
        # e.g., AET (10) + Qss (20) > c_w_storage (25)
        available_water1 = c_w_storage  # available_water1 -> 25
        AET = min(AET, available_water1)  # AET -> 10

        available_water2 = c_w_storage - AET  # available_water2 -> 15
        Qss = min(Qss, available_water2)  # Qss -> 15
    else:
        pass
    # update the c_w_storage, now c_w_storage >=0 always true
    c_w_storage = c_w_storage - AET - Qss

    # (4) update the surface flow Qs
    if c_w_storage >= w_storage_cap:
        Qs = c_w_storage - w_storage_cap
    else:
        Qs = 0

    # (5) update the c_w_storage,
    # now it equals:
    # initial_w_storage + w_in - AET - Qss - Qs
    c_w_storage = c_w_storage - Qs

    return Qs, Qss, AET, c_w_storage


def single_reservoir_a(w_in, initial_w_storage, temperature,
                       PET, alpha,
                       w_storage_cap, w_residence_time):
    '''
    Simulate water discharge for a single reservoir at one time step.
    All the input and output is 1D.

    Notice: there is a logic bug.
    In real case, AET, Qs, and Qss are happens at the same time,
    which will lead "initial_w_storage + w_in < AET + Qss + Qs".
    If this happens, we assign the AET, Qs, and Qss with a fixed ratio of "initial_w_storage + w_in".

    Args:
        w_in: float, net water input (precipitation - snow_acc + snow_melt) [mm]
        initial_w_storage: float, water storage from previous timestep [mm]
        temperature: float, air temperature [°C]
        PET: float, potential evapotranspiration [mm]
        alpha: float, parameter controlling AET efficiency (dimensionless)
        w_storage_cap: float, reservoir storage capacity [mm]
        w_residence_time: float, mean residence time of water [hours or timesteps]

    Returns:
        Qs: float, surface runoff [mm]
        Qss: float, subsurface discharge [mm]
        AET: float, actual evapotranspiration [mm]
        c_w_storage: float, updated water storage for next timestep [mm]
    '''

    # (1) update the current water storage for the reservoir
    c_w_storage = initial_w_storage + w_in

    # (2) update the AET
    # when no forzen, there will AET, Qss, but NO Qpercolation
    if temperature > 0:
        # theoretical AET
        b = 1.0 - np.exp(-alpha * c_w_storage / w_storage_cap)
        AET = b * PET
    else:
        AET = 0

    # (3) update the Qss, Qs
    if c_w_storage >= w_storage_cap:
        # Notice: if the c_w_storage -> infity, Qss -> infity
        Qss = c_w_storage / w_residence_time
        Qs = c_w_storage - w_storage_cap
    else:
        Qss = 0
        Qs = 0

    # (4) update the c_w_storage
    # the final c_w_storage = initial_w_storage + w_in - AET - Qss - Qs
    if AET + Qss + Qs > c_w_storage:
        # when water supply is insufficient,
        # then reduce the AET, Qss, Qs as part of c_w_storage
        AET, Qss, Qs = c_w_storage / 2, c_w_storage / 4, c_w_storage / 4
        print(f"Warning! AET + Qss + Qs > c_w_storage"
              f"AET={AET}, Qss={Qss}, Qs={Qs}, c_w_storage={c_w_storage}")
    else:
        pass
    c_w_storage = c_w_storage - AET - Qss - Qs

    return Qs, Qss, AET, c_w_storage


def check_water_balance(water_t0, water_t1, Qss, Qs, print_log=True):
    delta_t0_to_t1 = water_t0 - water_t1

    if delta_t0_to_t1 > 0:
        # when water is lost, probably,
        # manually add it back
        Qss, Qs = Qss + (delta_t0_to_t1 / 2), Qs + (delta_t0_to_t1 / 2)

        if print_log is True:
            print(f"Warning <check_water_balance>!\n"
                  f"delta_t0_to_t1={delta_t0_to_t1} > 0, "
                  f"The water before {water_t0} "
                  f"and after {water_t1} update not equal.")
    elif delta_t0_to_t1 < 0:
        # when water is gain, not possible,
        # manually remove it back
        Qss, Qs = Qss - (delta_t0_to_t1 / 2), Qs - (delta_t0_to_t1 / 2)
        if print_log is True:
            print(f"Warning <check_water_balance>!\n"
                  f"delta_t0_to_t1={delta_t0_to_t1} < 0, "
                  f"The water before {water_t0} "
                  f"and after {water_t1} update not equal.")
    else:
        # all good
        pass

    return Qss, Qs


def single_reservoir(w_in, initial_w_storage, temperature,
                     PET, alpha,
                     w_storage_cap, w_residence_time):
    # give pority AET, Qss, Qs

    # (1) update the current water storage for the reservoir
    c_w_storage = initial_w_storage.copy() + w_in.copy()

    # (2) update the AET
    # when no forzen, there will AET, Qss, but NO Qpercolation with one reservoir
    if temperature > 0:
        # theoretical AET
        b = 1.0 - np.exp(-alpha * c_w_storage / w_storage_cap)
        AET = b * PET
        AET = min(AET, c_w_storage)
    else:
        AET = 0
    # update the reservoir
    c_w_storage = c_w_storage - AET

    # (3) update the Qss
    if c_w_storage > w_storage_cap:
        # theoretical Qss
        # if the c_w_storage -> infity, Qss -> infity
        # it can be greater than c_w_storage
        Qss_pot = c_w_storage / w_residence_time
        Qss = min(Qss_pot, c_w_storage)
    else:
        Qss = 0
    # update the reservoir
    c_w_storage = c_w_storage - Qss

    # (4) update the Qs
    if c_w_storage > w_storage_cap:
        Qs = c_w_storage - w_storage_cap  # positive value
    else:
        Qs = 0
    # update the reservoir
    c_w_storage_updated = c_w_storage - Qs

    # (5) make sure the water in-out balance
    water_t0 = initial_w_storage.copy() + w_in.copy()  # water before update
    water_t1 = AET + Qss + Qs + c_w_storage_updated  # water after update

    assert water_t0 == water_t1, (f"Warning!\n"
                                  f"Water in-out not balance."
                                  f"water_t0={water_t0}, water_t1={water_t1}")

    return AET, Qss, Qs, c_w_storage_updated


def multiple_reservoir(w_in, initial_w_storage, temperature,
                       PET, alpha,
                       w_storage_cap, w_residence_time):
    # (0) prepare the params
    num_reservoir = len(w_storage_cap)
    # percolation container
    Qperc = np.zeros(num_reservoir - 1)

    # (1) update the current water storage for the reservoir
    c_w_storage = initial_w_storage.copy()
    c_w_storage[0] = c_w_storage[0] + w_in.copy()  # add water input to top reservoir

    # (2) update the AET
    # when no forzen, there will AET, Qss, and Qpercolation
    if temperature > 0:
        # theoretical AET
        b = 1.0 - np.exp(-alpha * c_w_storage[0] / w_storage_cap[0])
        AET = b * PET
        AET = min(AET, c_w_storage[0])  # limit the AET
    else:
        AET = 0
    # update the top reservoir
    c_w_storage[0] = c_w_storage[0] - AET

    # (3) update the percolation
    # when no forzen, there will AET, Qss, and Qpercolation
    if temperature > 0:
        # theoretical percolation
        for j in range(num_reservoir - 1):  # loop all Non-last reservoirs
            # ensure percolation (Qperc[j]) from reservoir j to j+1 does not
            # exceed available water (c_w_storage[j]) in reservoir j
            Qperc[j] = min(c_w_storage[j] / w_residence_time[j], c_w_storage[j])

            # if you want to modify how the water percolate,
            # you can replace the previous part as follows, the physical means:
            # if the reservoir is not saturated, it can not transfer water to lower reservoir
            # if c_w_storage[j] > w_storage_cap[j]:
            #     Qperc[j] = min(c_w_storage[j] / w_residence_time[j], c_w_storage[j])
            # else:
            #     Qperc[j] = 0 # repeat to improve the code readablity

    assert np.min(Qperc) >= 0, f"Warning!\n The percolation has negative value.\n{Qperc}"
    assert np.min(c_w_storage) >= 0, f"Warning!\n The c_w_storage has negative value.\n{c_w_storage}"

    # (3) update all reservoirs with percolation
    for j in range(num_reservoir):
        if j == 0:
            # first reservoir
            # # remove the percolation
            c_w_storage[j] = c_w_storage[j] - Qperc[j]  # incase negative water storage
        elif 1 <= j < num_reservoir - 1:
            # middle reservoir
            # receive the previous reservoir and remove the percolation
            c_w_storage[j] = Qperc[j - 1] + c_w_storage[j] - Qperc[j]  # incase negative water storage
        else:
            # last reservoir
            # receive the previous reservoir
            c_w_storage[j] = Qperc[j - 1] + c_w_storage[j]

    # (4) update the Qss from the last reserior
    if c_w_storage[-1] > w_storage_cap[-1]:
        # theoretical Qss
        # if the c_w_storage -> infity, Qss -> infity
        # it can be greater than c_w_storage
        Qss_pot = c_w_storage[-1] / w_residence_time[-1]
        Qss = min(Qss_pot, c_w_storage[-1])
    else:
        Qss = 0
    # update the last reservoir
    c_w_storage[-1] = c_w_storage[-1] - Qss

    # (5) update the Qs from the top reserior
    # there are two cases will gernerate the Qs,
    # Infiltration-excess runoff,
    # (a) top reservoir saturated, AND,
    # (b) rain or snow melt input exceeds percolation capacity, AND
    # (c) Deeper layer not saturated
    #
    # Saturation-excess runoff
    # (a) top reservoir saturated, AND
    # (b) deeper layers also saturated

    if c_w_storage[0] > w_storage_cap[0]:
        # top reservoir is saturated and cannot drain downward anymore
        Qs = c_w_storage[0] - w_storage_cap[0]
    else:
        Qs = 0
    # update the top reservoir
    c_w_storage[0] = c_w_storage[0] - Qs

    # (6) make sure the water in-out balance
    water_t0 = initial_w_storage[0] + w_in + np.sum(initial_w_storage[1:])  # water before update
    water_t1 = np.sum(c_w_storage) + Qs + Qss  # water after update

    assert water_t0 == water_t1, (f"Warning!\n"
                                  f"Water in-out not balance."
                                  f"water_t0={water_t0}, water_t1={water_t1}")

    c_w_storage_updated = c_w_storage

    return AET, Qss, Qs, c_w_storage_updated



def h_model(snow_acc, snow_melt, PET, precipitation, temperature, alpha, initial_w_storage, w_storage_cap,
            w_residence_time):
    # initial_w_storage = S0, w_storage_cap = Scap, w_residence_time = k

    #  initialization
    num_data = len(snow_acc)
    num_reservoir = len(w_storage_cap)
    if num_reservoir == 1:
        h_func = single_reservoir
    else:
        h_func = multiple_reservoir

    # convert to numpy array
    Q = np.zeros(num_data)  # dischagre
    Qs = np.zeros(num_data)  # discharge from overland (surface) flow
    Qss = np.zeros(num_data)  # discharge from subsurface flow
    AET = np.zeros(num_data)  # Actural et

    # water store in soli, each column represents one reservoir (from bottom to top)
    Vw = np.zeros((num_data, num_reservoir))
    # initial condition
    for i in range(num_reservoir):
        w_in = precipitation[i] - snow_acc[i] + snow_melt[i]

        AET_t, Qss_t, Qs_t, c_w_storage_updated = h_func(w_in, initial_w_storage, temperature[i],
                                                         PET[i], alpha,
                                                         w_storage_cap, w_residence_time)
        initial_w_storage = c_w_storage_updated
        Qs[i] = Qs_t
        Qss[i] = Qss_t
        AET[i] = AET_t
        Q[i] = Qs_t + Qss_t
        Vw[i, :] = c_w_storage_updated

    # total system storage
    Vw_tot = np.sum(Vw, axis=1)

    data = {'Q': Q,  # dischagre [mm]
            'Qs': Qs,  # discharge from overland (surface) flow [mm]
            'Qss': Qss,  # discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
            'Vw': Vw_tot,  # state of total water storage in solis [mm]
            'AET': AET,  # Actual ET [mm]
            }

    # single bucket storage
    i = 0
    for col in Vw.T:
        name = 'Vw%i' % i
        data[name] = col
        i += 1

    hyd = pd.DataFrame(data=data)

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
                temp = np.array(HYM[HRU_id][column], dtype=float) * shares[HRU_id]  # return as
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


HRU_id = 0
snow_melt_r, T_theta_a, T_theta_m = 0.08, 0.5, 0.6
snow_albedo, soil_albedo = [0.4, 0.65], [0.15, 0.25]

# the temperature, precipitation are from climate_forcing (xarray.Dataset)
temperature = climate_forcing.temperature.values  # unit by degree per delta_t
precipitation = climate_forcing.precipitation.values  # unit by mm per delta_t

modelled_SWE, delta_depth, snow_acc, snow_melt, albedo = snow_water_equivalent(temperature, precipitation, snow_melt_r,
                                                                               T_theta_a, T_theta_m,
                                                                               snow_albedo[HRU_id], soil_albedo[HRU_id])

# update the xarray.Dataset
hydro_output["modelled_SWE"][:, HRU_id] = modelled_SWE
hydro_output["delta_depth"][:, HRU_id] = delta_depth
hydro_output["snow_acc"][:, HRU_id] = snow_acc
hydro_output["snow_melt"][:, HRU_id] = snow_melt
hydro_output["albedo"][:, HRU_id] = albedo

cloud_cover_r, elevation, U = 1, 1600.0, 0.8

# the temperature, precipitation are from climate_forcing (xarray.Dataset)
temperature = climate_forcing.temperature.values  # unit by degree per delta_t
sun_radiation = climate_forcing.sun_radiation.values  # unit by mm per delta_t
albedo = hydro_output.albedo[:, HRU_id].values
sps_temperature = climate_forcing.attrs["resolution"]
PET = potential_et(temperature, sun_radiation, albedo, sps_temperature, cloud_cover_r, elevation, U)
hydro_output["PET"][:, HRU_id] = PET
hydro_output["PET"].sel(HRU_id=1,
                        time=slice(float(UTCDateTime("2004-03-13T11:00:00")),
                                   float(UTCDateTime("2004-03-14T11:00:00")))
                        )

# one reservoir
alpha, initial_w_storage, w_storage_cap, w_residence_time = 20.0, [0], [4], [23]
hyd = h_model(snow_acc, snow_melt, PET, precipitation, temperature,
              alpha, initial_w_storage, w_storage_cap, w_residence_time)

# multiple reservoir
alpha, initial_w_storage, w_storage_cap, w_residence_time = 20.0, [0, 0], [72, 27], [94, 235]
hyd = h_model(snow_acc, snow_melt, PET, precipitation, temperature,
              alpha, initial_w_storage, w_storage_cap, w_residence_time)
