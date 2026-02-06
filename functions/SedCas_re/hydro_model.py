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


# Snow water equivalent module
def snow_water_equivalent(temperature, precipitation,
                          snow_melt_r, T_theta_a, T_theta_m, snow_albedo, soil_albedo):
    """
    Compute snow water equivalent (SWE) and surface albedo using a temperature-index snow model.

    Snow accumulates when air temperature is below the accumulation threshold
    and melts when temperature exceeds the melt threshold.

    Args:
        temperature (np.ndarray): Air temperature time series [°C, consistent with thresholds], shape (n_time,).
        precipitation (np.ndarray): Total precipitation time series [mm per time step], shape (n_time,).
        snow_melt_r (float): Snowmelt rate factor [mm / (time step · °C)].

        T_theta_a (float): Temperature threshold for snow accumulation [°C].
        Precipitation falls as snow when temperature ≤ T_theta_a.

        T_theta_m (float): Temperature threshold for snowmelt [°C].
        Snow melts when temperature > T_theta_m.

        snow_albedo (float): Surface albedo when snow cover exists [-].
        soil_albedo (float): Surface albedo when no snow cover exists [-].

    Returns:
        modelled_SWE (np.ndarray): Snow water equivalent stored on the ground at each time step [mm], shape (n_time,).
        delta_depth (np.ndarray): Change in SWE per time step [mm / time step], shape (n_time,).

        snow_acc (np.ndarray): Snow accumulation per time step [mm / time step], shape (n_time,).
        snow_melt (np.ndarray): Actual snowmelt per time step [mm / time step], shape (n_time,).
        albedo (np.ndarray): Surface albedo time series [-], shape (n_time,).

    Notes:
        - The model is evaluated sequentially in time and depends on previous SWE states.
        - Snowmelt is limited by available snowpack to ensure mass conservation.
        - All returned arrays are newly allocated; input arrays are not modified in place.
    """

    # snow accumulation
    cold_cond = temperature <= T_theta_a  # cold condition
    warm_cond = temperature > T_theta_a
    snow_acc = precipitation.copy()
    snow_acc[warm_cond] = 0.0  # no snow pack accumulation where temperature is above

    # snow melt
    melt_cond = temperature > T_theta_m
    non_melt_cond = temperature <= T_theta_m
    T_grad = temperature.copy() - T_theta_m  # melting gradient
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


# Potential evapotranspiration module
def potential_et(temperature, sun_radiation, albedo, sps_temperature, cloud_cover_r, elevation, relative_humidity):
    """
    Compute potential evapotranspiration (PET) using a Priestley–Taylor–type energy balance approach.

    PET is estimated from net radiation, air temperature, humidity,
    and elevation. Negative PET values (e.g., dew formation) are truncated to zero.

    Args:
        temperature (np.ndarray): Air temperature time series [°C], shape (n_time,).
        sun_radiation (np.ndarray): Incoming shortwave solar radiation [W m⁻²], shape (n_time,).
        albedo (np.ndarray): Surface albedo [-], shape (n_time,).

        sps_temperature (float): Length of the simulation time step [s]. If not equal to 3600 s, PET is scaled accordingly.
        cloud_cover_r (float): Cloud cover coefficient [-], used to estimate effective atmospheric emissivity.
        elevation (float): Catchment mean elevation above sea level [m].
        relative_humidity (float): Relative humidity fraction [-], range [0 extremely dry –> 1 air is saturated].

    Returns:
        PET (np.ndarray): Potential evapotranspiration per time step [mm], shape (n_time,).

    Notes:
        - Net radiation includes shortwave absorption and longwave emission.
        - Atmospheric pressure is corrected for elevation.
        - The Priestley–Taylor coefficient is fixed at 1.26.
        - PET values less than zero are set to zero.
        - The function does not modify input arrays in place.
    """

    esat = 611 * np.exp(17.27 * temperature / (237.3 + temperature))  ## Vapor Pressure Saturation
    ea = relative_humidity * esat

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
    # the unit of sps_temperature and 3600 is second
    if not np.isclose(a=sps_temperature, b=3600):
        PET = PET * sps_temperature  ##[mm/sps_temperature]

    # PET may be negative due to dew in the winter.
    # However, we do not consider dew and are just interested in positive values.
    PET[PET < 0] = 0

    return PET


# Hydrological module
def single_reservoir(w_in, temperature,
                     PET, alpha,
                     initial_w_storage, w_storage_cap, w_residence_time,
                     tolerance=1e-6):
    """
    Compute water balance for a single linear soil reservoir with priority-based outflows.

    The model updates reservoir storage sequentially within one time step following the priority order:
    1) Actual evapotranspiration (AET)
    2) Subsurface flow (Qss)
    3) Surface runoff (Qs)

    All fluxes are limited by available water to ensure mass conservation.

    Args:
        w_in (float): Liquid water input to the reservoir during the time step (e.g. rainfall + snowmelt) [mm / time step].

        temperature (float): Air temperature at the current time step [°C].
        Evapotranspiration occurs only when temperature > 0.

        PET (float): Potential evapotranspiration for the time step [mm / time step].
        alpha (float): Shape parameter controlling the nonlinear relationship between soil moisture and AET [-].

        initial_w_storage (float): Water stored in the reservoir at the previous time step [mm].
        w_storage_cap (float): Maximum water storage capacity of the reservoir [mm].
        w_residence_time (float): Mean residence time of water in the reservoir under saturated conditions [time step].

        tolerance (float, optional): Numerical tolerance for water balance closure check. Default is 1e-6 [mm].

    Returns:
        AET (float): Actual evapotranspiration during the time step [mm].

        Qss (float): Subsurface (baseflow) discharge from the reservoir [mm].
        Qs (float): Surface runoff generated by storage exceeding capacity [mm].
        c_w_storage_updated (float): Updated reservoir water storage at the end of the time step [mm].

    Notes:
        - Evapotranspiration is computed first and limited by available storage.
        - Subsurface flow follows a linear reservoir formulation.
        - Surface runoff occurs only after AET and Qss have been satisfied.
        - The function enforces strict water mass conservation within a tolerance.
        - Inputs are not modified in place.
    """

    # (1) update the current water storage for the reservoir
    c_w_storage = initial_w_storage.copy() + w_in

    # (2) update the AET
    # when no forzen, there will AET, Qss, but NO Qpercolation with one reservoir
    if temperature > 0:
        # theoretical AET
        b = 1.0 - np.exp(-alpha * c_w_storage / w_storage_cap)
        AET = min(b * PET, c_w_storage)
        # print(b, PET, c_w_storage,  w_storage_cap)
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
    water_t0 = initial_w_storage.copy() + w_in  # water before update
    water_t1 = AET + Qss + Qs + c_w_storage_updated  # water after update

    assert abs(water_t0 - water_t1) < tolerance, (f"Warning in single_reservoir!\n"
                                                  f"Water in-out not balance under tolerance = {tolerance}.\n"
                                                  f"water_t0={water_t0}, water_t1={water_t1}")

    return AET, Qss, Qs, c_w_storage_updated


def multiple_reservoir(w_in, temperature,
                       PET, alpha,
                       initial_w_storage, w_storage_cap, w_residence_time,
                       tolerance=1e-6):
    """
    Compute water balance for a vertically stacked system of linear soil reservoirs.

    The model represents an HRU as a cascade of vertically connected reservoirs.
    Water inputs enter the top reservoir and are redistributed sequentially
    within one time step following this priority order:
    1) Actual evapotranspiration (AET) from the top reservoir
    2) Vertical percolation between reservoirs
    3) Subsurface discharge (Qss) from the deepest reservoir
    4) Surface runoff (Qs) from the top reservoir

    All fluxes are limited by available water to ensure mass conservation.

    Args:
        w_in (float): Liquid water input to the reservoir during the time step (e.g. rainfall + snowmelt) [mm / time step].

        temperature (float): Air temperature at the current time step [°C].
        Evapotranspiration occurs only when temperature > 0.

        PET (float): Potential evapotranspiration for the time step [mm / time step].
        alpha (float): Shape parameter controlling the nonlinear relationship between soil moisture and AET [-].

        initial_w_storage (float): Water stored in the reservoir at the previous time step [mm].
        Reservoirs are ordered from top to bottom, shape (num_reservoirs,).

        w_storage_cap (float): Maximum water storage capacity of the reservoir [mm], shape (num_reservoirs,).
        w_residence_time (float): Mean residence time of water in the reservoir under saturated conditions [time step],
        shape (num_reservoirs,).

        tolerance (float, optional): Numerical tolerance for water balance closure check. Default is 1e-6 [mm].

    Returns:
        AET (float): Actual evapotranspiration during the time step [mm].

        Qss (float): Subsurface (baseflow) discharge from the reservoir [mm].
        Qs (float): Surface runoff generated by storage exceeding capacity [mm].
        c_w_storage_updated (float): Updated reservoir water storage at the end of the time step [mm], shape (n_reservoirs,).

    Notes:
        - Evapotranspiration is computed first and limited by available storage.
        - Vertical percolation follows a linear reservoir formulation and is limited by available storage.
        - Subsurface discharge (Qss) occurs only from the deepest reservoir.
        - Surface runoff (Qs) is generated when the top reservoir exceeds its storage capacity,
          representing infiltration- or saturation-excess runoff.
        - The function enforces strict water mass conservation within a tolerance.
        - Inputs are not modified in place.
    """

    # (0) prepare the params
    num_reservoir = len(w_storage_cap)
    # percolation container
    Qperc = np.zeros(num_reservoir - 1)

    # (1) update the current water storage for the reservoir
    c_w_storage = initial_w_storage.copy()
    c_w_storage[0] = c_w_storage[0] + w_in  # add water input to top reservoir

    # (2) update the AET
    # when no forzen, there will AET, Qss, and Qpercolation
    if temperature > 0:
        # theoretical AET
        b = 1.0 - np.exp(-alpha * c_w_storage[0] / w_storage_cap[0])
        AET = min(b * PET, c_w_storage[0])  # limit the AET
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
    water_t0 = w_in + np.sum(initial_w_storage.copy())  # water before update
    water_t1 = np.sum(c_w_storage) + Qs + Qss + AET  # water after update

    assert abs(water_t0 - water_t1) < tolerance, (f"Warning in multiple_reservoir!\n"
                                                  f"Water in-out not balance under tolerance = {tolerance}.\n"
                                                  f"water_t0={water_t0}, water_t1={water_t1}")

    c_w_storage_updated = c_w_storage

    return AET, Qss, Qs, c_w_storage_updated


def h_model(snow_acc, snow_melt,
            temperature, precipitation,
            PET, alpha,
            initial_w_storage, w_storage_cap, w_residence_time):
    """
    Compute coupled precipitation–snow-soil–runoff water balance using a conceptual reservoir model.

    This function integrates precipitation, snow accumulation, and snow melt with a soil water balance
    model composed of one or multiple linear reservoirs. The model is evaluated sequentially in time,
    enforcing mass conservation at each time step.

    At each time step, liquid water input to the soil system is computed as:
        w_in = precipitation - snow_acc + snow_melt

    The soil water balance is then solved using either a single-reservoir or
    multi-reservoir formulation, depending on the number of reservoirs specified.

    Args:
        snow_acc (np.ndarray): Snow accumulation per time step [mm / time step], shape (n_time,).
        snow_melt (np.ndarray): Snowmelt per time step [mm / time step], shape (n_time,).

        temperature (np.ndarray): Air temperature time series [°C], shape (n_time,).
        precipitation (np.ndarray): Total precipitation time series [mm / time step], shape (n_time,).

        PET (np.ndarray): Potential evapotranspiration time series [mm / time step], shape (n_time,).

        alpha (float): Shape parameter controlling the nonlinear relationship
        between soil moisture and actual evapotranspiration [-].

        initial_w_storage (np.ndarray): Initial water storage in each reservoir
        at the beginning of the simulation [mm], shape (n_reservoir,).

        w_storage_cap (np.ndarray): Maximum storage capacity of each reservoir [mm], shape (n_reservoir,).

        w_residence_time (np.ndarray): Mean residence time of water in each
        reservoir under saturated conditions [time step], shape (n_reservoir,).

    Returns:
        AET (np.ndarray): Actual evapotranspiration time series [mm / time step], shape (n_time,).

        Qss (np.ndarray): Subsurface (baseflow) discharge time series [mm / time step], shape (n_time,).

        Qs (np.ndarray): Surface runoff (overland flow) time series [mm / time step], shape (n_time,).

        Q (np.ndarray): Total discharge time series (Qs + Qss) [mm / time step], shape (n_time,).

        w_storage (np.ndarray): Water storage in each reservoir at each time step [mm], shape (n_time, n_reservoir).

    Notes:
        - The model is evaluated sequentially in time and depends on previous reservoir states.
        - Evapotranspiration, subsurface flow, and surface runoff are computed
          following a priority-based order within each time step.
        - Strict water mass conservation is enforced inside the reservoir solver.
        - All output arrays are newly allocated; input arrays are not modified in place.
    """


    # initialization
    num_data = len(snow_acc)
    num_reservoir = int(len(w_storage_cap))
    # define the loop function in time domain
    if num_reservoir == 1:
        h_loop_func = single_reservoir
    else:
        h_loop_func = multiple_reservoir

    # prepare the container
    AET = np.zeros(num_data)  # actural et
    Qss = np.zeros(num_data)  # discharge from subsurface flow
    Qs = np.zeros(num_data)  # discharge from overland (surface) flow
    Q = np.zeros(num_data)  # total dischagre
    # water storage in reservoir,
    # row prepresent time, from column_0 to column_n represent reservoir from top to bottom
    w_storage = np.zeros((num_data, num_reservoir))

    # start the loop
    current_storage = initial_w_storage  # define the water storage in time t_i-1
    for i in range(1, num_data):
        # how much watere comes in
        w_in = precipitation[i] - snow_acc[i] + snow_melt[i]

        # start the loop
        AET_t, Qss_t, Qs_t, c_w_storage_updated = h_loop_func(w_in, temperature[i],
                                                              PET[i], alpha,
                                                              current_storage, w_storage_cap, w_residence_time)
        # update the current_storage
        current_storage = c_w_storage_updated
        # assign the values
        AET[i] = AET_t
        Qss[i] = Qss_t
        Qs[i] = Qs_t
        Q[i] = Qs_t + Qss_t
        w_storage[i, :] = current_storage

        #if w_in > 0:
            #print(AET_t, Qss_t, Qs_t, ":", c_w_storage_updated)

    return AET, Qss, Qs, Q, w_storage


# Area-weighted aggregation for hydrological modelled results
def area_weight_aggregate(hydro_container, weights):
    """
    Aggregate hydrological variables across two HRUs using area-weighted averaging.

    This function aggregates all hydrological variables in an xarray Dataset across
    the HRU dimension using predefined area weights.

    All variables except "w_storage" are reduced to a single area-weighted value,
    while "w_storage" is preserved with its original HRU dimension.

    This implementation assumes exactly 2 HRUs and will not work correctly for a different number of HRUs.

    Args:
        hydro_container (xr.Dataset): Dataset containing hydrological variables for each HRU.
        all variables are expected to have an "HRU_id" dimension, except "w_storage",
        which additionally contains a "reservoir_id" dimension.

        weights (array-like of float): Area weights for HRU aggregation.
            Must contain exactly two elements:
                - weights[0]: Area fraction of HRU_id = 0
                - weights[1]: Area fraction of HRU_id = 1
            Weights should sum to 1.0.

    Returns:
        xr.Dataset: Aggregated hydrological dataset where:
        - All variables except "w_storage" are area-weighted across HRUs and no longer vary by HRU.
        - "w_storage" is preserved without aggregation and retains the "HRU_id" dimension.

    Notes:
        - The function assumes the HRU dimension is named "HRU_id".
        - No validation is performed to check the length or sum of "weights".
        - Input dataset variables are not modified in place.
    """


    # weights[0] for bedrock, weights[1] for foret

    # except "w_storage", all other varilables will be aggregated
    vars_to_aggregat = []
    for var in hydro_container.data_vars:
        if var != 'w_storage':
            vars_to_aggregat.append(var)

    aggregated_data = {}
    for var in vars_to_aggregat:
        # weighted sum across HRU_id dimension
        aggregated_data[var] = (
                hydro_container[var].isel(HRU_id=0) * weights[0] +
                hydro_container[var].isel(HRU_id=1) * weights[1]
        )

    # Keep w_storage as is (with HRU_id dimension)
    aggregated_data['w_storage'] = hydro_container['w_storage']

    # Create new dataset
    hydro_output = xr.Dataset(
        data_vars=aggregated_data,
        coords={
            'time': hydro_container.coords['time'],
            'time_str': hydro_container.coords['time_str'],
            'reservoir_id': hydro_container.coords['reservoir_id'],
            'HRU_id': hydro_container.coords['HRU_id'],  # Keep for w_storage
        },
        attrs=hydro_container.attrs
    )

    return hydro_output
