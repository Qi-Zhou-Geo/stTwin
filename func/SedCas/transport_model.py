#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-03T11:15:22
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# __note__ = This code is adapted from SedCas (Author: Jacob Hirschberg, Created: 2022-02-03, Source: https://github.com/jacobhirschberg/SedCas)
#           and is distributed under the terms of the GNU General Public License v3.0 (GPL-3.0).


import pandas as pd
import numpy as np

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
from func.SedCas.sediment_model import truncated_powerlaw_sampler
from func.toolkit.physical_unit_converter import unit_converter


def check_ls_erosion(ls, catchment_area, num_year, ls_unit, ref_erosion_rate=0.39, ref_std=0.03):
    # ref_erosion_rate unit by m per year from https://doi.org/10.1002/esp.3263

    if ls_unit in ["mm"]:
        # "area-weighted"
        ls = unit_converter(input=ls, catchment_area=catchment_area, method="area-aggregated")
    elif ls_unit in ["m^3", "m**3"]:
        pass
    else:
        print(f"Unknown ls_unit {ls_unit}.")

    # return mean_erosion_rate with unit by m/year
    mean_erosion_rate = np.sum(ls) / (catchment_area * 1e6 * num_year) # catchment_area * 1e6: km^2 -> m^2)

    if ref_erosion_rate - ref_std <= mean_erosion_rate <= ref_erosion_rate + ref_std:
        pass
        # print(f"All good!\n"
        #       f"{mean_erosion_rate} m/year, or {mean_erosion_rate * 1e3} mm/year")
    else:
        msg = (f"Warning!\n"
               f"total ls input: {np.sum(ls)}\n"
               f"mean_erosion_rate: {mean_erosion_rate} m/year, or {mean_erosion_rate * 1e3} mm/year")
        print(msg)

def define_debris_flow(Qs, sed_transport, min_df_v, max_s_c):
    """
    Identify debris-flow events from surface discharge (Qs) and sediment transport (sed_transport) time series.

    A debris flow is defined when:
    (1) sediment transport exceeds a minimum debris-flow volume (min_df_v), and
    (2) sediment concentration exceeds the maximum concentration (max_s_c) for fluvial transport.

    Consecutive time steps that satisfy both conditions are merged into a single debris-flow event,
    with the total volume assigned to the event start time.

    Args:
        Qs (pd.Series): Discharge time series [mm per time step], indexed by pandas DatetimeIndex.
        sed_transport (pd.Series): Realized sediment transport time series [mm].
        min_df_v (float): Minimum sediment volume required to form a debris flow [mm].
        max_s_c (float): Maximum sediment concentration for non-debris-flow (bedload) transport [-].

    Returns:
        dfs (np.ndarray): Debris-flow event time series [mm], same length as Qs.
                          Non-zero values indicate debris-flow initiation times.
        conc (np.ndarray): Sediment concentration time series [-].
    """

    idx = Qs.index
    desired_freq = idx[1] - idx[0]

    # sediment concentration
    Qs_copy = Qs.copy()
    Qs_copy[Qs_copy == 0] = np.nan # avoid division by zero
    sed_concentration = sed_transport / (sed_transport + Qs_copy)

    # debris-flow conditions:
    # (1) sufficient sediment volume
    cond1 = sed_transport >= min_df_v
    # (2) exceeds fluvial concentration limit
    cond2 = sed_concentration > max_s_c

    dfs_candidates = sed_transport[cond1 & cond2]
    idx_candidates = idx[cond1 & cond2]

    # merge consecutive debris-flow timesteps into single events
    dt = idx_candidates[1:] - idx_candidates[:-1]
    dt = dt.insert(0, pd.NaT)

    dfs_merged = dfs_candidates.copy()
    for i in range(len(dt) - 1, 0, -1):
        if dt[i] == desired_freq:
            # if two events are close to each other -> merge them
            dfs_merged.iloc[i - 1] = dfs_merged.iloc[i - 1] + dfs_merged.iloc[i]
            dfs_merged.iloc[i] = 0

    # keep only event start times
    idx_events = idx_candidates[dfs_merged > 0]
    dfs_events = dfs_merged[dfs_merged > 0]

    # insert into full time series
    dfs = pd.Series(0.0, index=idx)
    dfs.loc[idx_events] = dfs_events

    return dfs.values, sed_concentration.values

def df_transport_with_entrainment(Qs, erosion_k, channel_storage, channel_storage_cap, max_s2w):
    """
    Compute debris-flow sediment transport including channel entrainment.

    This function estimates theoretical sediment discharge under debris-flow
    conditions by scaling the supplied sediment flux with an entrainment
    factor and converting it using a maximum sediment-to-water volumetric concentration ratio.

    The entrainment factor increases transport as a function of relative
    channel storage, representing enhanced erosion and bulking during debris-flow propagation.

    Args:
        Qs (float): Incoming sediment discharge, unit by [mm per time step].
        erosion_k (float): Dimensionless entrainment coefficient controlling sensitivity of erosion to stored sediment.
        channel_storage (float): Current sediment volume stored in the channel, unit by [mm per time step].
        channel_storage_cap (float): Maximum channel sediment storage capacity, unit by [mm].
        max_s2w (float): Maximum volumetric sediment-to-water concentration ratio (dimensionless, 0 < max_s2w < 1).

    Returns:
        sed_transport_theory (float): Theoretical debris-flow sediment discharge
        including entrainment effects [mm per time step].

    Notes:
        - The formulation assumes that sediment concentration approaches a maximum volumetric limit defined by 'max_s2w'.
        - No explicit cap is applied to prevent exceeding the physical concentration limit; this should be enforced externally if required.
        - The entrainment factor is defined as:
              1 + erosion_k * (channel_storage / channel_storage_cap)
    """

    entrainment_factor = 1 + erosion_k * (channel_storage / channel_storage_cap)  # entrainment factor
    sed_transport_theory = (max_s2w / (1 - max_s2w)) * Qs * entrainment_factor

    return sed_transport_theory

def df_transport_without_entrainment(Qs, erosion_k, channel_storage, channel_storage_cap, max_s2w):

    sed_transport_theory = (max_s2w / (1 - max_s2w)) * Qs

    return sed_transport_theory


def sediment_transport_model(ls, h2s_r, Qs, modelled_SWE,
                             initial_hs_storage, initial_ch_storage,
                             hillslope_storage_cap, channel_storage_cap, erosion_k,
                             ls_min_v, ls_alpha_v, ls_max_v, c_area,
                             bedload_param_a, bedload_param_b, max_s2w,
                             Qbl, Qdf, df_transport_model):

    """
    Simulate (1) sediment transfer from hillslopes to channels and
    (2) from channels to the catchment outlet.

    The model represents two coupled processes:
    (1) hillslope-to-channel sediment transfer, including temporary hillslope storage and
    stochastic remobilization when storage capacity is exceeded;
    (2) channel-to-outlet sediment transport by bedload or debris flow,
    limited by sediment availability in channel storage.

    Transport mechanisms are controlled by two discharge thresholds, Qbl and Qdf:
    # 0     ->      Qbl  ->   Qdf    ->   infinity
    # | no transport | bedload | debris flow |

    Args:
        ls (float): Landslide sediment input at the current time step [mm, area-normalized sediment thickness].

        h2s_r (float): Fraction of landslide material temporarily retained on the hillslope [-].
        The remaining fraction (1 - h2s_r) enters the channel directly.

        Qs (float): Surface discharge at the current time step [same units as Qbl and Qdf].

        modelled_SWE (float): Modelled snow water equivalent on the ground [mm SWE].
        Bedload transport is suppressed when SWE > 0.

        initial_hs_storage (float): Hillslope sediment storage at the previous time step [mm].
        initial_ch_storage (float): Channel sediment storage at the previous time step [mm].
        hillslope_storage_cap (float): Maximum stable sediment storage on the hillslope [mm].

        ls_min_v (float): Minimum landslide volume parameter used for stochastic remobilization [m³ before area normalization].
        ls_alpha_v (float): Power-law exponent controlling the landslide size distribution [-].
        c_area (float): Catchment area [km²]. Used to convert landslide volumes to area-normalized sediment thickness.

        bedload_param_a (float): Scale parameter of the bedload transport rating curve.
        bedload_param_b (float): Shape parameter of the bedload transport rating curve [-].
        max_s2w (float): Maximum sediment-to-water volumetric ratio for debris flows [-].
        Qbl (float): Discharge threshold for initiation of bedload transport [mm / <time_resolution>].
        Qdf (float): Discharge threshold for initiation of debris-flow transport [mm / <time_resolution>].

    Returns:
        ls_remobilize (float): Sediment remobilized from hillslope storage during the current time step [mm].

        hillslope_storage (float): Updated hillslope sediment storage after deposition and remobilization [mm].
        channel_storage (float): Updated channel sediment storage after hillslope input and sediment export [mm].

        sed_transport_real (float): Actual sediment exported from the channel
            to the outlet during the current time step, limited by channel storage [mm].
        sed_transport_theory (float): Theoretical sediment exported from the channel
            to the outlet during the current time step, [mm].
        sed_limited (float): Status of sediment limited in channel storage,
            0 (Fasle) represents no sediment limited, 1 (True) represents sediment limited,

    Notes:
        - Hillslope remobilization is triggered only when hillslope storage exceeds its prescribed capacity
            and follows a stochastic power-law distribution.
        - Bedload transport occurs only under snow-free conditions.
        - Sediment export from the channel is supply-limited when channel storage is insufficient to meet transport capacity.
        - Mass conservation is enforced by preventing negative sediment storage.
    """


    ### (1) hillslope -> channel transfer
    # (a) hillslope storage receives: ls * h2s_r,
    #     where h2s_r is the fraction of landslide material temporarily retained on the hillslope
    # (b) channel storage receives: ls * (1 - h2s_r)
    hillslope_storage = initial_hs_storage + ls * h2s_r  # hillslope storage change
    channel_storage = initial_ch_storage + ls * (1 - h2s_r)  # channel storage change

    # remobilize the sediments
    if hillslope_storage > hillslope_storage_cap:

        # the hillslope has accumulated more sediment than it can stably store, then:
        # (1) a secondary landslide / remobilization will be triggered,
        # (2) the hillslope storage and channel storage will be updated

        # * 2 does not contain any physical meaning,
        # it just makes the condition is True
        ls_remobilize = hillslope_storage_cap * 2
        num_ls = 1
        seed = 0
        while ls_remobilize >= hillslope_storage_cap:
            # ls_remobilize = randht(1, 'xmin', ls_min_v, 'powerlaw', ls_alpha_v)[0]

            # return as a signle value
            ls_remobilize = truncated_powerlaw_sampler(num_ls, ls_min_v, ls_max_v, ls_alpha_v, seed=seed)[0]

            ls_remobilize = ls_remobilize / (c_area * 1e6)  # return area-normalized landslide thickness, unit: m
            ls_remobilize = ls_remobilize * 1e3  # convert m to mm

            seed = seed + 1
        # update the storage
        hillslope_storage = hillslope_storage - ls_remobilize
        assert hillslope_storage >= 0, f"Warning! hillslope_storage={hillslope_storage} is negative."
        channel_storage = channel_storage + ls_remobilize
    else:
        # do not need to remobilize
        ls_remobilize = ls

    ### (2) channel -> outlet transfer
    # 0     ->      Qbl  ->   Qdf    ->   infinity
    # | no transport | bedload | debris flow |

    # theoretical sediment transport
    if 0 <= Qs < Qbl:
        # no sediemnts transport in the channel
        sed_transport_theory = 0
    elif Qbl <= Qs < Qdf:
        # bedload transport
        if modelled_SWE > 0:
            # if there is snow, bedload will not be initiated
            sed_transport_theory = 0
        else:
            sed_transport_theory = bedload_param_a * (Qs - Qbl) ** bedload_param_b
    else:
        # debris flow
        sed_transport_theory = df_transport_model(Qs, erosion_k, channel_storage, channel_storage_cap, max_s2w)

    # check the channel_storage and theoretical transport
    if channel_storage >= sed_transport_theory:
        # transport limited
        # channel storage (sediment) is sufficient, but Qs is too small
        sed_transport_real = sed_transport_theory
        sed_limited = 0 # False
    else:
        # supply limited
        # channel storage (sediment) is too small, but Qs is sufficient
        sed_transport_real = channel_storage
        sed_limited = 1  # True

    # update the channel_storage
    channel_storage = channel_storage - sed_transport_real
    assert channel_storage >= 0, f"Channel_storage is negative: {channel_storage}"

    return ls_remobilize, hillslope_storage, channel_storage, sed_transport_real, sed_transport_theory, sed_limited


def redistribute_ls_time(daily_ls, desired_freq, redistribute_method="fixed"):
    """
    Redistribute daily landslide magnitudes onto a sub-daily time grid.

    This function converts a daily landslide time series into a sub-daily
    time series by assigning each day's landslide magnitude to specific
    sub-daily time steps according to a chosen redistribution method.

    Supported redistribution strategies include:
        - fixed: all landslide material is released at a fixed time of day (12:00).
        - uniform: landslide material is distributed evenly across all sub-daily time steps within the day.
        - random: placeholder for future stochastic redistribution (not implemented).

    Args:
        daily_ls (pd.Series or pd.DataFrame): Daily landslide magnitude time series indexed by date.
            Values represent area-normalized landslide thickness [mm per day].
            If a DataFrame is provided, the first column is used as "daily_ls".

        desired_freq (int): Sub-daily temporal resolution in minutes.
            For example, "desired_freq=10" creates a 10-minute time step.

        redistribute_method (str, optional): Method used to redistribute daily landslide magnitudes in time.
            Must be one of {"fixed", "uniform", "random"}. Default is "fixed".

    Returns:
        pd.DataFrame: Sub-daily landslide time series with a DatetimeIndex at the specified resolution.
            The single column contains redistributedlandslide magnitudes with units:
            "Magnitude [mm per area-normalized landslide thickness]".

    Notes:
        - The output time index spans from the start of the first day to the end of the last day (exclusive), with sub-daily resolution.
        - Days with zero landslide magnitude are skipped efficiently.
        - The "random" redistribution method is intentionally left empty for future extension.
        - Mass is conserved: the sum of sub-daily magnitudes equals the original daily magnitude for each day.
    """

    # accept both Series and DataFrame
    if isinstance(daily_ls, pd.DataFrame):
        daily_ls = daily_ls.iloc[:, 0]

    # build sub-daily time axis
    start_time = pd.to_datetime(daily_ls.index[0]).normalize()
    end_time = pd.to_datetime(daily_ls.index[-1]).normalize() + pd.Timedelta("1D")

    q_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{desired_freq}min",  # minutes
        inclusive="left"
    )

    num_data = len(q_index)
    processed_ls = np.zeros(num_data)
    steps_per_day = int(1440 / desired_freq)

    for day, mag in daily_ls.items():
        day = pd.to_datetime(day).normalize() # type: ignore

        if mag == 0:
            continue

        if redistribute_method == "fixed":
            # all landslide happens at noon 12:00
            t = day + pd.Timedelta(hours=12)
            idx = q_index.get_indexer([t])[0]
            if idx >= 0:
                processed_ls[idx] = processed_ls[idx] + mag

        elif redistribute_method == "uniform":
            # distribute uniformly within the day
            mask = (q_index >= day) & (q_index < day + pd.Timedelta("1D"))
            processed_ls[mask] = processed_ls[mask] + mag / steps_per_day

        elif redistribute_method == "random":
            # intentionally empty (future extension)
            pass

        else:
            raise ValueError(f"Unknown redistribute_method: {redistribute_method}")

    processed_ls = pd.DataFrame(
        processed_ls,
        index=q_index,
        columns=["Magnitude [mm per area-normalized landslide thickness]"]
    )

    return processed_ls


def define_bedload_params(Qdf, min_df_v, max_s_c, bedload_param_b):

    """
    Derive bedload transport parameters and
    an effective bedload threshold consistent with a debris-flow triggering discharge.

    This function computes:
    (1) an effective bedload initiation discharge threshold Qbl, and
    (2) the scaling parameter of the bedload transport rating curve,

    such that bedload transport smoothly transitions into debris-flow
    transport at the debris-flow threshold Qdf while respecting a minimum debris-flow sediment volume and concentration.

    The formulation ensures:
    - Bedload sediment concentration remains below debris-flow concentration.
    - The minimum debris-flow solid volume is not exceeded by bedload transport.
    - Continuity of sediment transport at Qdf.

    Args:
        Qdf (float): Critical discharge for debris-flow initiation.
            Units must be consistent with sediment volume formulation (e.g., mm per time step).

        min_df_v (float): Minium debris-flow volume required to classify an event as a debris flow [m^3].

        max_s_c (float): Max possible sediment concentration for bedload [-].

        bedload_param_b (float): Shape exponent of the bedload transport rating curve [-].
            Typical values range from 1.3 to 1.7 for bedload transport.

    Returns:
        Qbl (float): Effective bedload initiation discharge threshold.
            Below this discharge, no sediment transport occurs. Constrained to be non-negative.

        bedload_param_a (float): Scaling parameter of the bedload transport rating curve, defined such that sediment transport is continuous at Qdf.

    Notes:
        - Qbl is not a physically observed threshold
            but a derived transition discharge ensuring consistency between bedload and debris-flow transport regimes.
        - The bedload transport formulation follows a power-law of the form:
            O_bedload = bedload_param_a · (Q - Qbl)^bedload_param_b
        - Debris-flow transport above Qdf is governed by a fixed sediment concentration equal to max_s_c.
    """

    Qbl_theory = Qdf - (min_df_v * (1 - max_s_c) / (max_s_c * Qdf)) ** (1 / (1 - bedload_param_b))
    Qbl = max(Qbl_theory, 0) # make sure the thorshlds not negative

    bedload_param_a = max_s_c * Qdf / ((Qdf - Qbl) ** bedload_param_b * (1 - max_s_c))

    return Qbl, bedload_param_a

# sediment transfer model
def trans_model(large_ls, small_ls, Qs, modelled_SWE,
                desired_freq,
                h2s_r,
                initial_hs_storage, initial_ch_storage,
                hillslope_storage_cap, channel_storage_cap, erosion_k,
                ls_min_v, ls_alpha_v, ls_max_v, c_area,
                bedload_param_a, bedload_param_b, max_s2w,
                Qbl, Qdf, entrainment=False
                ):
    """
    Simulate time-resolved sediment transfer from hillslopes to the catchment outlet
    driven by landslide inputs and discharge-dependent transport capacity.

    The model couples:
    (1) landslide sediment inputs (large and small),
    (2) hillslope and channel sediment storage dynamics, and
    (3) discharge-controlled sediment transport regimes (no transport, bedload, debris flow).

    Daily landslide volumes are first redistributed to a sub-daily time resolution
    and then routed through hillslope and channel storages at each time step.

    Args:
        large_ls (pd.Series or pd.DataFrame): Daily large-landslide sediment input
            [mm, area-normalized sediment thickness].
            Landslides are assumed to occur at 12:00 unless redistributed otherwise.

        small_ls (pd.Series or pd.DataFrame): Daily small-landslide sediment input
            [mm, area-normalized sediment thickness].
            Landslides are assumed to occur at 12:00 unless redistributed otherwise.

        Qs (np.ndarray): Discharge time series at sub-daily resolution [consistent discharge units], shape (n_time,).

        modelled_SWE (np.ndarray): Snow water equivalent time series [mm], shape (n_time,).
            Bedload transport is suppressed when SWE > 0.

        desired_freq (int): Sub-daily time resolution in minutes (e.g., 10 or 60).

        h2s_r (float): Fraction of incoming landslide material temporarily retained on hillslopes [-].

        initial_hs_storage (float): Initial hillslope sediment storage [mm].
        initial_ch_storage (float): Initial channel sediment storage [mm].
        hillslope_storage_cap (float): Maximum stable hillslope sediment storage capacity [mm].
            Excess sediment triggers secondary remobilization.

        ls_min_v (float): Minimum landslide volume used in stochastic remobilization [m³].
        ls_alpha_v (float): Power-law exponent for landslide volume distribution [-].

        c_area (float): Catchment area [km^2], used to convert volumes to area-normalized thickness.

        bedload_param_a (float): Scaling parameter of the bedload transport rating curve.
        bedload_param_b (float): Exponent of the bedload transport rating curve.
        max_s2w (float): Maximum sediment-to-water ratio for debris flows [-].

        Qbl (float): Bedload initiation discharge threshold. No sediment transport occurs when Qs < Qbl.
        Qdf (float): Debris-flow initiation discharge threshold.
            Transport above this threshold follows a concentration-based formulation.

    Returns:
        ls_remobilize (np.ndarray): Remobilized landslide sediment per time step [mm],
            including secondary landslides triggered by hillslope overfilling.

        hillslope_storage (np.ndarray): Hillslope sediment storage time series [mm].
        channel_storage (np.ndarray): Channel sediment storage time series [mm].

        sed_transport_real (np.ndarray): Actual sediment export at the catchment outlet per time step [mm].
        sed_transport_theory (np.ndarray): Theoretical sediment exported from the channel
            to the outlet during the current time step, [mm].
        sed_limited (np.ndarray): Status of sediment limited in channel storage,
            0 (Fasle) represents no sediment limited, 1 (True) represents sediment limited,

    Notes:
        - Landslide inputs are redistributed from daily to sub-daily resolution
            using a fixed-time assumption (12:00 by default).
        - Sediment transport is limited by the minimum of transport capacity
            (discharge-controlled) and available channel storage.
        - Bedload transport is suppressed during snow-covered conditions (SWE > 0).
        - The model is evaluated sequentially in time and depends on previous storage states.
        - All arrays returned are newly allocated; inputs are not modified in place.
    """

    large_ls = redistribute_ls_time(daily_ls=large_ls, desired_freq=desired_freq, redistribute_method="fixed")
    small_ls = redistribute_ls_time(daily_ls=small_ls, desired_freq=desired_freq, redistribute_method="fixed")
    ls = large_ls.iloc[:, 0].values + small_ls.iloc[:, 0].values # type: ignore
    
    
    if len(ls) != len(Qs):
        
        msg = (f"Warning!"
              f"length Qs != ls.\n"
              f"len(Qs) = {len(Qs)}, len(ls) = {len(ls)}")
        msg = (f"<ls> will be slice as same length as <Qs>")
        
        ls = ls[:len(Qs)]
        
    # initialize
    num_data = len(Qs)  # length of time series
    ls_real_input = np.array(ls) # the ls input to the system
    ## when the slope can NOT hold the new coming ls >> landslides will remobilize >> none zero ls_remobilize
    ## when the slope can hold the new coming ls >> landslides will NOT remobilize >> zero ls_remobilize
    ls_remobilize = np.zeros(num_data)
    
    hillslope_storage = np.zeros(num_data)  # hillslope_storage
    ## the change in hillslope_storage is small.
    ## without this adjustment, the plot range becomes too wide and may cause plotting issues.
    hillslope_storage[0] = hillslope_storage_cap
    
    channel_storage = np.zeros(num_data)  # channel_storage
    sed_transport_real = np.zeros(num_data)  # catchment sediment output
    sed_transport_theory = np.zeros(num_data)  # catchment sediment output in theory
    sed_limited = np.zeros(num_data) # whether sediments are limited in channel storage, 0(False), 1(True)

    # select the debris flow transport model
    if entrainment is True:
        df_transport_model = df_transport_with_entrainment
    else:
        df_transport_model = df_transport_without_entrainment

    # hillslope_storage_cap
    current_hs_storage = initial_hs_storage
    current_ch_storage = initial_ch_storage
    for i in range(1, num_data):
        current_ls = ls[i]
        current_Qs = Qs[i]
        current_modelled_SWE = modelled_SWE[i]
        temp = sediment_transport_model(current_ls, h2s_r, current_Qs, current_modelled_SWE,
                                        current_hs_storage, current_ch_storage,
                                        hillslope_storage_cap, channel_storage_cap, erosion_k,
                                        ls_min_v, ls_alpha_v, ls_max_v, c_area,
                                        bedload_param_a, bedload_param_b, max_s2w,
                                        Qbl, Qdf, df_transport_model)

        (updated_ls, hillslope_storage_t, channel_storage_t,
         sed_transport_real_t, sed_transport_theory_t, sed_limited_t) = temp

        # update the current storage
        current_hs_storage = hillslope_storage_t
        current_ch_storage = channel_storage_t

        # add to container
        ls_remobilize[i] = updated_ls
        hillslope_storage[i] = hillslope_storage_t
        channel_storage[i] = channel_storage_t
        sed_transport_real[i] = sed_transport_real_t
        sed_transport_theory[i] = sed_transport_theory_t
        sed_limited[i] = sed_limited_t

    return ls_real_input, ls_remobilize, hillslope_storage, channel_storage, sed_transport_real, sed_transport_theory, sed_limited
