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

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy

project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.SedCas.sediment_model import randht


def get_dfs(q, s, mindf, smax_nodf, idx):
    q2 = q.copy()
    q2[q2 == 0] = np.nan
    conc = s / (s + q2)  # volumetric sediment concentration in flow

    cond1 = s >= mindf  # first condition, sediment output must be greater than the minimum possibld DF
    cond2 = conc > smax_nodf  # second condition, the sediment concentration must be greater than for fluvial transport

    dfs = s[cond1 & cond2]
    idxdfs = idx[cond1 & cond2]

    dt = idxdfs[1:] - idxdfs[:-1]  # get spacing between debris flows
    dt = dt.insert(0, pd.NaT)  # put a NaT at the first position becuase  it doesn't have a valiue before

    # if there are consecutive values, add them
    dfsnew = dfs.copy()
    for i in range(len(dt) - 1, 0, -1):
        # this should mean, that when using daily data, the values are not added
        if dt[i] == pd.to_timedelta('1 hour'):
            dfsnew[i - 1] = dfsnew[i - 1] + dfsnew[i]  # add to the previous hour
            dfsnew[i] = 0
    idxdfs = idxdfs[dfsnew > 0]
    dfsnew = dfsnew[dfsnew > 0]

    # insert values in full modelling time series
    dfs = pd.Series(np.zeros(len(idx)), index=idx)
    dfs.loc[idxdfs] = dfsnew
    dfs = dfs.values

    return dfs, conc


def sediment_transport_model(ls, h2s_r, Qs, modelled_SWE,
                             initial_hs_storage, initial_ch_storage,
                             hillslope_storage_cap,
                             ls_min_v, ls_alpha_v, c_area,
                             bedload_param_a, bedload_param_b, max_s2w,
                             Qbl, Qdf):
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
        while ls_remobilize >= hillslope_storage_cap:
            ls_remobilize = randht(1, 'xmin', ls_min_v, 'powerlaw', ls_alpha_v)[0]

            ls_remobilize = ls_remobilize / (c_area * 1e6)  # return area-normalized landslide thickness, unit: m
            ls_remobilize = ls_remobilize * 1e3  # convert m to mm

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
        sed_transport_theory = (max_s2w / (1 - max_s2w)) * Qs

    # check the channel_storage and theoretical transport
    if channel_storage >= sed_transport_theory:
        # transport limited
        # channel storage (sediment) is sufficient, but Qs is too small
        sed_transport_real = sed_transport_theory
    else:
        # supply limited
        # channel storage (sediment) is too small, but Qs is sufficient
        sed_transport_real = channel_storage

    # update the channel_storage
    channel_storage = channel_storage - sed_transport_real
    assert channel_storage >= 0, f"Channel_storage is negative: {channel_storage}"

    return ls_remobilize, hillslope_storage, channel_storage, sed_transport_real


def redistribute_ls_time(daily_ls, desired_freq, redistribute_method="fixed"):
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
        day = pd.to_datetime(day).normalize()

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

    Qbl_theory = Qdf - (min_df_v * (1 - max_s_c) / (max_s_c * Qdf)) ** (1 / (1 - bedload_param_b))
    Qbl = max(Qbl_theory, 0) # make sure the thorshlds not negative

    bedload_param_a = max_s_c * Qdf / ((Qdf - Qbl) ** bedload_param_b * (1 - max_s_c))

    return Qbl, bedload_param_a

# sediment transfer model
def trans_model(large_ls, small_ls, Qs, modelled_SWE,
                desired_freq,
                h2s_r,
                initial_hs_storage, initial_ch_storage,
                hillslope_storage_cap,
                ls_min_v, ls_alpha_v, c_area,
                bedload_param_a, bedload_param_b, max_s2w,
                Qbl, Qdf
                ):
    # min_df_v -> mindf, b (Shape parameter for bedload transport) -> scaling_b, smax_nodf ->max_s_c

    # large_ls_t, small_ls_t is daily landslides (sediments) inputs at 12:00:00

    # determine 'a' and 'Qmin_nondf'
    # this is based on two facts
    # 1) the sediment concentration for sub-critical bedload transport
    # cannot exceed the concentration given by smax_nodf
    # 2) the volume of the sediment transported cannot exceed the minimal debris-flow solid volume
    # if method == 'exp':
    #     Qmin_nodf = Qdf - (mindf * (1 - smax_nodf) / (smax_nodf * Qdf)) ** (1 / (1 - b))
    #     if Qmin_nodf < 0:
    #         Qmin_nodf = 0
    #     a = smax_nodf * Qdf / ((Qdf - Qmin_nodf) ** b * (1 - smax_nodf))
    #

    # desired_freq = 10  # unit is minutes
    large_ls = redistribute_ls_time(daily_ls=large_ls, desired_freq=desired_freq, redistribute_method="fixed")
    small_ls = redistribute_ls_time(daily_ls=small_ls, desired_freq=desired_freq, redistribute_method="fixed")

    ls = large_ls.iloc[:, 0].values + small_ls.iloc[:, 0].values
    assert len(ls) == len(Qs), (f"length Qs != ls.\n"
                                f"len(Qs) = {len(Qs)}, len(ls) = {len(ls)}")

    # initialize
    num_data = len(Qs)  # length of time series
    ls_remobilize = np.zeros(num_data) # when the slope can not hold the new coming ls, landslides remobilize
    hillslope_storage = np.zeros(num_data)  # hillslope_storage
    channel_storage = np.zeros(num_data)  # channel_storage
    sed_transport_real = np.zeros(num_data)  # catchment sediment output

    # hillslope_storage_cap
    current_hs_storage = initial_hs_storage
    current_ch_storage = initial_ch_storage
    for i in range(1, num_data):
        current_ls = ls[i]
        current_Qs = Qs[i]
        current_modelled_SWE = modelled_SWE[i]
        temp = sediment_transport_model(current_ls, h2s_r, current_Qs, current_modelled_SWE,
                                        current_hs_storage, current_ch_storage,
                                        hillslope_storage_cap,
                                        ls_min_v, ls_alpha_v, c_area,
                                        bedload_param_a, bedload_param_b, max_s2w,
                                        Qbl, Qdf)

        updated_ls, hillslope_storage_t, channel_storage_t, sed_transport_real_t = temp
        # update the current storage
        current_hs_storage = hillslope_storage_t
        current_ch_storage = channel_storage_t

        # add to container
        ls_remobilize[i] = updated_ls
        hillslope_storage[i] = hillslope_storage_t
        channel_storage[i] = channel_storage_t
        sed_transport_real[i] = sed_transport_real_t

    # # output is debris flow when
    # # 1) the volume is larger than mindf, and
    # # 2) the sediment concentration is larger than smax_nodf
    # if ('mindf') in kwargs.keys():
    #     df, conc = get_dfs(q, so, mindf, smax_nodf, idx)
    #     dfp, concp = get_dfs(q, sopot, mindf, smax_nodf, idx)
    #

    #
    # # output
    # data = {'ls': ls,
    # data = {'ls': ls,
    #         'hillslope_storage': sh,
    #         'channel_storage': sc,
    #         'sed_output_catchment': so,  # catchment sediment output time series [mm]
    #         'sed_output_catchment_q': sopot,  # potential sediment output based on discharge [mm]
    #         'dfs': df,
    #         'dfspot': dfp} # potential debris flows events
    #
    # sed = pd.DataFrame(data=data, index=idx)

    return ls_remobilize, hillslope_storage, channel_storage, sed_transport_real
