#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-24
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
#__note__ = This code is adapted from SedCas (Author: Jacob Hirschberg, Created: 2022-02-03, Source: https://github.com/jacobhirschberg/SedCas)
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
        if dt[i] == pd.to_timedelta(
                '1 hour'):  # this should mean, that when using daily data, the values are not added
            dfsnew[i - 1] = dfsnew[i - 1] + dfsnew[i]  # add to the previous hour
            dfsnew[i] = 0
    idxdfs = idxdfs[dfsnew > 0]
    dfsnew = dfsnew[dfsnew > 0]

    # insert values in full modelling time series
    dfs = pd.Series(np.zeros(len(idx)), index=idx)
    dfs.loc[idxdfs] = dfsnew
    dfs = dfs.values

    return dfs, conc

# sediment transfer model
def trans_model(large_ls_t, small_ls_t, hyd, Q_theta, s_max, d_h, hs_theta, area, method, ls_trigger,
                rainfall_triggered_ls_theta, initial_hs_storage=0, initial_ch_storage=0,
                **kwargs):
    '''
    Sediment cascade from hillslope to channel to outlet.
    Note: inactive storage not considered yet, but it's not really needed...

    Args:
        large_ls_t : time series of sum of large landslides
        small_ls_t : time series of sum of small landslides
        hydro : discharge from hydro model [mm]
        Q_theta : critical discharge for triggering of debris flow [mm/t?]
        s_max : maximim potential volumetric ration of sediment (with density of bedrock) to water in the flow
        d_h : redopsition rate from hillslope to channel [-]
        hs_theta : hillslope storage capacity [mm]
        area : catchment area [km2]
        method : method for sediment transport, ['lin' 'exp']
        ls_trigger: landslide trigger mechiam
        rainfall_triggered_ls_theta: Precipitation threshold for rainfall-landslides to be triggered
        initial_hs_storage : initial hillslope storage [mm]
        initial_ch_storage : initial channel storage [mm]

        kwargs : depending on the method
            if method is 'lin' semdiment transport starts when a critical discharge is exceeded:
                no additional inputs required (only Q_theta)
            if method is 'exp' sediment transport is follows discharge in an exponential relationship:
                (a : scaling parameter , determined automatically)
                b : shape parameter
            mindf : if given, sediment output will also be given in terms of debris flows [mm]
            smax_nodf : max sediement concentration for sub-critical flow conditions

    Returns:
        sed : data frame containing...
    sh : hillslope storage time series [mm]
    sc : channel storage time series [mm]
    so : catchment sediment output time series [mm]
    sopot : potential sediment output based on discharge [mm]
    dfs : debris flows, sediment output above minimum threshold and concentration of debris flows[mm]
    conc : sediment concentration in flow [-]
    '''

    q = hyd.Qs.copy()
    snow = hyd.snow_depth.copy()

    # check if given kwargs are valid
    valid_kwargs = ['b', 'mindf', 'smax_nodf']
    for key in kwargs.keys():
        if not key in valid_kwargs:
            raise AttributeError('%s is not a valid property.' % key)

    # check if required arguments are povided, else raise error
    if method == 'exp':
        if not ('a' and 'b' and 'smax_nodf' and 'mindf') in kwargs.keys():
            raise AttributeError('for method "exp" keyword arguments a and b must be provided.')

    # unpack kwargs
    def unpack_kwargs(kwargs, key):
        try:
            var = kwargs[key]
        except KeyError:
            var = np.nan
        return var

    b = unpack_kwargs(kwargs, 'b') # Shape parameter for bedload transport
    mindf = unpack_kwargs(kwargs, 'mindf')
    smax_nodf = unpack_kwargs(kwargs, 'smax_nodf')

    # determine 'a' and 'Qmin_nondf'
    # this is based on two facts
    # 1) the sediment concentration for sub-critical bedload transport cannot exceed the concentration given by smax_nodf
    # 2) the volume of the sediment transported cannot exceed the minimal debris-flow solid volume
    if method == 'exp':
        Qmin_nodf = Q_theta - (mindf * (1 - smax_nodf) / (smax_nodf * Q_theta)) ** (1 / (1 - b))
        if Qmin_nodf < 0:
            Qmin_nodf = 0
        a = smax_nodf * Q_theta / ((Q_theta - Qmin_nodf) ** b * (1 - smax_nodf))

    # landslides (ls) are daily, needs to be padded
    freq = q.index[1] - q.index[0]  # desired frequency
    # print(f"Simulated data sampling Freq.: {freq}")

    delta = pd.to_timedelta('1 day') - freq
    dates = large_ls_t.index.values.copy()
    dates[-1] = dates[-1] + delta

    if 'datetime' in str(large_ls_t.index.dtype):
        dates = pd.to_datetime(dates)
    elif 'timedelta' in str(large_ls_t.index.dtype):
        dates = pd.to_timedelta(dates, unit='h')
    else:
        raise AttributeError('Your input index must be of type "timedelta" or "datetime"')

    if (ls_trigger == 'thermal') or (ls_trigger == 'random'):
        # the ls is daily SPS now
        ls = large_ls_t.mag.copy() + small_ls_t.mag.copy()

        ls.index = dates  # the last value is now the same as for the sub-daily time-series, needed for padding
        #ls = ls.resample(freq).pad()  # this is the time series at desired frequency
        ls = ls.resample(freq).ffill() # this is the time series at desired frequency

        if 'datetime' in str(ls.index.dtype):
            cond = ls.index.time == pd.to_datetime('12:00').time()  # hillslope failure always happen at noon
            ls[~cond] = 0  # set the other hours to 0
        elif 'timedelta' in str(ls.index.dtype):
            cond1 = ls.index.astype('timedelta64[h]').astype('int64') % 12 == 0
            cond2 = ls.index.astype('timedelta64[h]').astype('int64') / 12 % 2 == 1
            ls[~(cond1 & cond2)] = 0

    elif ls_trigger == 'rainfall':
        ### NOT SURE THIS IS GENERIC FOR ALL TEMPORAL RESOLUTIONS AND DATA TYPES
        Prc = hyd.Pr.groupby(pd.Grouper(freq='24h')).cumsum()  # daily cumsum
        Prc[Prc <= rainfall_triggered_ls_theta] = np.nan  # set all smaller than the triggering threshold to nan

        daily_min = Prc.groupby(pd.Grouper(freq='24h')).min()  # this is the minimum of each day
        daily_min.index = dates
        daily_min = daily_min.resample(freq).pad()  # assign min for every modeling time step

        diff = Prc - daily_min
        cond1 = diff == 0  # where the difference to the min is 0, is the first time the precipitation exceeds the triggering threshold on that day
        dfcond = pd.DataFrame(data=cond1)
        dfcond.columns = ['cond1']
        dfcond['cond2'] = np.nan
        dfcond['cond2'][1:] = dfcond.cond1[:-1]
        dfcond.cond2.iloc[0] = False
        cond1 = dfcond.cond1.values
        cond2 = dfcond.cond2.values

        large_ls_t.index = dates
        large_ls_t = large_ls_t.resample(freq).pad()
        large_ls_t[~cond1] = 0  # where the difference is not 0
        large_ls_t[cond2] = 0  # where the the difference is 0, but the previous one is already tagged

        small_ls_t.index = dates
        small_ls_t = small_ls_t.resample(freq).pad()
        if 'datetime' in str(small_ls_t.index.dtype):
            cond = small_ls_t.index.time == pd.to_datetime('12:00').time()  # hillslope failure always happen at noon
            small_ls_t[~cond] = 0  # set the other hours to 0
        elif 'timedelta' in str(small_ls_t.index.dtype):
            cond1 = small_ls_t.index.astype('timedelta64[h]').astype('int64') % 12 == 0
            cond2 = small_ls_t.index.astype('timedelta64[h]').astype('int64') / 12 % 2 == 1
            small_ls_t[~(cond1 & cond2)] = 0

        ls = large_ls_t.mag.copy() + small_ls_t.mag.copy()

    # test if too long
    i = np.argwhere(ls.index == q.index[-1])[0][0]
    ls = ls[:i + 1]

    # convert to arrays if Data Frames or Series
    try:
        idx = ls.index
        ls = ls.values
        q = q.values
        snow = snow.values
    except AttributeError:
        pass

    # initialize
    num_t = len(q)  # length of time series
    sh, sc, so, sopot = np.zeros(num_t), np.zeros(num_t), np.zeros(num_t), np.zeros(num_t)  # initialize output arrays

    # initial conditions
    sh[0] = initial_hs_storage
    sc[0] = initial_ch_storage

    # parameters for large landslides distribution, from Bennett et al. (2012)
    xmin = 233  # Minimum landslide volume from the power-law tail
    alpha = 1.65  # Power law scaling exponent in landslide distribution

    for i in range(1, num_t):

        ## TRANSFER NUMERICAL SCHEME: outputs at t depend on storage states at t,
        # i.e. first the storages, considering inputs, are recomputed, then the debris flow triggering is computed.

        # hillslope -> channel transfer
        # concept: from each landslide a fraction defined by the d_h parameter is redeposited in the channel storage,
        # the rest of the landslide goes directly into the channel
        # (1) like the landslids (materials) trapped on the slope
        re_deposition = ls[i] * d_h  # redeposition according to redeposition rate
        sh[i] = sh[i - 1] + re_deposition  # hillslope storage change
        # (2) except the landslids (materials) trapped on the slope, the rest of materies will go to the channel
        sc[i] = sc[i - 1] + ls[i] - re_deposition  # channel storage change

        if sh[i] > hs_theta:
            # if this conditional is Ture, it means:
            # the hillslope has accumulated more sediment than it can stably store, then ->
            # (1) a secondary landslide / remobilization will be triggered,
            # (2) the hillslope storage "sh[i]" and channel storage "sc[i]" will be updated with "ls_sh"
            ls_sh = hs_theta * 2 # 2 does not contain any physical meaning, it just makes the condition (ls_sh >= hs_theta) is True
            while ls_sh >= hs_theta:
                ls_sh = randht(1, 'xmin', xmin, 'powerlaw', alpha)[0]

                ls_sh = ls_sh / area * 1e-3  # convert m3 to mm

            sh[i] = sh[i] - ls_sh
            sc[i] = sc[i] + ls_sh

        # channel -> outlet transfer
        # there are two methods for sediment entrainment: lin and exp...
        # Note
        # Q_theta is equivalent to the critical shear stress
        # q-qdf is equivalent to the excess shear stress
        if method == 'lin':
            # # (1) If runoff exceeds a threshold, Qdf, a debris flow is triggered
            if q[i] >= Q_theta:
                sopot[i] = s_max / (1 - s_max) * q[i]  # - Q_theta) # potential sediment output

        if method == 'exp':
            if (q[i] < Q_theta) and (q[i] >= Qmin_nodf):
                sopot[i] = a * (q[i] - Qmin_nodf) ** b  # exponential function for small flows
            elif q[i] >= Q_theta:
                sopot[i] = s_max / (1 - s_max) * q[i]  # linear function for debris flows (flows above Q_theta)

        # # the sediment flow might not be initiated because:
        # case 1: there is snow --> no DF
        if snow[i] > 0:
            so[i] = 0
            sopot[i] = 0
        # case 2: transport limited, channel storage is big enough
        elif sc[i] >= sopot[i]:
            so[i] = sopot[i]
            sc[i] = sc[i] - so[i]
        # case 3: supply limited, channel storage is too small
        else:
            so[i] = sc[i]
            sc[i] = 0

    ############

    # output is debris flow when
    # 1) the volume is larger than mindf, and
    # 2) the sediment concentration is larger than smax_nodf
    if ('mindf') in kwargs.keys():
        df, conc = get_dfs(q, so, mindf, smax_nodf, idx)
        dfp, concp = get_dfs(q, sopot, mindf, smax_nodf, idx)

    # output
    data = {'ls': ls,
            'hillslope_storage': sh,
            'channel_storage': sc,
            'sed_output_catchment': so,  # catchment sediment output time series [mm]
            'sed_output_catchment_q': sopot,  # potential sediment output based on discharge [mm]
            'dfs': df,
            'dfspot': dfp} # potential debris flows events

    sed = pd.DataFrame(data=data, index=idx)

    return sed
