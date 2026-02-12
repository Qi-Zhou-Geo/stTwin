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


# random landslides from heavy-tailed distribution
def randht(n, *varargin, seed='none'):
    '''
    RANDHT generates n observations distributed as some continous heavy-tailed distribution.
    Options are power law, log-normal, stretched exponential, power law with cutoff, and exponential.
    Can specify lower cutoff, if desired.
    ---------------------
    
    Input
    -----
    n : generate n observations
    *args:
        xmin : 
        Type : type of distribution as string
            - PL : pwerlaw, reqires ls_alpha_v
            - PC : cutoff, requires ls_alpha_v and Lambda (?)
            - EX : exponential, requires Lambda
            - LN : log-normal, requires mu and sigma
            - ST : stretched, requires Lambda and beta
            
    seed : initialize random number generator; for reproducibility; by default it is 'none', which means new randomness each time, else set a number.
    
    Output
    ------
    x : 
    
    Details
    -------
    original source : http://www.santafe.edu/~aaronc/powerlaws/
    Ported to python by Joel Ornstein (2011 August), joel_ornstein@hmc.edu
    '''

    # update variables
    #    v = {'Type': '',
    #         'xmin': 1,
    #         'ls_alpha_v': 2.5,
    #         'beta': 1,
    #         'Lambda': 1,
    #         'mu': 1,
    #         'sigma': 1}
    #
    #    for key, value in kwargs.items():
    #        if key not in v.keys():
    #            raise NameError('Invalid keyword argument input.')
    #
    #    v.update(kwargs)
    #    globals().update(v)
    #
    #
    #    if n<1:
    #        raise AttributeError('(RANDHT) Error: invalid ''n'' argument.')
    #
    #    if xmin < 1:
    #        raise AttributeError('(RANDHT) Error: invalid ''xmin'' argument.')

    Type = ''
    xmin = 1
    ls_alpha_v = 2.5
    beta = 1
    Lambda = 1
    mu = 1
    sigma = 1

    # parse command-line parameters; trap for bad input
    i = 0
    while i < len(varargin):
        argok = 1
        if type(varargin[i]) == str:
            if varargin[i] == 'xmin':
                xmin = varargin[i + 1]
                i = i + 1
            elif varargin[i] == 'powerlaw':
                Type = 'PL'
                ls_alpha_v = varargin[i + 1]
                i = i + 1
            elif varargin[i] == 'cutoff':
                Type = 'PC'
                ls_alpha_v = varargin[i + 1]
                Lambda = varargin[i + 2]
                i = i + 2
            elif varargin[i] == 'exponential':
                Type = 'EX'
                Lambda = varargin[i + 1]
                i = i + 1
            elif varargin[i] == 'lognormal':
                Type = 'LN'
                mu = varargin[i + 1]
                sigma = varargin[i + 2]
                i = i + 2
            elif varargin[i] == 'stretched':
                Type = 'ST'
                Lambda = varargin[i + 1]
                beta = varargin[i + 2]
                i = i + 2
            else:
                argok = 0

        if not argok:
            print('(RANDHT) Ignoring invalid argument')  # ' ,i+1

        i = i + 1

    if n < 1:
        print('(RANDHT) Error: invalid ''n'' argument; using default.\n')
        n = 10000

    if xmin < 1:
        print('(RANDHT) Error: invalid ''xmin'' argument; using default.\n')
        xmin = 1

    # methods

    random.seed(seed)  #### SET THE SEED

    x = []
    if Type == 'EX':
        x = []
        for i in range(n):
            x.append(xmin - (1. / Lambda) * math.log(1 - random.random()))
    elif Type == 'LN':
        y = []
        for i in range(10 * n):
            y.append(math.exp(mu + sigma * random.normalvariate(0, 1)))

        while True:
            y = filter(lambda X: X >= xmin, y)
            q = len(y) - n
            if np.isclose(q, 0.):
                break

            if q > 0.:
                r = range(len(y))
                random.shuffle(r)
                ytemp = []
                for j in range(len(y)):
                    if j not in r[0:q]:
                        ytemp.append(y[j])
                y = ytemp
                break
            if (q < 0.):
                for j in range(10 * n):
                    y.append(math.exp(mu + sigma * random.normalvariate(0, 1)))

        x = y

    elif Type == 'ST':
        x = []
        for i in range(n):
            x.append(pow(pow(xmin, beta) - (1. / Lambda) * math.log(1. - random.random()), (1. / beta)))
    elif Type == 'PC':

        x = []
        y = []
        for i in range(10 * n):
            y.append(xmin - (1. / Lambda) * math.log(1. - random.random()))
        while True:
            ytemp = []

            for i in range(10 * n):
                if random() < pow(y[i] / float(xmin), -ls_alpha_v):
                    ytemp.append(y[i])

            y = ytemp
            x = x + y
            q = len(x) - n
            if np.isclose(q, 0.):
                break

            if (q > 0):
                r = range(len(x))
                random.shuffle(r)

                xtemp = []
                for j in range(len(x)):
                    if j not in r[0:q]:
                        xtemp.append(x[j])
                x = xtemp
                break

            if (q < 0.):
                y = []
                for j in range(10 * n):
                    y.append(xmin - (1. / Lambda) * math.log(1. - random.random()))


    else:
        x = []
        for i in range(n):
            x.append(
                xmin * pow(1. - random.random(), -1. / (ls_alpha_v - 1.)))  # random.random() is uniform distribution [0,1]

    return x


# probabilistic hillslope erosion

def generate_large_ls(ls_trigger,
                      temperature, prec, snow,
                      theta_sd, theta_prec, theta_sa,
                      theta_ls_freeze,
                      ls_min_v, ls_alpha_v, cutoff,
                      area=1e6,
                      seed=0,
                      max_attempts=5000):
    '''
    Generation of large landslides by thermal trigger (procedure 1 in Bennett et al., 2014).
    Parameters taken from Bennett et al. (2012/13) are not altered.

    Args:
        temperature : temperature [degreeC]
        prec : Precipitation [mm]
        snow : data frame from degree-day-model [mm SWE]
        theta_sd : threshold snowdepth for landslides to be triggered [mm SWE]
        theta_prec : threshold liquid precipitation for landslides to be triggered [mm]
        theta_sa : snow temperature accumulation threshold [°C]
        xmin :
        ls_alpha_v : power law scaling exponent in landslide distribution
        cutoff :
        theta_ls_freeze :
        ls_trigger : Landslide triggering mechanism ['thermal', 'rainfall', 'random']
        area : catchment area [km2], default is 10^6, i.e. output can also be interpreted in [m3]
        seed : initialize random number generator

    Returns:
        large_ls : time series of large landslides, [m3] if area not provided, else [mm]
    '''

    # 'rainfall' have not been used, if you need it, please add it from: https://github.com/jacobhirschberg/SedCas/blob/main/modules.py
    if ls_trigger == 'thermal':
        # temperature and snow have to be resampled to daily mean
        temperature_daily = temperature.resample('24h').mean()

        # temperature_daily_1 = temperature_daily.shift(1) # this is T of 1 day before
        idx = temperature_daily.index
        temperature_daily = temperature_daily.values

        # temperature_daily_1 = temperature_daily_1.values
        snow_day = snow.resample('24h').mean()
        snow_day = snow_day.values

        # a large landslide is triggered when
        # (1) temperature is subfreezing, T < 0 $^\circ$C
        # (2) the day before was not freezing # (?? really ?? QZ) this seems not include
        # (3) and the snow depth is below threshold
        cond1 = temperature_daily < theta_ls_freeze  # freezing days
        # cond2 = temperature_daily_1 > 0            # positive T days
        cond3 = snow_day < theta_sd  # days of only little snow
        # lsdays = cond1 & cond2 & cond3             # boolean array with days of possible landslides
        ls_days = cond1 & cond3

        num_ls = np.sum(ls_days)  # number of big lansdslides, False -> 0, Ture -> 1
        large_ls = np.zeros(len(temperature_daily))
    else:
        print(f"Error! please check the ls_trigger={ls_trigger}")

    # generate "num_ls" large landslide magnitudes (volume, m3).
    # iteration is needed in order to avoid unreasonable large volumes, greater than "cutoff"
    for attempt in range(max_attempts):
        # mags is a list
        mags = randht(num_ls, 'xmin', ls_min_v, 'powerlaw', ls_alpha_v, seed=seed)
        if max(mags) < cutoff:
            break
        seed = seed + 1e4
    else:
        raise RuntimeError("Failed to sample landslide magnitudes below cutoff")

    # output
    large_ls[ls_days] = mags # assign the sampled large landsides to the time domain

    # Note: This step converts landslide volume [m^3] to an equivalent mean thickness (or depth) [mm] over the given area.
    #
    # Physical meaning:
    # Dividing landslide volume (large_ls, m^3) by area (area, km^2) gives the area-normalized landslide thickness.
    #
    # Unit conversion:
    #   area [km^2] = 1e6 m^2
    #   m^3 / m^2 = m
    #   m -> mm = 1e3
    # Therefore:
    #   m^3 / (km^2) * 1e-3 = mm
    #
    # The factor 1e-3 combines both conversions: (1 / 1e6) * 1e3 = 1e-3

    large_ls = large_ls / (area * 1e6)  # return area-normalized landslide thickness, unit: m
    large_ls = large_ls * 1e3  # convert m to mm

    data = {'mag': large_ls} # magnitude[mm]
    large_ls = pd.DataFrame(data, index=idx)

    return large_ls


def generate_small_ls(num_days, num_large_ls, ls_min_v,
                      area=1e6,
                      seed=0,
                      mu=3.36, sigma=1.18, ratio=3.36):
    '''
    Generation of small landslides (procedure 1 in Bennett et al., 2014).
    Parameters taken from Bennett et al. (2012/13) are not altered.

    Args:
        num_days: length of time series, number of days as integer
        num_large_ls: number of large landslides, because the number of small landslides comes from a ratio
        ls_min_v: Minimum landslide volume from the power-law tail
        area: catchment area, unit by m^2
        seed: initilaize random state of the generator, defualt=None, else set a number
        mu: Mean of lognormal distribution
        sigma: Standard deviation of lognormal distribution
        ratio: 3.36 is the average ratio of small to large failures

    Returns:
        small_ls  : time series of small land slides, [m3] if area not provided, else [mm]
    '''

    np.random.seed(seed)

    # generate spacing of small LS
    num_small_ls = int(ratio * num_large_ls)  # number of small LS according to ratio of small/large LS

    dt = int(num_days / num_small_ls)  # mean tempormal spacing between small landslides, like delta in st[0].stats.delta
    dt_exp = np.random.exponential(dt, num_small_ls)
    dt_exp = np.ceil(dt_exp)  # get full days, in this case max one per day
    ids = np.cumsum(dt_exp)

    if max(ids) >= num_days:
        # if the days go beyond the time series length,
        # rescale to length of time series // (t-2) to ensure max value after ceiling in next line as well
        nids = ids / max(ids) * (num_days - 2)
        nids = np.ceil(nids)
        ids = nids

    # generate small landslides
    # draw many from random distribution, should represent the theoretical distribution
    mags_theo = np.random.lognormal(mu, sigma, size=int(1e6))
    mags_cond = mags_theo[mags_theo <= ls_min_v]  # represents theoretical distribution but constrained by "ls_min_v"
    mags = np.random.choice(mags_cond, num_small_ls)  # n samples from constrained distribution

    # output
    ids = [int(i) for i in ids]
    small_ls = np.zeros(num_days)  # initialize days
    small_ls[ids] = mags

    small_ls = small_ls / (area * 1e6)  # return area-normalized landslide thickness, unit: m
    small_ls = small_ls * 1e3  # convert m to mm

    data = {'mag': small_ls} # magnitude[mm]
    small_ls = pd.DataFrame(data=data)

    return small_ls
