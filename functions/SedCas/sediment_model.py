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
            - PL : pwerlaw, reqires alpha
            - PC : cutoff, requires alpha and Lambda (?)
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
#         'alpha': 2.5,
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

    Type   = ''
    xmin   = 1
    alpha  = 2.5
    beta   = 1
    Lambda = 1
    mu     = 1
    sigma  = 1


    # parse command-line parameters; trap for bad input
    i=0
    while i<len(varargin): 
        argok = 1
        if type(varargin[i])==str: 
            if varargin[i] == 'xmin':
                xmin = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'powerlaw':
                Type = 'PL'
                alpha  = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'cutoff':
                Type = 'PC'
                alpha  = varargin[i+1]
                Lambda = varargin[i+2]
                i = i + 2
            elif varargin[i] == 'exponential':
                Type = 'EX'
                Lambda = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'lognormal':
                Type = 'LN'
                mu = varargin[i+1]
                sigma = varargin[i+2]
                i = i + 2
            elif varargin[i] == 'stretched':
                Type = 'ST'
                Lambda = varargin[i+1]
                beta = varargin[i+2]
                i = i + 2
            else:
                argok=0
        
      
        if not argok: 
            print('(RANDHT) Ignoring invalid argument') #' ,i+1 
      
        i = i+1 

    if n<1:
        print('(RANDHT) Error: invalid ''n'' argument; using default.\n')
        n = 10000

    if xmin < 1:
        print('(RANDHT) Error: invalid ''xmin'' argument; using default.\n')
        xmin = 1


    # methods
    
    random.seed(seed)   #### SET THE SEED
    
    x=[]
    if Type == 'EX':
        x=[]
        for i in range(n):
            x.append(xmin - (1./Lambda)*math.log(1-random.random()))
    elif Type == 'LN':
        y=[]
        for i in range(10*n):
            y.append(math.exp(mu+sigma*random.normalvariate(0,1)))

        while True:
            y= filter(lambda X:X>=xmin,y)
            q = len(y)-n
            if np.isclose(q, 0.):
                break

            if q>0.:
                r = range(len(y))
                random.shuffle(r)
                ytemp = []
                for j in range(len(y)):
                    if j not in r[0:q]:
                        ytemp.append(y[j])
                y=ytemp
                break
            if (q<0.):
                for j in range(10*n):
                    y.append(math.exp(mu+sigma*random.normalvariate(0,1)))
            
        x = y
        
    elif Type =='ST':
        x=[]
        for i in range(n):
            x.append(pow(pow(xmin,beta) - (1./Lambda)*math.log(1.-random.random()),(1./beta)))
    elif Type == 'PC':
        
        x = []
        y=[]
        for i in range(10*n):
            y.append(xmin - (1./Lambda)*math.log(1.-random.random()))
        while True:
            ytemp=[]
            for i in range(10*n):
                if random()<pow(y[i]/float(xmin),-alpha):ytemp.append(y[i])
            y = ytemp
            x = x+y
            q = len(x)-n
            if np.isclose(q, 0.):
                break

            if (q>0):
                r = range(len(x))
                random.shuffle(r)

                xtemp = []
                for j in range(len(x)):
                    if j not in r[0:q]:
                        xtemp.append(x[j])
                x=xtemp
                break
            
            if (q<0.):
                y=[]
                for j in range(10*n):
                    y.append(xmin - (1./Lambda)*math.log(1.-random.random()))


    else:
        x=[]
        for i in range(n):
            x.append(xmin*pow(1.-random.random(),-1./(alpha-1.))) # random.random() is uniform distribution [0,1]

    return x


# probabilistic hillslope erosion

def generate_large_ls(ls_trigger,
                      temperature, prec, snow,
                      theta_sd, theta_prec, theta_sa,
                      theta_ls_freeze,
                      min_ls_volume, alpha, cutoff,
                      area=1e6, seed=None):
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
        alpha : power law scaling exponent in landslide distribution
        cutoff :
        theta_ls_freeze :
        ls_trigger : Landslide triggering mechanism ['thermal', 'rainfall', 'random']
        area : catchment area [km2], default is 10^6, i.e. output can also be interpreted in [m3]
        seed : initialize random number generator

    Returns:
        large_ls : time series of large land slides, [m3] if area not provided, else [mm]
    '''

    if (ls_trigger == 'thermal') or (ls_trigger == 'random'):
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
        # (2) the day before was not freezing
        # (3) and the snow depth is below threshold
        cond1 = temperature_daily < theta_ls_freeze  # freezing days
        # cond2 = temperature_daily_1 > 0            # positive T days
        cond3 = snow_day < theta_sd                  # days of only little snow
        # lsdays = cond1 & cond2 & cond3             # boolean array with days of possible landslides
        ls_days = cond1 & cond3

        N = len(ls_days[ls_days == True])  # number of big lansdslides

        large_ls = np.zeros(len(temperature_daily))

    elif ls_trigger == 'rainfall':
        Prl = prec.copy()
        Prl[temperature <= theta_sa] = 0  # liquid precipitation

        Prl_day = Prl.resample('24h').sum()  # daily sums
        idx = Prl_day.index
        ls_days = Prl_day > theta_prec
        N = len(ls_days[ls_days == True])

        large_ls = np.zeros(len(Prl_day))

    elif ls_trigger == 'random':
        # N from thermal triggering
        nt = len(temperature_daily)
        dt = int(nt / N)  # mean (?) spacing between small landslides
        dtexp = np.random.exponential(dt, N)
        dtexp = np.ceil(dtexp)  # get full days, in this case max one per day
        ids = np.cumsum(dtexp)
        if max(ids) >= nt:
            # if the days go beyond the time series length,
            # rescale to length of time series // (t-2) to ensure max value after ceiling in next line as well
            nids = ids / max(ids) * (nt - 2)
            nids = np.ceil(nids)
            ids = nids
        ids = [int(i) for i in ids]
        ls_days = ids
    else:
        print(f"Error! please check the ls_trigger={ls_trigger}")

    # generate N large landslide magnitudes (volume, m3).
    # iteration is needed in order to avoid unreasonable large volumes, greater than cutoff
    # this is not effective computation...condition should be in randth
    cond = False
    while not cond:
        mags = randht(N, 'xmin', min_ls_volume, 'powerlaw', alpha, seed=seed)

        cond = max(mags) < float(eval(cutoff))  # cutoff # convert string '3*10**6' to float: 3000000
        if not cond:
            seed = seed + 10000

    # max_attempts = 1000
    # attempt = 0
    # while attempt < max_attempts:
    #     mags = randht(N, 'xmin', min_ls_volume, 'powerlaw', alpha, seed=seed)
    #
    #     if max(mags) < cutoff: # make sure all elements smaller than cutoff
    #         break
    #     seed += 10000
    #     attempt += 1
    # else:
    #     raise ValueError("Could not generate mags below cutoff after 1000 attempts")
    #

    # output
    large_ls[ls_days] = mags
    large_ls = large_ls / area * 10. ** -3  # convert m3 to mm
    data = {'mag': large_ls}

    large_ls = pd.DataFrame(data, index=idx)

    return large_ls

def generate_small_ls(num_t, num_large_ls, min_ls_volume, area=1e6, seed=None, mu=3.36, sigma=1.18, ratio=3.36):
    '''
    Generation of small landslides (procedure 1 in Bennett et al., 2014).
    Parameters taken from Bennett et al. (2012/13) are not altered.

    Args:
        num_t: length of time series, number of days as integer
        num_large_ls: number of large landslides, because the number of small landslides comes from a ratio
        min_ls_volume: Minimum landslide volume from the power-law tail
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
    s_ls = np.zeros(num_t)  # initialize days
    num_small_ls = int(ratio * num_large_ls)  # number of small LS according to ratio of small/large LS
    dt = int(num_t / num_small_ls)  # mean (?) spacing between small landslides
    dtexp = np.random.exponential(dt, num_small_ls)
    dtexp = np.ceil(dtexp)  # get full days, in this case max one per day
    ids = np.cumsum(dtexp)

    if max(ids) >= num_t:
        # if the days go beyond the time series length,
        # rescale to length of time series // (t-2) to ensure max value after ceiling in next line as well
        nids = ids / max(ids) * (num_t - 2)
        nids = np.ceil(nids)
        ids = nids

    # generate small landslides
    # draw many from random distribution, should represent the theoretical distribution
    mags_teo = np.random.lognormal(mu, sigma, size=int(1e6))
    mags_con = mags_teo[mags_teo <= min_ls_volume]  # represents theoretical distribution but constrained by xmin
    mags = np.random.choice(mags_con, num_small_ls)  # n samples from constrained distribution

    # output
    ids = [int(i) for i in ids]
    s_ls[ids] = mags
    s_ls = s_ls / area * 10. ** -3  # convert m3 to mm

    data = {'mag': s_ls}
    small_ls = pd.DataFrame(data=data)

    return small_ls

