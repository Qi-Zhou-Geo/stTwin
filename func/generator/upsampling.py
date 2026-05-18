#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-20
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd


import numpy as np
from scipy.ndimage import gaussian_filter1d


# upsampler for precipitation
def upsampler_prcp(daily_total, n_samples=144, start_idx=None, duration=None, seed=None, 
                   fluctuation_min=0.5, fluctuation_max=1.5):
    
    rng = np.random.default_rng(seed)
    
    # (1) define the precipitation event length, duration 
    # 3 * 10 minutes <= duration <= 144 * 10 minutes
    if duration is None:
        duration = rng.integers(3, n_samples + 1)
    
    # (2) define the precipitation start time, start_idx, 
    # 0 <= start_idx <= (1440 - duration)
    # Note: all precipitation must fall within the current day
    if start_idx is None:
        start_idx = rng.integers(0, n_samples - duration + 1)
        
    # (3) boundary Check: Ensure the event doesn't exceed 144 indices
    end_idx = min(start_idx + duration, n_samples)
    actual_duration = end_idx - start_idx
    
    # (4) generate uniform random noise for the rain window
    # make the precipitation "noisy", not flat for the entire period
    fluctuation = rng.uniform(fluctuation_min, fluctuation_max, size=actual_duration)
    
    # (5) scale to match daily_total
    # this ensures the sum of the 10-minute blocks equals the daily input
    rain_event = fluctuation * (daily_total / np.sum(fluctuation))
    
    # (6) place into the day time series
    upsampled_data = np.zeros(n_samples)
    upsampled_data[start_idx:end_idx] = rain_event

    # (7) double check
    record = f"""daily_total= {daily_total} != np.sum(upsampled_data) = {np.sum(upsampled_data)}"""
    actual_sum = np.sum(upsampled_data)
    if not np.isclose(daily_total, actual_sum):
        raise AssertionError(f"{record}")

    return upsampled_data

def upsampler_temp(daily_mean, n_samples=144, seed=None, mu=0, 
                   sigma1=3, sigma2=3, coldest_hour=6, warmest_hour=15):
    
    rng = np.random.default_rng(seed)
    
    # (1) generate noise from stand normal distribution N(mu=0, sigma=scale=3)
    fluctuation = rng.normal(loc=mu, scale=sigma1, size=n_samples)
    
    # (2) smooth the noise, for 10-min data, 
    # sigma=3 -> looking the nighbors of around 30 minutes
    fluctuation = gaussian_filter1d(fluctuation, sigma=sigma2)
    
    # (4) force the <np.sum(noise_mean) = 0>, 
    # so we do change the daily_mean
    noise_mean = fluctuation - np.mean(fluctuation)
    
    # (5) add the <noise mean = 0> to the daily_mean
    raw_data = daily_mean + noise_mean # add noise to mean temperature
    
    
    
    # (6) reorder the data
    max_idx = warmest_hour * 6 # from hour idx (0-24) to 10-minutes idx (0-144)
    min_idx = coldest_hour * 6
    
    # (7) create an empty template
    template = np.zeros(len(raw_data))
    
    # (8) temperature increasing period
    # this period is from "coldest_hour (early morning)" to "warmest_hour (afternoon)"
    # simulated by a half sine wave (from -0.5 * pi to 0.5 * pi)
    rising_len = max_idx - min_idx
    template[min_idx:max_idx] = np.sin( np.linspace(-np.pi/2, np.pi/2, rising_len) )
    
    # (9) temperature decreasing period
    # this period is from "warmest_hour (today's afternoon)" to "coldest_hour (next early morning)"
    # simulated by a half sine wave (from 0.5 * pi to 1.5 * pi)
    falling_indices = np.concatenate(
        [np.arange(max_idx, 144), # "warmest_hour (today's afternoon)"
         np.arange(0, min_idx)]) # "coldest_hour (next early morning)"
    falling_len = len(falling_indices)
    template[falling_indices] = np.sin(np.linspace(np.pi/2, 1.5 * np.pi, falling_len))
    
    # (10) reordering
    # get the idx (from min to max) of template
    order = np.argsort(np.argsort(template))
    # sort the raw data from min to max
    sorted_data = np.sort(raw_data)
    # mapping the smllest data to the min_idx
    upsampled_data = sorted_data[order]
    
    # (11) double check
    record = f"""daily_mean = {daily_mean} != np.mean(upsampled_data) = {np.mean(upsampled_data)}"""
    actual_mean = np.mean(upsampled_data)
    if not np.isclose(daily_mean, actual_mean):
        raise AssertionError(f"{record}")

    return upsampled_data

def upsampler_radi(daily_mean, n_samples=144, seed=None, mu=0, 
                   sigma1=3, sigma2=3, sunrise_hour=6, sunset_hour=18, max_radi_hour=13):
    
    rng = np.random.default_rng(seed)
    
    # (0) create an empty template
    template = np.zeros(n_samples)
    
    sunrise_idx = sunrise_hour * 6 # from hour idx (0-24) to 10-minutes idx (0-144)
    sunset_idx = sunset_hour * 6
    peak_idx = max_radi_hour * 6
    num_radi = sunset_idx - sunrise_idx
    
    # (1) generate noise from stand normal distribution N(mu=0, sigma=scale=3)
    fluctuation = rng.normal(loc=mu, scale=sigma1, size=num_radi)

    # (2) smooth the noise, for 10-min data, 
    # sigma=3 -> looking the nighbors of around 30 minutes
    fluctuation = gaussian_filter1d(fluctuation, sigma=sigma2)
    
    # (4) force the np.mean(raw_data[sunrise_idx:sunset_idx]) same as daily_mean
    weights = np.maximum(1.0 + fluctuation, 0.01)
    weights = weights / np.mean(weights)
    
    # (5) add the <noise mean = 0> to the daily_mean
    raw_data = daily_mean * weights # Note: we use the * not +
    

    # (6) radiation increasing period
    # this period is from "sunrise_hour (early morning)" to "max_radi_hour"
    # simulated by a half sine wave (from 0 * pi to 0.5 * pi)
    rising_len = peak_idx - sunrise_idx
    template[sunrise_idx:peak_idx] = np.sin(np.linspace(0, np.pi/2, rising_len))
    
    # (7) radiation decreasing period
    # this period is from "max_radi_hour" to "sunset_hour (late afternoon)"
    # simulated by a half sine wave (from 0.5 * pi to 1 * pi)
    falling_len = sunset_idx - peak_idx
    template[peak_idx:sunset_idx] = np.sin(np.linspace(np.pi/2, np.pi, falling_len))
    
    # (8) reordering
    order = np.argsort(np.argsort(template[sunrise_idx:sunset_idx]))
    sorted_day_data = np.sort(raw_data)
    
    # Place it into the final array
    upsampled_data = np.zeros(n_samples)
    upsampled_data[sunrise_idx:sunset_idx] = sorted_day_data[order]
    # make sure the radiation is not negative
    upsampled_data = np.maximum(upsampled_data, 0)
    
    # (9) double check
    record = f"""daily_mean = {daily_mean} != np.mean(upsampled_data[upsampled_data>0]) = {np.mean(upsampled_data[upsampled_data>0])}"""
    actual_mean = np.mean(upsampled_data[sunrise_idx:sunset_idx])
    if not np.isclose(daily_mean, actual_mean):
        raise AssertionError(f"{record}")

    return upsampled_data


