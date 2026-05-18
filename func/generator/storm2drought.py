#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-20
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


def storm2drought_generator(cycle_period, num_year, storm_onset, 
                            storm2drought_ratio, data_sampling_freq,
                            leap_year=False, plot=False):
    """
    Generate a binary time series representing alternating
    drought (1) and storm (-1) periods using a shifted sine wave.

    r"y(t) = \sin(2\pi f t) - \sin \left( \frac{\pi(1 - R_{s2d})}{2(1 + R_{s2d})} \right)"

    Args:
        cycle_period (float):
            Cycle frequency (cycles per unit time).
            One cycle corresponds to one storm + drought period.

        num_year (float):
            Total simulation year, unit by year.

        storm2drought_ratio (float):
            Ratio of storm num_year to drought num_year (Ts / Td), must be >= 0.

        data_sampling_freq (float):
            Sampling frequency (samples per unit time).

    Returns:
        t (np.ndarray): Time array of shape (n_samples,).

        y_t (np.ndarray): Discrete signal array of shape (n_samples,),
            where:
            1   -> drought (y >= 0),
            -1  -> storm   (y < 0)
    """

    # (0) check the config
    if leap_year is True:
        num_day = num_year * 366
    else:
        num_day = num_year * 365

    if num_day < 2 * cycle_period:
        raise ValueError(f'num_day must be >= 2 * (1 / cycle_period).\n'
                         f'num_day = num_year * 366 (or 365) = {num_day}', 
                         f'2 * (1 / cycle_period) = {2 * cycle_period}')

    if data_sampling_freq < 10 / cycle_period:
        raise ValueError(
            f"Considing the Nyquist Frequency theory, the current sampling frequency is too low.\n"
            f"Recommended: data_sampling_freq >= 10 * cycle_period\n"
            f"Got: data_sampling_freq={data_sampling_freq} vs cycle_period={cycle_period}\n"
        )

    # (1) construct time vector
    n_samples = int(num_day * data_sampling_freq)

    # (2) generate continuous signal
    t1 = np.linspace(0, storm_onset, storm_onset, endpoint=False)
    y1 = np.full(shape=storm_onset, fill_value=1) # always start from drought status

    # alway generate more t2 and y2
    t2 = np.linspace(0, n_samples * 3, n_samples * 3, endpoint=False)
    y2 = np.sin(2 * np.pi *  (1 / cycle_period) * t2)
    # convert storm-to-drought ratio to vertical shift
    psi = np.sin(
        (np.pi * (1 - storm2drought_ratio)) / 
        (2 * (1 + storm2drought_ratio))
    )
    y2 = np.where(y2 >= - psi, 1, -1) # -1 -> storm, 1 -> drought
    
    # (3) re-selected data from the first storm, where the first "-1"
    t2_selected = np.linspace(storm_onset, num_day, num_day - storm_onset, endpoint=False)
    idx = np.where(y2 == -1)[0][0] # find the first "-1"
    y2_selected = y2[idx:(idx + num_day - storm_onset)]
    

    t = np.append(t1, t2_selected)
    y_t = np.append(y1, y2_selected)

    # (4) threshold to obtain discrete states

    # (5) plot
    if plot is True:
        plot_storm2drought(t, y_t, save=False)

    return t, y_t

def plot_storm2drought(t, y_t, save=False):

    plt.rcParams.update({'font.size': 7,
                         'axes.formatter.limits': (-4, 6),
                         'axes.formatter.use_mathtext': True})

    fig = plt.figure(figsize=(5, 3))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    ax.plot(t, y_t, color='black')
    storm_onset = np.where(y_t==-1)[0][0]
    ax.axvline(x=storm_onset, color="red", ls="--", lw=1, label=f"Storm Onset ({storm_onset})")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Status [y(t)]", fontweight='bold')
    ax.set_yticks([-1, 1], ["Strom (-1)", "Drought (1)"])
    ax.set_xlabel(f"Number of Day [t]", fontweight='bold')
    ax.legend(loc="lower right", fontsize='6')

    if t[-1] == 365:
        ax.set_xticks([1, 50, 100, 150, 200, 250, 300, 350, 365],
                      [1, 50, 100, 150, 200, 250, 300, 350, 365])
    elif t[-1] == 366:
        ax.set_xticks([1, 50, 100, 150, 200, 250, 300, 350, 366],
                      [1, 50, 100, 150, 200, 250, 300, 350, 366])
    else:
        pass

    label = r"y(t) = \sin(2\pi f t) - \sin \left( \frac{\pi(1 - R_{s2d})}{2(1 + R_{s2d})} \right)"

    plt.tight_layout()
    if save is True:
        plt.savefig(f"./storm2drought_example.png", dpi=600)  # , transparent=True
    plt.show()
    plt.close(fig=fig)


def main(ref_data_resolution = "h"):

    cycle_period, num_year, storm_onset = 1 / 120, 1, 121  # every 60 day, start from 1st May, 365 days
    storm2drought_ratio = 0.2  # storm to drought ratio

    if ref_data_resolution == "h":
        data_sampling_freq = 1  # 1 data per day
    elif ref_data_resolution == "t":
        data_sampling_freq = 144  # 144 data per day

    storm2drought_generator(cycle_period, num_year, storm_onset, 
                            storm2drought_ratio, data_sampling_freq,
                            leap_year=False, plot=True)

if __name__ == main:
    main(ref_data_resolution = "h")
