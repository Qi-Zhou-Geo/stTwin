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


def storm2drought_generator(t0, Rs2d, cycle_period, 
                            leap_year=False, 
                            drought_value=1, 
                            storm_value=-1, 
                            Rs2d_tolerance=0.01):

    """
    Generate a binary time series representing alternating
    drought (1) and storm (-1) periods using a shifted sine wave.

    num_strom_day + num_drought_day + t0 = num_day
    num_strom_day / num_drought_day - t0 = Rs2d
    
    Args:
        t0 (int or float): first storm onset day, unit by day of year

        Rs2d (float): num_strom_day in whole year, unit by dimensionless

        cycle_period (int or float): total length of one "strom" + "drought" cycle, unit by days
    
    Returns:
        t (np.ndarray): Time array of shape (num_day,).

        y_t (np.ndarray): Discrete signal array of shape (num_day,),
            where:
            1   -> drought (y(t) >= 0),
            -1  -> storm   (y(t) < 0)
    """

    if leap_year is True:
        num_day = 366
    else:
        num_day = 365

    # day of year is from 1, python is from 0
    t0 = t0 - 1

    # number of strom and drought day in one year
    num_storm_day = num_day * (Rs2d / (1 + Rs2d))
    num_storm_day = round(num_storm_day)

    num_drought_day = num_day - num_storm_day
    num_drought_day_after_t0 = num_drought_day - t0

    # assert the logic
    if num_drought_day_after_t0 <= 0:
        raise AssertionError(f"Logic Error!\n"
                             f"<num_drought_day_after_t0>={num_drought_day_after_t0}, it can not <= 0.")

    # num cycle after t0
    num_cycle = (num_day - t0) // cycle_period # floor division
    if num_cycle > num_storm_day or num_cycle == 0:
        raise AssertionError(f"Logic Error!\n"
                             f"<num_cycle>={num_cycle}, it can not greater than <num_storm_day>={num_storm_day}, or be zero.")

    # prepare the last "strom" + "drought" cycle
    num_storm_in_last_cycle = num_storm_day % num_cycle # modulo
    value_in_last_cycle = [storm_value] * num_storm_in_last_cycle

    # normal "strom" + "drought" cycle
    num_storm_in_1_cycle = (num_storm_day - num_storm_in_last_cycle) // num_cycle # floor division
    num_drought_in_1_cycle = cycle_period - num_storm_in_1_cycle
    value_in_1_cycle = [storm_value] * num_storm_in_1_cycle + [drought_value] * num_drought_in_1_cycle

    # assume all time is drought
    t = np.arange(num_day)
    y_t = np.full(shape=num_day, fill_value=drought_value, dtype=int)

    # start the replace value
    for id_c in range(num_cycle + 1): # add one more cycle for "num_storm_in_last_cycle"

        # not the last cycle
        if id_c != num_cycle:
            id1 = t0 + id_c * cycle_period
            id2 = t0 + (id_c + 1) * cycle_period

            y_t[id1:id2] = value_in_1_cycle

        # last cycle
        else:
            
            if num_storm_in_last_cycle == 0:
                # it can be zero like: 4 % 2
                # nothing to check or assign, this is fine
                pass
            else:
                
                id1 = -1 * num_storm_in_last_cycle
                # if these julday occupied by "storm_value"
                if np.any(y_t[id1:] == storm_value):
                    raise AssertionError(f"Logic Error!\n"
                                        f"Strom alread assigned.")
                # they are free to use
                else:
                    y_t[id1:] = value_in_last_cycle

    # check the Rs2d
    num_d = len(np.where(y_t == drought_value)[0])
    num_s = len(np.where(y_t == storm_value)[0])

    if (np.abs(num_s / num_d - Rs2d) > Rs2d_tolerance):
        raise ValueError(f"Warning!\n"
                         f"num_s / num_d - Rs2d > Rs2d_tolerance.\n"
                         f"num_s={num_s}, num_d={num_d}, Rs2d={Rs2d}, Rs2d_tolerance={Rs2d_tolerance}")

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
    ax.legend(loc="best", fontsize='6')

    if t[-1] == 365:
        ax.set_xticks([1, 50, 100, 150, 200, 250, 300, 350, 365],
                      [1, 50, 100, 150, 200, 250, 300, 350, 365]) # type: ignore
    elif t[-1] == 366:
        ax.set_xticks([1, 50, 100, 150, 200, 250, 300, 350, 366],
                      [1, 50, 100, 150, 200, 250, 300, 350, 366]) # type: ignore
    else:
        pass

    label = r"y(t) = \sin(2\pi f t) - \sin \left( \frac{\pi(1 - R_{s2d})}{2(1 + R_{s2d})} \right)"

    plt.tight_layout()
    if save is True:
        plt.savefig(f"./storm2drought_example.png", dpi=600)  # , transparent=True
    plt.show()
    plt.close(fig=fig)


def main(leap_year=False):

    cycle_period = 60  # every 60 day
    storm_onset = 32  # first storm start from 1st Feb., 365 days
    storm2drought_ratio = 0.2  # storm to drought ratio

    storm2drought_generator(
        cycle_period=cycle_period,
        t0=storm_onset,
        Rs2d=storm2drought_ratio,
        leap_year=leap_year
    )

if __name__ == main:
    main()
