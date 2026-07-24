#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-20T16:44:24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import argparse
import numpy as np


def storm2drought_generator(t0, Rs2d, cycle_period, 
                            leap_year=False, 
                            drought_value=1, 
                            storm_value=-1, 
                            Rs2d_tolerance=0.01,
                            plot=False, save_plot=False):

    """
    Generate a binary storm-to-drought time series consisting of alternating storm and drought periods.

    num_storm_day + num_drought_day = num_day
    num_storm_day / num_drought_day = Rs2d
    
    Args:
        t0 (int or float): first storm onset day, unit by day of year, real calendar julian day (1-indexed),
        where:
            start from 1 (Jan 1st), end at 365 or 366 (Dec 31st),
            e.g., t0=32 means 1st February is the first storm day

        Rs2d (float): storm-to-drought ratio, unit by dimensionless
        where:
            num_storm_day / num_drought_day

        cycle_period (int or float): length of one storm-to-drought cycle, unit by days
        where:
            each cycle contains a contiguous storm period followed by a contiguous drought period.
            
    Returns:
        t (np.ndarray): Day of year (num_day,) real calendar julian day (1-indexed),
            the first day index is 1, 
            the last day index is 365 for non-leap year or 366 for leap year,

        y_t (np.ndarray): time series of storm-to-drought states, shape by (num_day,),
            elements equal to "drought_value=1" indicate drought
            elements equal to "storm_value=-1" indicate storm
    """

    # (1) set the number of day in one year
    if leap_year is True:
        num_day = 366
    else:
        num_day = 365

    # (2) number of storm and drought day in one year
    t0 = t0 - 1 # convert to python 0-indexed 
    num_storm_day = num_day * (Rs2d / (1 + Rs2d))
    num_storm_day = round(num_storm_day)

    num_drought_day = num_day - num_storm_day
    num_drought_day_after_t0 = num_drought_day - t0

    # assert the logic
    if num_drought_day_after_t0 <= 0:
        raise AssertionError(f"Logic Error!\n"
                             f"<num_drought_day_after_t0>={num_drought_day_after_t0}, it can not <= 0.")

    # num cycle after t0
    num_cycle = int((num_day - t0) // cycle_period) # floor division
    if num_cycle > num_storm_day or num_cycle == 0:
        raise AssertionError(f"Logic Error!\n"
                             f"<num_cycle>={num_cycle}, it can not greater than <num_storm_day>={num_storm_day}, or be zero.")

    # (3) prepare the last "storm" + "drought" cycle
    num_storm_in_last_cycle = int(num_storm_day % num_cycle) # modulo
    value_in_last_cycle = [storm_value] * num_storm_in_last_cycle

    # normal "storm" + "drought" cycle
    num_storm_in_1_cycle = int((num_storm_day - num_storm_in_last_cycle) // num_cycle) # floor division
    num_drought_in_1_cycle = cycle_period - num_storm_in_1_cycle
    value_in_1_cycle = [storm_value] * num_storm_in_1_cycle + [drought_value] * num_drought_in_1_cycle

    # (4) assume all time is drought
    t = np.arange(1, num_day + 1) # real calendar julian day (1-indexed),
    y_t = np.full(shape=num_day, fill_value=drought_value, dtype=int)

    # (5) start the replace value
    for id_c in range(num_cycle + 1):

        # not the last cycle >> process all full cycles
        if id_c != num_cycle:
            id1 = t0 + id_c * cycle_period
            id2 = t0 + (id_c + 1) * cycle_period

            y_t[id1:id2] = value_in_1_cycle

        # last cycle >> assign any remaining "storm" days to the end of the year.
        else:
            
            if num_storm_in_last_cycle == 0:
                # int(num_storm_day % num_cycle) can be zero like: 4 % 2
                # nothing to check or assign, this is fine
                print(f"Note! int(num_storm_day % num_cycle)={int(num_storm_day % num_cycle)} can be zero like: 4 % 2")
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


    # (6) plot the sythenstic data
    if plot is True:
        plot_storm2drought(t, y_t, t0, Rs2d, cycle_period, num_s, save=save_plot)
    
    return t, y_t


def plot_storm2drought(t, y_t, t0, Rs2d, cycle_period, num_s, storm_value=-1, save=False):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({'font.size': 7,
                        'axes.formatter.limits': (-4, 6),
                        'axes.formatter.use_mathtext': True})

    if t0 != np.where(y_t==storm_value)[0][0]:
        raise ValueError(f"Warning! t0 is not as excepted.\n"
                         f"t0={t0}\n"
                         f"np.where(y_t==storm_value)[0][0]={np.where(y_t==storm_value)[0][0]}")
    
    label = (f"First Storm Onset (t0={t0})\n"
             f"Strom-to-Drought Ratio (Rs2d={Rs2d})\n"
             f"Cycle Period (1/f={cycle_period})\n"
             f"Num. of Strom Day (num_s={num_s})")
    
    fig = plt.figure(figsize=(5, 3))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    ax.plot(t, y_t, color='black')
    ax.axvline(x=t0, color="red", ls="--", lw=1, label=label)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Status [y(t)]", fontweight='bold')
    ax.set_yticks([-1, 1], ["Strom (-1)", "Drought (1)"])
    ax.set_xlabel(f"Day of Year [t]", fontweight='bold')
    ax.legend(loc="best", fontsize='6')

    
    if t[-1] == 365: # non-leap year
        non_leap_year = np.array([1, 50, 100, 150, 200, 250, 300, 350, 365])
        ax.set_xticks(non_leap_year, non_leap_year) # type: ignore
    elif t[-1] == 366: # leap year
        leap_year = np.array([1, 50, 100, 150, 200, 250, 300, 350, 366])
        ax.set_xticks(leap_year, leap_year) # type: ignore
    else:
        raise ValueError(f"Warning! Got unexcept t[-1]={t[-1]}.")

    label = r"y(t) = \sin(2\pi f t) - \sin \left( \frac{\pi(1 - R_{s2d})}{2(1 + R_{s2d})} \right)"

    plt.tight_layout()
    if save is True:
        plt.savefig(f"./storm2drought_example.png", dpi=600)  # , transparent=True
    
    plt.show()
    plt.close(fig=fig)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    
    # first storm arrives at 2nd Feb., unit by day of year, start from 1 to 365 or 366
    parser.add_argument("--storm_onset", type=int, default=32)
    
    # total storm day / total drought day, 0.05 >> 17 storm days
    parser.add_argument("--storm2drought_ratio", type=float, default=0.05)
    
    # one idea storm-drought cycle
    parser.add_argument("--cycle_period", type=int, default=60)
    
    # leap year or not
    parser.add_argument("--leap_year", type=bool, default=False)
    
    args, _ = parser.parse_known_args()

    t, y_t = storm2drought_generator(
        t0=args.storm_onset,
        Rs2d=args.storm2drought_ratio,
        cycle_period=args.cycle_period,
        leap_year=args.leap_year,
        
        plot=True, 
        save_plot=False
    )

