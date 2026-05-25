#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import time
import schedule
from obspy import UTCDateTime

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
from func.liveshow.t1s1_fetch_data import request_latest_10min_data
from func.liveshow.t1s2_run_model import simulate

def run_pipeline():
    
    try:
        no_new_data = request_latest_10min_data()
    except Exception as e:
        print(f"{UTCDateTime.now().isoformat()} \n <request_latest_10min_data> failed:\n {e}")


    if no_new_data is True:
        pass
    else:
        try:
            simulate()
        except Exception as e:
            print(f"{UTCDateTime.now().isoformat()} \n <simulate> failed:\n {e}")



if __name__ == "__main__":
    
    # run it immediately
    run_pipeline()
    
    # repeat every 10 minutes
    schedule.every(10).minutes.do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(5) # sleep 5 seconds, then heck schedule
        print(f"{UTCDateTime.now().isoformat()}\n"
              f"<t1-pipeline> is sleeping\n")
