#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-08-07T11:08:36
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import logging
import argparse

import time
import schedule

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
from func.toolkit.logger_printer import setup_logger

def run_pipeline(logger, num_iteration):
    
    try:
        is_new_data, msg = request_latest_10min_data(station="mve", time_resolution="10 minutes")
    except Exception as e:
        is_new_data = False
        msg = f"<request_latest_10min_data> failed:\n {e}"
    logger.info(msg)


    if is_new_data is False:
        # no new data
        pass
    else:
        # find new data
        try:
            simulate(num_iteration=num_iteration)
        except Exception as e:
            msg = f"<simulate> failed:\n {e}"
            logger.info(msg)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--output_dir", type=str, default=f"{project_root}/deploy/liveshow_cache/logs")
    parser.add_argument("--log_filename", type=str, default="t1_main.log")
    parser.add_argument("--num_iteration", type=int, default=10)
    args = parser.parse_args()
    

    # setuo logger
    logger = setup_logger(args.output_dir, args.log_filename, force_reset=False)
    
    # run it immediately
    run_pipeline(logger, args.num_iteration)
    
    # repeat every 10 minutes
    schedule.every(10).minutes.do(run_pipeline, logger, args.num_iteration)

    
    while True:
        schedule.run_pending()
        time.sleep(5) # sleep 5 seconds, then check schedule
        
        msg = f"<t1-main> is sleeping.\n"
        logger.info(msg)