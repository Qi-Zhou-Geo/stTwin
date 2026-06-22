#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-22T09:31:17
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import argparse
import numpy as np

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
from func.liveshow.t2s3_dash_baord import create_app
from func.toolkit.logger_printer import setup_logger


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    
    parser.add_argument("--output_dir", type=str, default=f"{project_root}/deploy/liveshow_cache/logs")
    parser.add_argument("--log_filename", type=str, default="t2_main_logs.txt")
    args = parser.parse_args()
    

    # setup logger
    logger = setup_logger(args.output_dir, args.log_filename, force_reset=False)
    
    # run app
    app = create_app()
    
    # host="127.0.0.1"
    # Binds the app only to localhost (same machine access only)

    # port=8050
    # Local service port

    # debug=False
    # Disable Flask debug mode for stability in production
    
    app.run(host=args.host, port=args.port, debug=False)