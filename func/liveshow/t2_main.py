#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

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
from func.liveshow.t2s2_dash_baord import app


if __name__ == "__main__":
    # host="127.0.0.1"
    # Binds the app only to localhost (same machine access only)

    # port=8050
    # Local service port

    # debug=False
    # Disable Flask debug mode for stability in production
    
    app.run(host="127.0.0.1", port=8050, debug=False)