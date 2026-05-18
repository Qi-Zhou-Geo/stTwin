#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-29
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import numpy as np

# region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# endregion

from func.liveshow.t2s2_dash_baord import app


if __name__ == "__main__":

    app.run(host="127.0.0.1", port=8050, debug=True)