#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-12-26
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

from filelock import FileLock

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy

project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# endregion

# import the custom functions

def dump_as_row(output_dir, output_name, variable_str, *args):
    '''
    dump the variables to local

    Args:
        output_dir:
        output_name:
        variable_str: pass one as str
        *args: pass any str or float, then conect by ","

    Returns:

    '''

    lock_path = f"{output_dir}/{output_name}.lock"
    # lock the file to avoid the information lost when multiple process
    with FileLock(lock_path):
        with open(f"{output_dir}/{output_name}.txt", "a") as f:

            record = f"{variable_str}"
            if args:
                # append additional arguments
                record += ", " + ", ".join(str(arg) for arg in args)

            f.write(record + "\n")  # Write to file with newline
