#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-24
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


file1 = f"{project_root}/data/SedCas_output/Hydro_new.txt"
file2 = f"{project_root}/data/SedCas_output/Hydro_original.txt"

# file1 = f"{project_root}/data/SedCas_output/Sediment_new.txt"
# file2 = f"{project_root}/data/SedCas_output/Sediment_original.txt"
#

same = True
with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    # Skip the header line
    next(f1)
    next(f2)

    for lineno, (line1, line2) in enumerate(zip(f1, f2), start=2):  # start=2 since first line is header
        if line1 != line2:
            same = False
            print(f"Files differ at line {lineno}:")
            print(f"  {file1}: {line1.rstrip()}")
            print(f"  {file2}: {line2.rstrip()}")

    # Check if files have different number of lines
    if same:
        extra_line1 = f1.readline()
        extra_line2 = f2.readline()
        if extra_line1 or extra_line2:
            same = False
            print("Files differ in length.")
            if extra_line1:
                print(f"  {file1} has extra line: {extra_line1.rstrip()}")
            if extra_line2:
                print(f"  {file2} has extra line: {extra_line2.rstrip()}")

if same:
    print("Files are exactly the same")
else:
    print("Files are different")
