#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-12
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    filename="log_print.txt",
    filemode="a"
)

def log_print(msg):

    logging.info(msg)
