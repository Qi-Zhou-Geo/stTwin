#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2026-02-13
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

from obspy import UTCDateTime

def round_time(t):

    # t is UTCTimeDate, t_str is str time
    if isinstance(t, UTCDateTime) is False:
        t = UTCDateTime(t)
    else:
        pass

    if t.minute >= 30:
        t_updated = UTCDateTime(t.year, t.month, t.day, t.hour, minute=0, second=0) + 3600 # unit is second
    else:
        t_updated = UTCDateTime(t.year, t.month, t.day, t.hour, minute=0, second=0)

    t_str = t_updated.strftime("%Y-%m-%dT%H:%M:%S")

    return t_str