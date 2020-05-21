# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.utils.Singleton import Singleton

# class : Constants
class Constants(metaclass=Singleton):
    ### early stop
    EARLY_TYPE_NONE = "none"
    EARLY_TYPE_MIN = "min"
    EARLY_TYPE_MAX = "max"
    EARLY_TYPE_VAR = "var"
