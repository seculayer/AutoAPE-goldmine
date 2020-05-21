# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.utils.Singleton import Singleton
from hps.utils.MPLogger import MPLogger
# class : Common
class Common(metaclass=Singleton):
    LOGGER = MPLogger(log_dir=".", log_name="HyperParamSearch.log", log_level="INFO")