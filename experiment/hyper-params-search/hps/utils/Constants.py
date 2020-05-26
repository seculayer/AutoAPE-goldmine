# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import os

from hps.utils.Singleton import Singleton
from hps.utils.Configurations import Configurations

# class : Constants
class Constants(metaclass=Singleton):
    __FILE_REAL_PATH = os.path.dirname(os.path.realpath(__file__))

    ### early stop
    EARLY_TYPE_NONE = "none"
    EARLY_TYPE_MIN = "min"
    EARLY_TYPE_MAX = "max"
    EARLY_TYPE_VAR = "var"

    ### default config
    DEFAULT = Configurations(config_path=__FILE_REAL_PATH+"/../conf/default.conf")

    ### DIR SETTING
    DIR_DATA = __FILE_REAL_PATH + "/../.." + DEFAULT.get("DIR_CONFIG", "DIR_DATA")

if __name__ == '__main__':
    print(Constants.DIR_DATA)