# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import sys

from hps.utils.Common import Common

if __name__ == '__main__':
    LOGGER = Common.LOGGER.getLogger()
    if len(sys.argv) < 2:
        print("usage : python3.5 -m hps.main [param.json]")
    else :
        param_json_nm = sys.argv[1]
        LOGGER.info("Hyper-Parameter Search Start...")



