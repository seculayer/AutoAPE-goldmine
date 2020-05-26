# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import sys
import json

from hps.common.Constants import Constants
from hps.common.Common import Common
from hps.algorithms.HPOptimizerFactory import HPOptimizerFactory

# class : HPSearcher
class HPSearcher(object):
    def __init__(self, param_json_nm):
        self.LOGGER = Common.LOGGER.getLogger()
        f = open(Constants.DIR_PARAMS + "/" + param_json_nm, "r")
        param_str = f.read()
        f.close()
        self.hps_param_dict = json.loads(param_str)
        self.LOGGER.info(self.hps_param_dict)
        self.LOGGER.info("Hyper-Parameter Search Start...")

    def run(self):
        hpo_algorithm = HPOptimizerFactory.create(self.hps_param_dict)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage : python3.5 -m hps.main [param.json]")
    else :
        param_json_nm = sys.argv[1]
        hp_searcher = HPSearcher(param_json_nm)
        hp_searcher.run()



