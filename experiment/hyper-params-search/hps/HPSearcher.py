# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import json
import multiprocessing
import sys
import time
from typing import List

from hps.algorithms.HPOptimizerFactory import HPOptimizerFactory
from hps.common.Common import Common
from hps.common.Constants import Constants
from hps.dataset.DataLoader import DataLoader
from hps.ml.MLProcessorPool import MLProcessorPool


# class : HPSearcher
class HPSearcher(object):
    def __init__(self, param_json_nm):
        self.LOGGER = Common.LOGGER.getLogger()
        f = open(Constants.DIR_PARAMS + "/" + param_json_nm, "r")
        param_str = f.read()
        f.close()
        self.hps_param_dict: dict = json.loads(param_str)
        self.LOGGER.info(self.hps_param_dict)

        self.data_queue = multiprocessing.Queue()
        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.data_loader = DataLoader(self.hps_param_dict.get("dataset"), self.data_queue)
        self.pool = MLProcessorPool(self.job_queue, self.result_queue,
                                    self.hps_param_dict.get("ml_alg"))
        self.LOGGER.info("Hyper-Parameter Search initialized...")

    def run(self):
        self.LOGGER.info("Hyper-Parameter Search Start...")
        # --- prepare
        ml_dataset = self._get_data()
        self.LOGGER.info("Hyper-Parameter Search.... data load complete!")
        self.pool.initialize(ml_dataset)
        self.pool.start()

        # --- HPO Algorithm
        hpo_algorithm = HPOptimizerFactory.create(self.hps_param_dict,
                                                  self.job_queue,
                                                  self.result_queue)
        hpo_algorithm.optimize()
        self.pool.terminate()
        self.pool.join()

    def _get_data(self) -> List:
        self.data_loader.start()
        dataset: List = self.data_queue.get()
        self.data_loader.join()
        return dataset


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage : python3.5 -m hps.main [param.json]")
    else:
        param_json_filename = sys.argv[1]
        hp_searcher = HPSearcher(param_json_filename)
        hp_searcher.run()
        time.sleep(1)
