# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import multiprocessing
import os
import datetime

from hps.common.Constants import Constants
from hps.common.Common import Common
from hps.dataset.DatasetFactory import DatasetFactory


# class : DataLoader
class DataLoader(multiprocessing.Process):
    def __init__(self, dataset_nm: str, queue: multiprocessing.Queue):
        multiprocessing.Process.__init__(self)
        self.LOGGER = Common.LOGGER.getLogger()

        self.queue = queue
        self.dataset_nm = dataset_nm

    def run(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.LOGGER.info("DataLoader start")
        start = datetime.datetime.now()
        ds_train, ds_test = DatasetFactory.load(self.dataset_nm, dim=1)
        duration = datetime.datetime.now() - start
        self.LOGGER.info("data load duration : {}".format(str(duration)))

        self.queue.put([ds_train, ds_test])
        self.LOGGER.info("DataLoader end")


if __name__ == '__main__':
    data_q = multiprocessing.Queue()
    data_loader = DataLoader("mnist", data_q)
    data_loader.start()
    data_q.get()
    data_loader.join()
