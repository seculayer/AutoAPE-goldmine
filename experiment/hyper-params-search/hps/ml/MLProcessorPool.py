# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import threading
import multiprocessing
import os

from typing import List
from tensorflow.python.client import device_lib
from hps.common.Constants import Constants
from hps.ml.MLPoolingProcessor import MLPoolingProcessor
from hps.common.Common import Common

# class : MLProcessorPool
class MLProcessorPool(threading.Thread):
    MEM_LIMIT = 1024

    def __init__(self, job_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue,
                 alg_nm: str, max_proc_num: int = 0):
        threading.Thread.__init__(self)
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.alg_nm = alg_nm
        self.processes = self._create_processes(max_proc_num)
        self.LOGGER = Common.LOGGER.getLogger()
        self._n_processes = int()

    def initialize(self, dataset):
        for proc in self.processes:
            proc.initialize(dataset)

    def _create_processes(self, max_proc_num: int) -> List[MLPoolingProcessor]:
        proc_list: List[MLPoolingProcessor] = list()

        # only one process
        if max_proc_num == 1:
            # get device index
            device_idx = "-1"
            if Constants.DEVICE_MODE.lower() == "gpu":
                device_idx = "0"

            proc = self._create_proc(device_idx, int(Constants.DEVICE_MEM))
            proc_list.append(proc)

        # dynamic process
        elif max_proc_num == 0:
            for i in range(int(Constants.NUM_DEVICES)):
                for j in range(int(int(Constants.DEVICE_MEM) / self.MEM_LIMIT)):
                    proc = self._create_proc(str(i), self.MEM_LIMIT)
                    proc_list.append(proc)

        return proc_list

    def _create_proc(self, device_idx, mem_limit):
        return MLPoolingProcessor(job_queue=self.job_queue, result_queue=self.result_queue,
                                  device_idx=device_idx, mem_limit=mem_limit, alg_nm=self.alg_nm)

    def run(self) -> None:
        for proc in self.processes:
            proc.start()

        for proc in self.processes:
            proc.join()
    def check_processes(self):
        pid_list = list()
        for proc in self.processes:
            pid_list.append(proc.pid)
        return pid_list

    def terminate(self):
        for i in range(len(self.processes)):
            self.job_queue.put(None)


if __name__ == '__main__':
    job_q = multiprocessing.Queue()
    res_q = multiprocessing.Queue()
    pool = MLProcessorPool(job_q, res_q)
    pool.initialize(list())
    pool.start()
    pool.join()
    print(device_lib.list_local_devices())
