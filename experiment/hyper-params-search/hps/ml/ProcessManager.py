# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import threading, time
from multiprocessing import Queue

from hps.common.Constants import Constants
from hps.common.Common import Common
from hps.ml.MLProcessor import MLProcessor

# class : ProcessManager
class ProcessManager(threading.Thread):
    def __init__(self, dataset_nm):
        threading.Thread.__init__(self)

        self.dataset_nm = dataset_nm

        self.proc_list = dict()
        self.queue = Queue()
        self.result_list = list()

        ### GPU 장비 할당
        self.device_list = list()
        if Constants.DEVICE_MODE.lower() == "gpu":
            self.device_list.append(int(Constants.DEVICE_MEM))
        self.MEM_LIMIT = 2048

    def append(self, hash_val, ml_alg, param_dict):
        proc = MLProcessor(hash_val, ml_alg, param_dict, self.dataset_nm, self.queue)
        self.proc_list[hash_val] = {"proc" : proc , "status" : "init"}

    def _device_allocate(self, mem_limit=1024):
        if Constants.DEVICE_MODE.lower() == "gpu":
            for idx, device in enumerate(self.device_list):
                if mem_limit < self.device_list[idx]:
                    self.device_list[idx] -= mem_limit
                    return idx
            return None
        else:
            return -1

    def _device_free(self, idx, mem_limit):
        if Constants.DEVICE_MODE.lower() == "gpu":
            self.device_list[idx] += mem_limit

    def _subproc_start(self, hash_val):
        device = self._device_allocate(mem_limit=self.MEM_LIMIT)
        if device is not None:
            proc_dict = self.proc_list.get(hash_val)
            proc = proc_dict.get("proc")
            proc.set_devices(device, self.MEM_LIMIT)
            proc.start()
            proc_dict["status"] = "start"
            proc_dict["device"] = device

    def _sub_proc_join(self, hash_val):
        proc_dict = self.proc_list.get(hash_val)
        proc_dict.get("proc").join()
        proc_dict["status"] = "end"
        self._device_free(proc_dict["device"], self.MEM_LIMIT)

    def run(self):
        while True:
            ### starting process
            for hash_val in self.proc_list.keys():
                if self.proc_list[hash_val]["status"] == "init":
                    self._subproc_start(hash_val)

            ### get results
            if not self.queue.empty():
                results = self.queue.get()
                self.result_list.append(results)
                self._sub_proc_join(results.get("hash_value"))

            ### exit check
            complete = 0
            for hash_val in self.proc_list.keys():
                if self.proc_list.get(hash_val)["status"] == "end":
                    complete += 1

            if complete == len(self.proc_list.keys()):
                break
            time.sleep(1)

    def get_results(self):
        return self.result_list

if __name__ == '__main__':
    manager = ProcessManager("MNIST")
    manager.start()