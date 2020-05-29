# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 


import multiprocessing

from hps.common.Constants import Constants
from hps.common.Common import Common
from hps.ml.MLAlgorithmFactory import MLAlgorithmFactory
from hps.dataset.DatasetFactory import DatasetFactory
from hps.utils.TensorFlowUtils import TensorFlowUtils

# class : MLProcessor
class MLProcessor(multiprocessing.Process):
    def __init__(self, hash_value, algorithm_name, param_dict, dataset_nm, queue, gpu_idx=-1, mem_limit=1024):
        multiprocessing.Process.__init__(self)
        self.LOGGER = Common.LOGGER.getLogger()

        # ml
        self.hash_value = hash_value
        self.algorithm_name = algorithm_name
        self.param_dict = param_dict
        self.dataset_nm = dataset_nm

        # multi process
        self.queue = queue

        # device
        self.gpu_idx = gpu_idx
        self.mem_limit = mem_limit

    def run(self):
        self.LOGGER.info("MLProcessing Running")
        TensorFlowUtils.device_memory_limit(self.gpu_idx, self.mem_limit)

        # ml algorithm
        algorithm = MLAlgorithmFactory.create(self.algorithm_name, self.param_dict)
        algorithm.build()

        # get dataset
        ds_train, ds_test = DatasetFactory.create(self.dataset_nm)

        # learning
        results = algorithm.learn(dataset=ds_train, verbose=0)
        self.queue.put({"results" : results, "hash_value" : self.hash_value})

if __name__ == '__main__':
    parameters = {
        ## model parameters
        "model_nm" : "DNN-test",
        "algorithm_type" : "classifier",
        "job_type" : "learn",
        ## learning parameters
        "global_step" : "10",
        "early_type": "none",
        "min_step": "10",
        "early_key": "accuracy",
        "early_value": "0.98",
        ## algorithm parameters
        "input_units": "784",
        "output_units": "10",
        "hidden_units" : "10, 20, 10",
        "dropout_prob" : "0.1",
        "optimizer_fn" : "Adam",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "act_fn" : "tanh",
    }


    result_queue = multiprocessing.Queue()
    proc = MLProcessor(1, "DNN", parameters, "MNIST", result_queue)
    proc.start()
    print(result_queue.get())
    proc.join()

