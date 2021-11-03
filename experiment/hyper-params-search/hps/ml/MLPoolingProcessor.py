# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import multiprocessing
from typing import List
import tensorflow as tf

from hps.common.Constants import Constants
from hps.common.Common import Common
from hps.ml.MLAlgorithmFactory import MLAlgorithmFactory

from hps.utils.TensorFlowUtils import TensorFlowUtils


# class : MLPoolingProcessor
class MLPoolingProcessor(multiprocessing.Process):
    def __init__(self, job_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue,
                 alg_nm, device_idx="-1", mem_limit=1024):
        multiprocessing.Process.__init__(self)
        self.LOGGER = Common.LOGGER.getLogger()

        self.job_queue = job_queue
        self.result_queue = result_queue
        self.device_idx = device_idx
        self.mem_limit = mem_limit
        self.alg_nm = alg_nm
        self.org_dataset: List = list()
        self.ds_train = None
        self.ds_test = None

    def initialize(self, dataset: List):
        self.org_dataset = dataset

    def run(self) -> None:
        self.LOGGER.info("MLPooling Processor Running!")
        TensorFlowUtils.device_memory_limit(self.device_idx, self.mem_limit)
        # self.ds_train, self.ds_test = self._make_dataset()
        self.ds_train = self.org_dataset[0]
        self.ds_test = self.org_dataset[1]
        self.LOGGER.info("make tensorflow dataset object!")
        self._do_work()
        self.LOGGER.info("MLPooling Processor terminated!")

    def _make_dataset(self) -> [tf.data.Dataset, tf.data.Dataset]:
        ds_train = self.org_dataset[0]
        ds_train = tf.data.Dataset.from_tensor_slices(ds_train)
        # ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(len(self.org_dataset[0]))
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = self.org_dataset[1]
        ds_test = tf.data.Dataset.from_tensor_slices(ds_test)
        # ds_train = ds_train.cache()
        ds_test = ds_test.shuffle(len(self.org_dataset[1]))
        ds_test = ds_test.batch(128)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_test

    def _do_work(self):
        while True:
            job = self.job_queue.get()
            self.LOGGER.info("job in sub process : {}".format(job))
            # ---- terminate
            if job is None:
                break

            temp_hash = job[0]
            param_dict = job[1]
            task = job[2]
            self.LOGGER.info("job : {}".format(job))

            # ---- Machine Learning
            algorithm = MLAlgorithmFactory.create(self.alg_nm, param_dict)
            algorithm.build()
            self.LOGGER.info("Algorithm build....")
            if task == "learn":
                results = algorithm.learn(dataset=self.ds_train, verbose=0)
            elif task == "predict":
                results = algorithm.predict(dataset=self.ds_test)
            else :
                raise NotImplementedError
            # --- result put
            self.result_queue.put(
                {"results": results, "hash_value": str(temp_hash)}
            )
            self.LOGGER.info("insert result ...")
