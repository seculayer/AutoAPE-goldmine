# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import tensorflow as tf

from hps.utils.common.Common import Common

# class : LearnResultCallback
class LearnResultCallback(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        tf.keras.callbacks.Callback.__init__(self)

        self.result_list = list()
        self.global_sn = kwargs["global_sn"]
        self.LOGGER = Common.LOGGER.getLogger()

    def on_epoch_end(self, epoch, logs=None):
        result = logs
        result["step"] = epoch + 1
        result["global_sn"] = self.global_sn
        self.LOGGER.info(result)
        self.result_list.append(result)

    def get_result(self):
        return self.result_list