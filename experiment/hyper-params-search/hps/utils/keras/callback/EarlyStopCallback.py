# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import tensorflow as tf
import numpy as np

from hps.utils.Constants import Constants
from hps.utils.Common import Common

# class : EarlyStopCallback
class EarlyStopCallback(tf.keras.callbacks.Callback):
    def __init__(self, learn_params):
        super(EarlyStopCallback, self).__init__()

        self.learn_params = learn_params
        self.AI_LOGGER = Common.LOGGER.getLogger()

        self.prev_val = np.inf
        self.early_steps = 0
        self.stopped_epoch = 0

    def stop_train(self, epoch):
        self.stopped_epoch = epoch
        self.model.stop_training = True
        self.AI_LOGGER.info("------ EARLY STOP !!!!! -----")

    def get_stopped_epoch(self):
        return self.stopped_epoch

    def on_epoch_end(self, epoch, logs=None):
        if not self.learn_params["early_type"] == Constants.EARLY_TYPE_NONE:
            key = self.learn_params["early_key"]
            if self.learn_params["early_type"] == Constants.EARLY_TYPE_MIN:
                if self.learn_params["minsteps"] < epoch:
                    if logs[key] < self.learn_params["early_value"]:
                        self.stop_train(epoch)
                        return

            elif self.learn_params["early_type"] == Constants.EARLY_TYPE_MAX:
                if self.learn_params["minsteps"] < logs["step"]:
                    if logs[key] > self.learn_params["early_value"]:
                        self.stop_train(epoch)
                        return

            elif self.learn_params["early_type"] == Constants.EARLY_TYPE_VAR:
                try:
                    if abs(logs[key] - self.prev_val) < self.learn_params["early_value"]:
                        self.early_steps += 1
                    else:
                        self.early_steps = 0
                except:
                    pass

                if self.early_steps >= self.learn_params["minsteps"]:
                    self.stop_train(epoch)
                    return

        self.stopped_epoch = epoch
        return
